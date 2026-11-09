"""Physical test of the UART result path: FPGA class -> board PS UART -> USB-UART -> PC.

Control runs over SSH; the results travel ONLY over the USB-UART (COM6). Steps:
  1. new run dir on the board with tools/board/fpga_uart_demo.py, the guard, the
     RAM demo set and the confidence parameters;
  2. stop the serial console (serial-getty@ttyPS0) and lower the kernel console
     log level, so the UART carries only frames (both restored at the end, no reboot);
  3. start the sender under the temperature guard and read COM6 on the PC with the
     production parser (host/frame_protocol.py) until the last frame;
  4. restore the console, fetch the board's sent-frame log and compare frame by frame:
     every frame received once, in order, byte-identical to the frame sent, class equal
     to the argmax of the reference logits of that dataset window, CONF equal to the
     PC-side calibrated confidence (host/confidence.py).

    python -m tools.uart_result_test --n 1024 --deployment /home/xilinx/deployments/<dep>
Evidence: artifacts/results/uart_results_<timestamp>/
"""
import argparse
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial

from host.confidence import display_confidence
from host.frame_protocol import ParseError, gap_after, parse_frame

REPO = Path(__file__).resolve().parents[1]
SET_DIR = REPO / "artifacts" / "ram_demo_sets" / "class_balanced_256_v1"
CAL = REPO / "artifacts" / "calibration" / "conf_temperature_v1" / "temperature.json"
REF_LOGITS = (REPO / "artifacts" / "results" / "fpga_realtime_bench_20261022"
              / "board_full_pipelined" / "logits_i32.bin")
HOST, KEY = "root@192.168.2.99", str(Path.home() / ".ssh" / "id_zturn")
DEFAULT_DEP = "/home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f"


def ssh(cmd, check=True, **kw):
    return subprocess.run(["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, cmd],
                          check=check, capture_output=True, text=True, **kw)


def scp_to(srcs, dst):
    subprocess.run(["scp", "-q", "-i", KEY] + [str(s) for s in srcs] + ["%s:%s" % (HOST, dst)], check=True)


def reader(port, stop, sink):
    with serial.Serial(port, 115200, timeout=0.2) as ser:
        ser.reset_input_buffer()
        sink["opened"] = time.time()
        buf = b""
        while not stop.is_set():
            buf += ser.read(4096)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                sink["lines"].append((time.time(), line.decode("ascii", errors="replace")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--period", type=float, default=1024 / 50000)
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--deployment", default=DEFAULT_DEP)
    args = ap.parse_args()

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "artifacts" / "results" / ("uart_results_" + tag)
    out.mkdir(parents=True)
    run = "/home/xilinx/uart_demo_" + tag
    cal = json.loads(CAL.read_text())
    conf_json = out / "conf.json"
    conf_json.write_text(json.dumps({"logits_scale": cal["s_logits"], "T": cal["T"]}))

    pre = ssh("pgrep -fa '[f]pga_|[t]emp_guard' || true; cat /proc/sys/kernel/printk; "
              "systemctl is-active serial-getty@ttyPS0.service || true").stdout
    (out / "preflight.txt").write_text(pre)
    if "fpga_" in pre.split("\n")[0]:
        raise SystemExit("another board workload is running:\n" + pre)
    printk = pre.strip().split("\n")[-2].split()[0]
    ssh("mkdir %s" % run)
    scp_to([REPO / "tools" / "board" / "fpga_uart_demo.py", REPO / "tools" / "board" / "temp_guard.sh",
            SET_DIR / "set.i32", SET_DIR / "indices.txt", conf_json], run + "/")
    ssh("sed -i 's/\\r$//' %s/temp_guard.sh; chmod +x %s/temp_guard.sh" % (run, run))

    stop, sink = threading.Event(), {"lines": []}
    t = threading.Thread(target=reader, args=(args.port, stop, sink), daemon=True)
    console_restored = False
    try:
        ssh("systemctl stop serial-getty@ttyPS0.service; dmesg -n 1")
        t.start()
        time.sleep(1.0)
        start_cmd = ("cd {dep}/driver; PYTHONPATH={dep}/driver nohup python3.6 -u {run}/fpga_uart_demo.py "
                     "{dep}/overlay/design_1_wrapper.bit {run}/set.i32 {run}/indices.txt {n} {p} "
                     "/dev/ttyPS0 {run}/conf.json {run}/out > {run}/run.out 2>&1 < /dev/null & "
                     "pid=$!; nohup {run}/temp_guard.sh $pid {run}/guard.log 75 1 > /dev/null 2>&1 "
                     "< /dev/null & echo $pid").format(dep=args.deployment, run=run, n=args.n, p=args.period)
        pid = ssh(start_cmd).stdout.strip()
        deadline = time.time() + args.n * args.period + 120
        while time.time() < deadline:
            time.sleep(5)
            if ssh("kill -0 %s 2>/dev/null && echo alive || echo done" % pid, check=False).stdout.strip() == "done":
                break
        time.sleep(2)
    finally:
        stop.set()
        t.join(timeout=5)
        r = ssh("dmesg -n %s; systemctl start serial-getty@ttyPS0.service; "
                "systemctl is-active serial-getty@ttyPS0.service" % printk, check=False)
        console_restored = r.stdout.strip().endswith("active")
    subprocess.run(["scp", "-q", "-r", "-i", KEY, "%s:%s/out" % (HOST, run), "%s:%s/run.out" % (HOST, run),
                    "%s:%s/guard.log" % (HOST, run), str(out)], check=False)
    (out / "received_lines.txt").write_text(
        "".join("%.6f\t%s\n" % (ts, line) for ts, line in sink["lines"]), encoding="utf-8")

    sent = [json.loads(l) for l in (out / "out" / "sent.jsonl").read_text().splitlines() if l]
    frames, malformed, gaps, prev = [], [], 0, None
    for ts, line in sink["lines"]:
        try:
            f = parse_frame(line)
        except ParseError:
            malformed.append(line)
            continue
        gaps += gap_after(prev, f.frame_id)
        prev = f.frame_id
        frames.append(f)
    ref = np.fromfile(REF_LOGITS, dtype="<i4").reshape(-1, 4)
    by_id = {}
    for f in frames:
        by_id.setdefault(f.frame_id, []).append(f)
    text_equal = cls_ref = conf_equal = 0
    for s in sent:
        got = by_id.get(s["k"] % 65536, [])
        if len(got) == 1 and got[0].format().strip() == s["line"]:
            text_equal += 1
        lg = ref[s["dataset_index"]]
        cls_ref += int(s["cls"] == int(np.argmax(lg)) and s["logits"] == lg.tolist())
        pc_conf, _ = display_confidence(s["logits"], s["cls"])
        conf_equal += int(got and got[0].confidence is not None and abs(got[0].confidence - round(pc_conf, 3)) < 5e-4)
    result = {
        "frames_sent": len(sent), "lines_received": len(sink["lines"]), "frames_parsed": len(frames),
        "malformed_lines": len(malformed), "malformed_examples": malformed[:5],
        "id_gaps": gaps, "duplicates": sum(len(v) - 1 for v in by_id.values() if len(v) > 1),
        "received_equal_to_sent_text": text_equal,
        "class_and_logits_equal_reference": cls_ref,
        "conf_equal_pc_calibrated": conf_equal,
        "console_restored": console_restored,
        "board_summary": json.loads((out / "out" / "summary.json").read_text()),
        "deployment": args.deployment, "port": args.port, "period_s": args.period,
    }
    (out / "uart_result_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
