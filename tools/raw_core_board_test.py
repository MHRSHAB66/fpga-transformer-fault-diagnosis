"""Physical validation of the raw-input core (preprocessing inside the FPGA).

Steps (each its own subcommand, all under the independent temperature guard):

  deploy    copy the versioned bundle to /home/xilinx/deployments/<name> and check
            its MANIFEST.sha256 on the board
  golden    golden_raw1024_argmax on the real board: logits + class register,
            41 vectors incl. ties and edge windows
  fulltest  all 95,404 raw test windows through the board (fpga_live_runner.py);
            compares every board logit with the C-model / Python reference
  rt        hard real-time acceptance run with the raw input (same configuration
            as R1 of 2026-10-24: 4,096 RAM windows, 30,000 + 200 warm-up,
            SCHED_FIFO 50 on CPU 1)

    python -m tools.raw_core_board_test deploy --bundle artifacts/deploy/seq256_raw_argmax_z7020_20261109
    python -m tools.raw_core_board_test golden|fulltest|rt --name <deployment dir name>
Evidence: artifacts/results/raw_core_board_<step>_<timestamp>/
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HOST, KEY = "root@192.168.2.99", str(Path.home() / ".ssh" / "id_zturn")
DEPLOYMENTS = "/home/xilinx/deployments"
RAW_GZ = REPO / "artifacts" / "results" / "fpga_rawinput_testset_20261109" / "fpga_testset_raw.i32.gz"
BOARD_RAW_GZ = "/home/xilinx/raw_testset_20261109/fpga_testset_raw.i32.gz"
CMODEL = REPO / "artifacts" / "results" / "onchip_preprocess_20261109" / "csim_raw_argmax" / "csim_logits.txt"
LABELS = REPO / "artifacts" / "results" / "fpga_rawinput_testset_20261109" / "labels.txt"
BOARD_TOOLS = ["verify_argmax_golden.py", "fpga_live_runner.py", "fpga_rt_acceptance.py", "temp_guard.sh"]


def ssh(cmd, check=True, timeout=None):
    return subprocess.run(["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, cmd],
                          check=check, capture_output=True, text=True, timeout=timeout)


def scp(srcs, dst, recursive=False):
    cmd = ["scp", "-q", "-i", KEY] + (["-r"] if recursive else [])
    subprocess.run(cmd + [str(s) for s in srcs] + ["%s:%s" % (HOST, dst)], check=True)


def fetch(remote, local):
    subprocess.run(["scp", "-q", "-r", "-i", KEY, "%s:%s" % (HOST, remote), str(local)], check=True)


def new_run(step):
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPO / "artifacts" / "results" / ("raw_core_board_%s_%s" % (step, tag))
    out.mkdir(parents=True)
    run = "/home/xilinx/raw_core_%s_%s" % (step, tag)
    busy = ssh("pgrep -fa '[f]pga_live_runner|[f]pga_rt_acceptance|[v]erify_argmax_golden|[f]pga_uart_demo|[r]un_cpu_baseline' || true").stdout.strip()
    if busy:
        raise SystemExit("board busy:\n" + busy)
    ssh("mkdir %s" % run)
    scp([REPO / "tools" / "board" / t for t in BOARD_TOOLS], run + "/")
    ssh("sed -i 's/\\r$//' %s/temp_guard.sh; chmod +x %s/temp_guard.sh" % (run, run))
    return out, run


def guarded(dep, run, cmd, logname, background=False):
    """Run <cmd> inside <dep>/driver with the guard watching it."""
    body = ("cd {dep}/driver; PYTHONPATH={dep}/driver:{run} {cmd} > {run}/{log}.out 2>&1 < /dev/null & "
            "pid=$!; nohup {run}/temp_guard.sh $pid {run}/{log}_guard.log 75 1 > /dev/null 2>&1 < /dev/null & ")
    body = body.format(dep=dep, run=run, cmd=cmd, log=logname)
    if background:
        return ssh(body + "echo $pid").stdout.strip()
    return ssh(body + "wait $pid; echo rc=$?", timeout=3600).stdout.strip()


def wait_pid(pid, poll=30):
    while ssh("kill -0 %s 2>/dev/null && echo alive || echo done" % pid, check=False).stdout.strip() == "alive":
        time.sleep(poll)


def cmd_deploy(args):
    bundle = Path(args.bundle)
    dest = "%s/%s" % (DEPLOYMENTS, bundle.name)
    if ssh("test -e %s && echo exists || echo free" % dest).stdout.strip() == "exists":
        raise SystemExit("%s exists on the board" % dest)
    scp([bundle], DEPLOYMENTS + "/", recursive=True)
    r = ssh("cd %s && sha256sum -c MANIFEST.sha256 --quiet && echo MANIFEST_OK" % dest, check=False)
    print(r.stdout + r.stderr)


def cmd_golden(args):
    dep = "%s/%s" % (DEPLOYMENTS, args.name)
    out, run = new_run("golden")
    rc = guarded(dep, run, "python3.6 -u {run}/verify_argmax_golden.py {dep}/overlay/design_1_wrapper.bit "
                 "{dep}/golden".format(run=run, dep=dep), "golden")
    fetch(run + "/golden.out", out)
    fetch(run + "/golden_guard.log", out)
    text = (out / "golden.out").read_text()
    print(rc, text.strip().splitlines()[-1] if text.strip() else "")


def cmd_fulltest(args):
    dep = "%s/%s" % (DEPLOYMENTS, args.name)
    if ssh("test -f %s && echo ok || echo missing" % BOARD_RAW_GZ).stdout.strip() == "missing":
        ssh("mkdir -p %s" % str(Path(BOARD_RAW_GZ).parent).replace("\\", "/"))
        scp([RAW_GZ], BOARD_RAW_GZ)
    out, run = new_run("fulltest")
    n = args.n
    pid = guarded(dep, run, "python3.6 -u {run}/fpga_live_runner.py {dep}/overlay/design_1_wrapper.bit {gz} "
                  "1024 8 {n} {run}/pred.txt".format(run=run, dep=dep, gz=BOARD_RAW_GZ, n=n),
                  "fulltest", background=True)
    t0 = time.time()
    wait_pid(pid)
    for f in ("fulltest.out", "fulltest_guard.log", "pred.txt"):
        fetch("%s/%s" % (run, f), out)
    rows = {}
    for line in (out / "fulltest.out").read_text().splitlines():
        if line.startswith("R "):
            kv = dict(p.split("=", 1) for p in line.split()[1:])
            rows[int(kv["idx"])] = (int(kv["cls"]), [int(v) for v in kv["logits"].split(",")],
                                    float(kv["lat_ms"]))
    ref = np.loadtxt(CMODEL, dtype=np.int64)
    labels = np.loadtxt(LABELS, dtype=np.int64)
    idx = np.array(sorted(rows))
    board_cls = np.array([rows[i][0] for i in idx])
    board_lg = np.array([rows[i][1] for i in idx])
    lat = np.array([rows[i][2] for i in idx])
    summary = [l for l in (out / "fulltest.out").read_text().splitlines() if l.startswith("SUMMARY")]
    result = {"windows_received": int(len(idx)), "requested": n,
              "logits_equal_cmodel": int((board_lg == ref[idx, 1:5]).all(axis=1).sum()),
              "class_equal_cmodel_register": int((board_cls == ref[idx, 5]).sum()),
              "accuracy": float((board_cls == labels[idx]).mean()),
              "correct": int((board_cls == labels[idx]).sum()),
              "latency_ms": {"mean": round(float(lat.mean()), 3), "p50": round(float(np.median(lat)), 3),
                             "p95": round(float(np.percentile(lat, 95)), 3), "max": round(float(lat.max()), 3)},
              "board_summary": summary[-1] if summary else None,
              "wall_s_pc": round(time.time() - t0, 1), "deployment": dep, "input": BOARD_RAW_GZ}
    (out / "fulltest_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def cmd_rt(args):
    dep = "%s/%s" % (DEPLOYMENTS, args.name)
    out, run = new_run("rt")
    rc = guarded(dep, run, "python3.6 -u {run}/fpga_rt_acceptance.py {dep}/overlay/design_1_wrapper.bit {gz} "
                 "4096 30200 200 1 1 50 {run}/out".format(run=run, dep=dep, gz=BOARD_RAW_GZ), "rt")
    fetch(run + "/out", out)
    fetch(run + "/rt.out", out)
    fetch(run + "/rt_guard.log", out)
    print(rc)
    print(json.dumps(json.loads((out / "out" / "summary.json").read_text()), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deploy")
    d.add_argument("--bundle", required=True)
    for name in ("golden", "fulltest", "rt"):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        if name == "fulltest":
            p.add_argument("--n", type=int, default=95404)
    args = ap.parse_args()
    {"deploy": cmd_deploy, "golden": cmd_golden, "fulltest": cmd_fulltest, "rt": cmd_rt}[args.cmd](args)


if __name__ == "__main__":
    main()
