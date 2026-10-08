"""Live FPGA inference client (runs on the Windows PC).

Drives REAL on-board inference over SSH/Ethernet and streams the results back
for display. The PC performs NO classification: it launches the board-side
``fpga_live_runner.py``, which feeds each test-set window through the AXI-DMA
transformer core and prints one result line per window; this module parses that
stream into events. The large test-set input lives on the board already (staged
over Ethernet); only compact result lines travel back.

It also (re)starts the INDEPENDENT board temperature monitor so the safety
watchdog runs in its own process, never inside the DMA path.

Event dicts yielded by ``stream()``:
    {"type": "status",  "msg": str}                       connection / lifecycle
    {"type": "temp_start", "temp_c": float|None}
    {"type": "meta",   "n": int, "seq": int, "ch": int}
    {"type": "result", "idx": int, "cls": int, "label": str,
                       "frame": str, "lat_ms": float, "temp_c": float|None}
    {"type": "summary","executed": int, "aborted": str|None, "wall_s": float,
                       "throughput": float, "temp_start":..,"temp_max":..,"temp_end":..}
    {"type": "abort",  "msg": str}
    {"type": "error",  "msg": str}

Headless CLI (real hardware; produces versioned evidence):
    python -m host.live_fpga --n 150
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, Iterator, List, Optional

from host.frame_protocol import LABELS, format_frame

# -- board defaults (match the proven full-test deployment) -------------------
BOARD_HOST = "192.168.2.99"
BOARD_USER = "root"
SSH_KEY = os.path.join(os.path.expanduser("~"), ".ssh", "id_zturn")
# The DMA-fixed deployment (C_SG_LENGTH_WIDTH=16 -> 64KB max) is the one that
# actually transfers a full 32768-byte window in a single MM2S; it backed the
# proven 95404-window full test. The earlier _114947 build is capped at 16KB.
DEP_DIR = "/home/xilinx/deployments/seq256_z7020_pynq_dma16_py36_20261008_131446"
BOARD_WORK = "/home/xilinx/fpga_fulltest_20261008"
BIT = DEP_DIR + "/overlay/design_1_wrapper.bit"
IN_GZ = BOARD_WORK + "/fpga_testset_in.i32.gz"
RUNNER_REMOTE = "/home/xilinx/fpga_live_runner.py"
MONITOR_REMOTE = "/tmp/temp_monitor_clean.sh"
SEQ, CH = 1024, 8

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class LiveError(Exception):
    pass


def _ssh_base(host, user, key, port=22, timeout=10) -> List[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=%d" % timeout,
            "-p", str(port), "-i", key, "%s@%s" % (user, host)]


class LiveFpgaClient:
    """Launches and consumes the board-side live inference stream."""

    def __init__(self, host=BOARD_HOST, user=BOARD_USER, key=SSH_KEY, port=22,
                 dep_dir=DEP_DIR, bit=BIT, in_gz=IN_GZ, n=150,
                 runner_remote=RUNNER_REMOTE, monitor_remote=MONITOR_REMOTE,
                 driver_subdir="driver", line_timeout=45.0, indices=None,
                 indices_remote="/home/xilinx/fpga_live_indices.txt"):
        self.host, self.user, self.key, self.port = host, user, key, port
        self.dep_dir, self.bit, self.in_gz, self.n = dep_dir, bit, in_gz, n
        self.driver_subdir = driver_subdir
        self.runner_remote, self.monitor_remote = runner_remote, monitor_remote
        self.line_timeout = line_timeout
        # Optional explicit dataset indices (real frame ids) for a multiclass
        # demo; None -> first n sequential (back-compatible).
        self.indices = list(indices) if indices is not None else None
        if self.indices is not None:
            self.n = len(self.indices)
        self.indices_remote = indices_remote
        self.proc: Optional[subprocess.Popen] = None
        self._stop = False

    def push_indices(self) -> None:
        """Write the index list to the board (one per line) before a sampled run."""
        if self.indices is None:
            return
        import tempfile
        cf = (0x08000000 if os.name == "nt" else 0)
        fd, tmp = tempfile.mkstemp(suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="ascii") as f:
                f.write("\n".join(str(int(i)) for i in self.indices) + "\n")
            subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                            "-P", str(self.port), "-i", self.key, tmp,
                            "%s@%s:%s" % (self.user, self.host, self.indices_remote)],
                           check=True, capture_output=True, text=True, creationflags=cf)
        finally:
            os.remove(tmp)

    # -- connectivity -------------------------------------------------------
    def check_connection(self) -> bool:
        try:
            r = subprocess.run(_ssh_base(self.host, self.user, self.key, self.port) +
                               ["echo OK"], capture_output=True, text=True, timeout=12,
                               creationflags=(0x08000000 if os.name == "nt" else 0))
            return r.returncode == 0 and "OK" in r.stdout
        except Exception:
            return False

    # -- stream -------------------------------------------------------------
    def _remote_cmd(self) -> str:
        """Single clean foreground command: run the live DMA runner and stream
        its per-window lines back over this SSH channel. The runner samples die
        temperature in its OWN thread (TempMonitor), so no external monitor
        process is needed and nothing holds the SSH pipe open after it exits."""
        drv = "%s/%s" % (self.dep_dir, self.driver_subdir)
        base = ("cd %s && PYTHONPATH=%s python3.6 -u %s %s %s %d %d %d"
                % (drv, drv, self.runner_remote, self.bit, self.in_gz, SEQ, CH, self.n))
        if self.indices is not None:
            base += " - %s" % self.indices_remote   # out_pred="-" (none) + indices file
        return base

    def stream(self) -> Iterator[Dict]:
        self._stop = False
        if not self.check_connection():
            yield {"type": "error", "msg": "SSH connect failed to %s@%s" % (self.user, self.host)}
            return
        if self.indices is not None:
            try:
                self.push_indices()
                yield {"type": "status", "msg": "pushed %d sampled indices to board" % self.n}
            except Exception as e:
                yield {"type": "error", "msg": "failed to push indices: %s" % e}
                return
        yield {"type": "status",
               "msg": "connected to %s; starting temperature monitor + live runner (N=%d)"
                      % (self.host, self.n)}

        argv = _ssh_base(self.host, self.user, self.key, self.port, timeout=15) + [self._remote_cmd()]
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1,
                                     creationflags=(0x08000000 if os.name == "nt" else 0))

        q: "list" = []
        lock = threading.Lock()
        done = {"v": False}

        def pump():
            try:
                for line in self.proc.stdout:
                    with lock:
                        q.append(line.rstrip("\n"))
            finally:
                done["v"] = True

        threading.Thread(target=pump, daemon=True).start()

        last_rx = time.monotonic()
        while True:
            if self._stop:
                yield {"type": "status", "msg": "stop requested"}
                break
            with lock:
                lines = q[:]
                del q[:]
            if lines:
                last_rx = time.monotonic()
            for ln in lines:
                ev = parse_line(ln)
                if ev is not None:
                    yield ev
            if done["v"] and not lines:
                break
            if time.monotonic() - last_rx > self.line_timeout:
                yield {"type": "error",
                       "msg": "no output for %.0fs (timeout)" % self.line_timeout}
                break
            time.sleep(0.05)

        rc = self.proc.poll()
        yield {"type": "status", "msg": "runner exited rc=%s" % rc}

    def stop(self) -> None:
        self._stop = True
        try:
            # ask the board to end the independent monitor cleanly and kill runner
            subprocess.run(_ssh_base(self.host, self.user, self.key, self.port) +
                           ['pkill -f "[f]pga_live_runner.py"; true'],
                           capture_output=True, text=True, timeout=12,
                           creationflags=(0x08000000 if os.name == "nt" else 0))
        except Exception:
            pass
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Line parsing (board -> event dict)
# ---------------------------------------------------------------------------
def parse_line(line: str) -> Optional[Dict]:
    s = line.strip()
    if not s:
        return None
    if s.startswith("R idx="):
        kv = _kv(s[2:])
        idx, cls = int(kv["idx"]), int(kv["cls"])
        temp = None if kv.get("temp_c", "NA") == "NA" else float(kv["temp_c"])
        label = LABELS.get(cls, "?")
        return {"type": "result", "idx": idx, "cls": cls, "label": label,
                "frame": format_frame(idx, cls, None).strip(),
                "lat_ms": float(kv["lat_ms"]), "temp_c": temp}
    if s.startswith("TEMP_START="):
        v = s.split("=", 1)[1]
        return {"type": "temp_start", "temp_c": None if v == "None" else float(v)}
    if s.startswith("META "):
        kv = _kv(s[5:])
        return {"type": "meta", "n": int(kv["n"]), "seq": int(kv["seq"]), "ch": int(kv["ch"])}
    if s.startswith("SUMMARY "):
        kv = _kv(s[8:])
        return {"type": "summary", "executed": int(kv["executed"]),
                "aborted": None if kv.get("aborted") == "None" else kv.get("aborted"),
                "wall_s": float(kv["wall_s"]), "throughput": float(kv["throughput"]),
                "temp_start": kv.get("temp_start"), "temp_max": kv.get("temp_max"),
                "temp_end": kv.get("temp_end")}
    if s.startswith("ABORT"):
        return {"type": "abort", "msg": s}
    return {"type": "status", "msg": s}


def _kv(s: str) -> Dict[str, str]:
    out = {}
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Headless CLI -> real-hardware evidence
# ---------------------------------------------------------------------------
def _load_int_file(path, limit):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(int(line))
            if len(out) >= limit:
                break
    return out


def push_board_files(host, user, key, port):
    """Copy the runner + monitor to the board (idempotent)."""
    runner = os.path.join(_REPO, "tools", "board", "fpga_live_runner.py")
    monitor = os.path.join(_REPO, "tools", "board", "temp_monitor_clean.sh")
    scp = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
           "-P", str(port), "-i", key]
    cf = (0x08000000 if os.name == "nt" else 0)
    subprocess.run(scp + [runner, "%s@%s:%s" % (user, host, RUNNER_REMOTE)],
                   check=True, capture_output=True, text=True, creationflags=cf)
    subprocess.run(scp + [monitor, "%s@%s:%s" % (user, host, MONITOR_REMOTE)],
                   check=True, capture_output=True, text=True, creationflags=cf)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Live FPGA inference over SSH/Ethernet (real board).")
    ap.add_argument("--n", type=int, default=150, help="windows to run live (>=100 for acceptance)")
    ap.add_argument("--sample-mode", choices=["sequential", "strided", "class_balanced"],
                    default="sequential",
                    help="window selection: sequential (first n), strided (spread), "
                         "class_balanced (multiclass demo, uses labels.txt for index order only)")
    ap.add_argument("--host", default=BOARD_HOST)
    ap.add_argument("--user", default=BOARD_USER)
    ap.add_argument("--key", default=SSH_KEY)
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--no-push", action="store_true", help="skip scp of runner/monitor")
    ap.add_argument("--out-dir", help="evidence dir (default artifacts/results/fpga_live_test_<ts>)")
    args = ap.parse_args(argv)

    fulltest = os.path.join(_REPO, "artifacts", "results", "fpga_fulltest_20261008_141335")
    all_labels = _load_int_file(os.path.join(fulltest, "labels.txt"), 10 ** 9)
    all_recorded = _load_int_file(os.path.join(fulltest, "fpga_pred.txt"), 10 ** 9)

    indices = None
    if args.sample_mode != "sequential":
        from host.live_sampling import build_plan
        indices = build_plan(args.sample_mode, total=len(all_labels), n=args.n,
                             labels=all_labels)
        print("sample-mode=%s -> %d indices (e.g. %s...)" % (args.sample_mode, len(indices), indices[:8]))

    if not args.no_push:
        print("pushing runner + monitor to board...")
        push_board_files(args.host, args.user, args.key, args.port)

    client = LiveFpgaClient(host=args.host, user=args.user, key=args.key,
                            port=args.port, n=args.n, indices=indices)
    results: List[Dict] = []
    summary: Optional[Dict] = None
    temp_start = None
    errors = []
    t0 = None
    for ev in client.stream():
        t = ev["type"]
        if t == "status":
            print("[status]", ev["msg"])
        elif t == "temp_start":
            temp_start = ev["temp_c"]
            print("[temp_start]", temp_start)
        elif t == "meta":
            print("[meta]", ev)
            t0 = time.monotonic()
        elif t == "result":
            ev["elapsed_s"] = round(time.monotonic() - t0, 3) if t0 else None
            results.append(ev)
            if len(results) <= 5 or len(results) % 25 == 0:
                print("  %s  lat=%.1fms temp=%s" % (ev["frame"], ev["lat_ms"], ev["temp_c"]))
        elif t == "summary":
            summary = ev
            print("[summary]", ev)
        elif t == "abort":
            errors.append(ev["msg"])
            print("[ABORT]", ev["msg"])
        elif t == "error":
            errors.append(ev["msg"])
            print("[ERROR]", ev["msg"])

    out_dir = args.out_dir or os.path.join(
        _REPO, "artifacts", "results", "fpga_live_test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    evidence = _build_evidence(args, results, summary, temp_start, errors, indices,
                               all_labels, all_recorded, out_dir)
    aborted = (summary or {}).get("aborted")
    print("\n==== LIVE FPGA EVIDENCE ====")
    print("requested/executed  : %d / %d" % (args.n, evidence["executed_windows"]))
    print("aborted             :", aborted)
    print("true-class counts   :", evidence["true_class_counts"])
    print("pred-class counts   :", evidence["predicted_class_counts"])
    print("accuracy vs truth   :", evidence["accuracy_vs_ground_truth"])
    print("macro P/R/F1        : %.4f / %.4f / %.4f" % (
        evidence["macro_precision"], evidence["macro_recall"], evidence["macro_f1"]))
    print("match vs recorded   : %d/%d (%s)" % (
        evidence["match_vs_recorded_fpga_pred"], evidence["executed_windows"],
        evidence["match_rate_vs_recorded"]))
    print("latency ms          :", evidence["latency_ms"])
    print("throughput inf/s    :", evidence["throughput_inferences_per_second"])
    print("temp C start/min/max/end:", evidence["temperature_c"])
    print("errors              :", evidence["errors"])
    print("evidence dir        :", out_dir)
    return 0 if (evidence["executed_windows"] == args.n and not aborted and not errors) else 1


def _file_sha256(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _build_evidence(args, results, summary, temp_start, errors, indices,
                    all_labels, all_recorded, out_dir):
    """Compute the full per-frame + summary evidence and write files."""
    from host.replay_data import MetricsAccumulator  # reuse the scoring layer

    executed = len(results)
    acc = MetricsAccumulator()
    frames = []
    true_counts = {c: 0 for c in LABELS}
    pred_counts = {c: 0 for c in LABELS}
    correct = match_recorded = 0
    for r in results:
        idx = r["idx"]
        true = all_labels[idx] if idx < len(all_labels) else None
        rec = all_recorded[idx] if idx < len(all_recorded) else None
        pred = r["cls"]
        is_correct = (true == pred)
        is_match = (rec == pred)
        if true is not None:
            acc.add(true, pred)
            true_counts[true] += 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        correct += 1 if is_correct else 0
        match_recorded += 1 if is_match else 0
        frames.append({
            "frame": idx, "dataset_index": idx,
            "true_class": true, "true_label": LABELS.get(true),
            "predicted_class": pred, "label": LABELS.get(pred),
            "result": "CORRECT" if is_correct else "WRONG",
            "match_recorded_fpga_pred": is_match,
            "latency_ms": r["lat_ms"], "temperature_c": r["temp_c"],
            "elapsed_s": r.get("elapsed_s"), "conf": "NA",
        })
    m = acc.finalize() if executed else {}
    lats = [r["lat_ms"] for r in results]
    temps = [r["temp_c"] for r in results if r["temp_c"] is not None]

    def pct(v, q):
        return round(statistics.quantiles(v, n=100)[q - 1], 3) if len(v) >= 2 else (v[0] if v else None)

    bundle = os.path.join(_REPO, "artifacts", "deploy",
                          "seq256_z7020_pynq_dma16_py36_20261008", "overlay")
    evidence = {
        "mode": "LIVE FPGA (real on-board AXI-DMA inference over SSH/Ethernet)",
        "real_hardware": True,
        "classification_on": "FPGA transformer_top core (PC performs no inference)",
        "timestamp": datetime.now().isoformat(),
        "board": {"host": args.host, "deployment": DEP_DIR,
                  "bitstream_on_board": BIT, "input_on_board": IN_GZ,
                  "bit_sha256_local_bundle": _file_sha256(os.path.join(bundle, "design_1_wrapper.bit")),
                  "hwh_sha256_local_bundle": _file_sha256(os.path.join(bundle, "design_1_wrapper.hwh"))},
        "sample_mode": args.sample_mode,
        "sampling_indices": list(indices) if indices is not None else "sequential",
        "requested_windows": args.n,
        "executed_windows": executed,
        "aborted": (summary or {}).get("aborted"),
        "true_class_counts": {LABELS[c]: true_counts.get(c, 0) for c in LABELS},
        "predicted_class_counts": {LABELS[c]: pred_counts.get(c, 0) for c in LABELS},
        "correct_vs_ground_truth": correct,
        "accuracy_vs_ground_truth": (correct / executed) if executed else None,
        "macro_precision": m.get("macro_precision"),
        "macro_recall": m.get("macro_recall"),
        "macro_f1": m.get("macro_f1"),
        "per_class_precision": m.get("per_class_precision"),
        "per_class_recall": m.get("per_class_recall"),
        "per_class_f1": m.get("per_class_f1"),
        "confusion_matrix": m.get("confusion_matrix"),
        "match_vs_recorded_fpga_pred": match_recorded,
        "match_rate_vs_recorded": (match_recorded / executed) if executed else None,
        "latency_ms": {"min": min(lats) if lats else None,
                       "mean": round(sum(lats) / len(lats), 3) if lats else None,
                       "p50": pct(lats, 50), "p95": pct(lats, 95),
                       "max": max(lats) if lats else None},
        "throughput_inferences_per_second": (summary or {}).get("throughput"),
        "temperature_c": {"start": temp_start,
                          "min": min(temps) if temps else None,
                          "max": max(temps) if temps else None,
                          "end": (summary or {}).get("temp_end")},
        "errors": errors,
        "summary_line": summary,
    }
    with open(os.path.join(out_dir, "live_evidence.json"), "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    with open(os.path.join(out_dir, "live_frames.jsonl"), "w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr) + "\n")
    with open(os.path.join(out_dir, "live_frames.csv"), "w", encoding="utf-8") as f:
        f.write("frame,true_class,true_label,pred_class,label,result,match_recorded,latency_ms,temp_c,elapsed_s\n")
        for fr in frames:
            f.write("%s,%s,%s,%s,%s,%s,%s,%.3f,%s,%s\n" % (
                fr["frame"], fr["true_class"], fr["true_label"], fr["predicted_class"],
                fr["label"], fr["result"], fr["match_recorded_fpga_pred"],
                fr["latency_ms"], fr["temperature_c"], fr["elapsed_s"]))
    return evidence


if __name__ == "__main__":
    raise SystemExit(main())
