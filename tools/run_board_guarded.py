"""Run one board workload under the independent temperature guard (PC side).

Sequence (any failure stops before inference; nothing is retried):
  1. read-only preflight (host.live_preflight) -> <out>/preflight.json; FAIL => stop
  2. create a NEW remote dir /home/xilinx/timing_runs_20261023/<tag> (fails if it exists)
  3. copy the listed files there (+ temp_guard.sh); record their sha256
  4. on the board: workload & guard(<workload pid>, 75 C, 1 s); if the guard exits
     while the workload is alive, the workload is terminated; exit codes recorded
     (nohup, so a dropped SSH session does not kill or orphan anything)
  5. poll until done, copy the remote dir to <out>/remote/, write <out>/run.json

The workload command is given after `--`, with {DIR} = remote run dir,
{DEP} = deployment dir, {BIT} = deployed bitstream, {IN} = test-set input.
It runs in {DEP}/driver with PYTHONPATH set to it (deployed driver).

Example:
  python -m tools.run_board_guarded --tag diag_dma --out artifacts/results/fpga_timing_20261023/diag_dma \
     --files tools/board/fpga_tail_diag.py -- \
     python3.6 -u {DIR}/fpga_tail_diag.py dma {BIT} {IN} 256 20000 {DIR}/out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

from host.live_fpga import BIT, BOARD_HOST, BOARD_USER, DEP_DIR, IN_GZ, SSH_KEY, _ssh_base
from host.live_preflight import all_ok, preflight

ROOT = Path(__file__).resolve().parents[1]
REMOTE_BASE = "/home/xilinx/timing_runs_20261023"
GUARD = ROOT / "tools" / "board" / "temp_guard.sh"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ssh(cmd: str, timeout=30):
    return subprocess.run(_ssh_base(BOARD_HOST, BOARD_USER, SSH_KEY) + [cmd],
                          capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--n-preflight", type=int, default=95404)
    ap.add_argument("--max-minutes", type=float, default=60)
    ap.add_argument("--finalize-only", action="store_true",
                    help="do not launch: wait for / collect an already launched <tag> "
                         "(used once when a launch call timed out locally)")
    ap.add_argument("--stream", action="store_true",
                    help="live-path mode: ONE foreground SSH session, workload stdout streamed to "
                         "the PC (no board-side log file, no polling); PC receive time recorded")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    rdir = "%s/%s" % (REMOTE_BASE, a.tag)
    if a.finalize_only:
        run = json.loads((a.out / "run.json").read_text()) if (a.out / "run.json").exists() else             {"tag": a.tag, "remote_dir": rdir}
        run["note"] = "collected with --finalize-only (launch call timed out locally; job kept running under its guard)"
        return _collect(a, rdir, run, time.perf_counter())
    a.out.mkdir(parents=True, exist_ok=False)
    run = {"tag": a.tag, "remote_dir": rdir, "deployment": DEP_DIR, "bit": BIT, "input": IN_GZ,
           "pc_start": datetime.now().isoformat(timespec="seconds")}

    t0 = time.perf_counter()
    checks = preflight(n=a.n_preflight)
    run["preflight_s"] = round(time.perf_counter() - t0, 2)
    (a.out / "preflight.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if not all_ok(checks):
        run["result"] = "NOT STARTED: preflight failed"
        (a.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(run["result"], [c for c in checks if not c["ok"]])
        return 2

    r = _ssh("mkdir -p %s && mkdir %s" % (REMOTE_BASE, shlex.quote(rdir)))
    if r.returncode != 0:
        run["result"] = "NOT STARTED: remote dir exists or mkdir failed: " + r.stderr.strip()
        (a.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(run["result"])
        return 2
    files = [Path(f) for f in a.files] + [GUARD]
    run["uploaded_sha256"] = {f.name: _sha(f) for f in files}
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-i", SSH_KEY] + [str(f) for f in files]
                   + ["%s@%s:%s/" % (BOARD_USER, BOARD_HOST, rdir)], check=True)

    sub = {"{DIR}": rdir, "{DEP}": DEP_DIR, "{BIT}": BIT, "{IN}": IN_GZ}
    work = " ".join(shlex.quote(c) if not any(k in c for k in sub) else c for c in cmd)
    for k, v in sub.items():
        work = work.replace(k, v)
    run["workload"] = work
    drv = DEP_DIR + "/driver"
    script = (
        "#!/bin/bash\n"
        "cd %(drv)s || { echo 'ABORT no driver dir'; echo DONE; exit 1; }\n"
        "export PYTHONPATH=%(drv)s\n"
        "date -u +START_UTC=%%FT%%TZ\n"
        "%(work)s %(redir)s & P=$!\n"
        "bash %(dir)s/temp_guard.sh $P %(dir)s/guard.log 75 1 & G=$!\n"
        "while kill -0 $P 2>/dev/null; do\n"
        "  if ! kill -0 $G 2>/dev/null; then kill -TERM $P 2>/dev/null; sleep 2; kill -KILL $P 2>/dev/null;\n"
        "    echo 'ABORT independent guard exited while workload alive'; fi\n"
        "  sleep 1\n"
        "done\n"
        "wait $P; R=$?; wait $G; S=$?\n"
        "echo WORKLOAD_RC=$R; echo GUARD_RC=$S; date -u +END_UTC=%%FT%%TZ; echo DONE\n"
        % {"drv": drv, "work": work, "dir": rdir,
           "redir": "2>&1" if a.stream else "> %s/workload.out 2>&1" % rdir})
    local_script = a.out / "run_remote.sh"
    local_script.write_bytes(script.encode())
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-i", SSH_KEY, str(local_script),
                    "%s@%s:%s/run_remote.sh" % (BOARD_USER, BOARD_HOST, rdir)], check=True)
    t1 = time.perf_counter()
    (a.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    if a.stream:
        return _stream(a, rdir, run, t1)
    # Detached (setsid, own session, no fds on the SSH channel): this call returns
    # at once and a dropped connection can neither kill nor orphan the workload.
    # Note `;` not `&&`: "cd x && cmd &" would background a subshell holding the
    # channel open. (util-linux 2.31 on the board has no `setsid -f`.)
    r = _ssh("cd %s || exit 1; setsid bash run_remote.sh > run.out 2>&1 < /dev/null & echo LAUNCHED" % rdir)
    if "LAUNCHED" not in r.stdout:
        run["result"] = "NOT STARTED: launch failed: " + (r.stderr or r.stdout).strip()
        (a.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(run["result"])
        return 2
    return _collect(a, rdir, run, t1)


def _fetch_remote(a, rdir):
    """Copy the remote run dir to <out>/remote (an earlier copy is kept, renamed;
    scp -r into an existing dir would otherwise nest it)."""
    dst = a.out / "remote"
    if dst.exists():
        dst.rename(a.out / ("remote_previous_%s" % datetime.now().strftime("%H%M%S")))
    subprocess.run(["scp", "-q", "-r", "-o", "BatchMode=yes", "-i", SSH_KEY,
                    "%s@%s:%s" % (BOARD_USER, BOARD_HOST, rdir), str(dst)])


def _stream(a, rdir, run, t1) -> int:
    """Foreground, single SSH session; lines go to the PC as in the live demo."""
    argv = _ssh_base(BOARD_HOST, BOARD_USER, SSH_KEY, timeout=15) + ["bash %s/run_remote.sh" % rdir]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, stdin=subprocess.DEVNULL)
    lines = []
    with open(a.out / "pc_rx.tsv", "w", encoding="utf-8") as rx:
        for line in proc.stdout:
            t = (time.perf_counter() - t1) * 1000.0
            line = line.rstrip("\n")
            rx.write("%.3f\t%s\n" % (t, line))
            lines.append(line)
    proc.wait()
    (a.out / "workload_stream.out").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run["board_wall_s"] = round(time.perf_counter() - t1, 1)
    run["run_out"] = [l for l in lines if not l.startswith("R ")][-12:]
    rc = dict(l.split("=", 1) for l in lines if l.startswith(("WORKLOAD_RC=", "GUARD_RC=")))
    run["workload_rc"], run["guard_rc"] = rc.get("WORKLOAD_RC"), rc.get("GUARD_RC")
    run["result"] = "OK" if rc.get("WORKLOAD_RC") == "0" and rc.get("GUARD_RC") == "0" \
        and not any(l.startswith("ABORT") for l in lines) else "FAILED/ABORTED (see run_out)"
    _fetch_remote(a, rdir)
    post = _ssh("ps -eo pid=,args= | grep -E 'python3[.]6 .*([f]pga_|[t]ail_diag)|[t]emp_guard[.]sh' || echo CLEAN")
    run["post_run_process_check"] = post.stdout.strip()
    run["pc_end"] = datetime.now().isoformat(timespec="seconds")
    (a.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("result", "board_wall_s", "post_run_process_check")}, indent=2))
    return 0 if run["result"] == "OK" else 1


def _collect(a, rdir, run, t1) -> int:
    deadline = time.time() + a.max_minutes * 60
    out = ""
    while time.time() < deadline:
        time.sleep(10)
        r = _ssh("cat %s/run.out" % rdir)
        out = r.stdout
        if "DONE" in out:
            break
    run["board_wall_s"] = round(time.perf_counter() - t1, 1)
    run["run_out"] = out.strip().splitlines()
    if "DONE" not in out:
        run["result"] = "INCOMPLETE: not finished within %.0f min (left running under its guard)" % a.max_minutes
    else:
        rc = {k: v for k, v in (l.split("=", 1) for l in run["run_out"] if l.endswith(tuple("0123456789")) and "_RC=" in l)}
        run["workload_rc"], run["guard_rc"] = rc.get("WORKLOAD_RC"), rc.get("GUARD_RC")
        run["result"] = "OK" if rc.get("WORKLOAD_RC") == "0" and rc.get("GUARD_RC") == "0" \
            and "ABORT" not in out else "FAILED/ABORTED (see run_out, remote/)"
    _fetch_remote(a, rdir)
    post = _ssh("ps -eo pid=,args= | grep -E 'python3[.]6 .*([f]pga_|[t]ail_diag)|[t]emp_guard[.]sh' || echo CLEAN")
    run["post_run_process_check"] = post.stdout.strip()
    run["pc_end"] = datetime.now().isoformat(timespec="seconds")
    (a.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("result", "board_wall_s", "post_run_process_check")}, indent=2))
    return 0 if run["result"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
