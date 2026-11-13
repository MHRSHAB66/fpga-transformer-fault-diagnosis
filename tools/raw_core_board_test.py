"""Physical validation of the raw-input core (preprocessing inside the FPGA).

Steps (each its own subcommand, all under the independent temperature guard):

  deploy    copy the versioned bundle to /home/xilinx/deployments/<name> and check
            its MANIFEST.sha256 on the board
  golden    golden_raw1024_argmax on the real board: logits + class register,
            41 vectors incl. ties and edge windows
  fulltest  all 95,404 raw test windows through the board (fpga_live_runner.py);
            compares every board logit with the C-model / Python reference; on a
            vote-variant overlay also checks vote_class / vote_count (--vote-k)
            against host/temporal_vote.py over the board's own class stream
  rt        hard real-time acceptance run with the raw input (same configuration
            as R1 of 2026-10-24: 4,096 RAM windows, 30,000 + 200 warm-up,
            SCHED_FIFO 50 on CPU 1); --n-total for a long endurance run

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
BOARD_NOISY_DIR = "/home/xilinx/raw_testset_perturbed_20261111"
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
    """Return when the board reports the process gone. A failed SSH check (empty
    answer) is retried, not taken as the end of the run."""
    while True:
        try:
            state = ssh("kill -0 %s 2>/dev/null && echo alive || echo done" % pid, check=False,
                        timeout=60).stdout.strip()
        except subprocess.TimeoutExpired:
            state = ""
        if state == "done":
            return
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
    local_gz = Path(args.input) if args.input else RAW_GZ
    board_gz = BOARD_RAW_GZ if args.input is None else "%s/%s" % (BOARD_NOISY_DIR, local_gz.name)
    if ssh("test -f %s && echo ok || echo missing" % board_gz).stdout.strip() == "missing":
        ssh("mkdir -p %s" % board_gz.rsplit("/", 1)[0])
        scp([local_gz], board_gz)
    out, run = new_run("fulltest")
    n = args.n
    pid = guarded(dep, run, "FPGA_RUNNER_VOTE_K={k} python3.6 -u {run}/fpga_live_runner.py {dep}/overlay/design_1_wrapper.bit {gz} "
                  "1024 8 {n} {run}/pred.txt".format(run=run, dep=dep, gz=board_gz, n=n, k=args.vote_k),
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
    ref = np.loadtxt(args.cmodel, dtype=np.int64)
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
              "wall_s_pc": round(time.time() - t0, 1), "deployment": dep, "input": board_gz,
              "cmodel_reference": str(Path(args.cmodel))}
    integ = [int(l.split(" integrity=")[1].split()[0]) for l in (out / "fulltest.out").read_text().splitlines()
             if l.startswith("R ") and " integrity=" in l]
    if integ:   # research B02 overlay: clean input must never raise a flag
        result["integrity_status"] = {"windows": len(integ), "nonzero": int(sum(v != 0 for v in integ)),
                                      "values": sorted(set(integ))}
    votes = vote_rows(out / "fulltest.out")
    if votes:
        result["temporal_vote"] = check_votes(votes, rows, labels, int(args.vote_k))
    (out / "fulltest_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def vote_rows(path):
    """{dataset index: (vote_class, vote_count)} from runner lines with vote=."""
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("R ") and " vote=" in line:
            kv = dict(p.split("=", 1) for p in line.split()[1:])
            c, n = kv["vote"].split(",")
            out[int(kv["idx"])] = (int(c), int(n))
    return out


def transition_delays(vote_cls, y):
    """Windows from each change of the true class until the vote first equals the
    new class, searched only inside that segment (until the next true-class
    change or the end). Events never resolved inside their segment are counted
    separately instead of being folded in as a capped delay (the first version
    capped the search at 244 windows and counted unresolved events as 244)."""
    changes = [j for j in range(1, len(y)) if y[j] != y[j - 1]]
    bounds = changes + [len(y)]
    resolved, unresolved = [], 0
    for j, end in zip(changes, bounds[1:]):
        hit = np.flatnonzero(vote_cls[j:end] == y[j])
        if len(hit):
            resolved.append(int(hit[0]))
        else:
            unresolved += 1
    stats = (lambda d: {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": int(max(d))}
             if d else None)
    return {"class_changes": len(changes), "resolved": len(resolved), "unresolved": unresolved,
            "windows_until_correct_resolved_only": stats(resolved),
            "method": "search within the segment up to the next true-class change; unresolved counted separately; "
                      "recording joins with the same class are not transitions"}


def check_votes(votes, rows, labels, k):
    """Board votes vs host/temporal_vote.py over the board's own window classes
    (one continuous stream in dataset order), system accuracy and the delay after
    every change of the true class (recording boundary)."""
    from host.temporal_vote import vote_sequence
    idx = sorted(rows)
    ref = vote_sequence([rows[i][0] for i in idx], k)
    board = [votes[i] for i in idx]
    vc = np.array([v[0] for v in board])
    y = labels[idx]
    return {"k": k, "windows": len(idx), "board_equal_python": int(sum(b == r for b, r in zip(board, ref))),
            "system_accuracy": float((vc == y).mean()), **transition_delays(vc, y)}


def cmd_recheck_votes(args):
    """Recompute the vote checks of an earlier fulltest from its fulltest.out
    (evidence hygiene: the first check capped transition delays at 244 windows).
    Writes vote_transition_recheck.json next to the original fulltest_check.json,
    which is left unchanged."""
    out = Path(args.run)
    rows = {}
    for line in (out / "fulltest.out").read_text().splitlines():
        if line.startswith("R "):
            kv = dict(p.split("=", 1) for p in line.split()[1:])
            rows[int(kv["idx"])] = (int(kv["cls"]), None, None)
    labels = np.loadtxt(LABELS, dtype=np.int64)
    original = json.loads((out / "fulltest_check.json").read_text())
    result = {"source": str(out / "fulltest.out"), "original_check": "fulltest_check.json (unchanged)",
              "deployment": original.get("deployment"), "input": original.get("input"),
              "original_temporal_vote": original.get("temporal_vote"),
              "temporal_vote": check_votes(vote_rows(out / "fulltest.out"), rows, labels, int(args.k))}
    (out / "vote_transition_recheck.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["temporal_vote"], indent=1))


def cmd_rt(args):
    dep = "%s/%s" % (DEPLOYMENTS, args.name)
    out, run = new_run("rt")
    # Started in the background on the board and polled, so a long (endurance)
    # run survives the SSH session and the PC sleeping; fetch the results later
    # from the printed run directory if this process stops polling.
    pid = guarded(dep, run, "python3.6 -u {run}/fpga_rt_acceptance.py {dep}/overlay/design_1_wrapper.bit {gz} "
                  "4096 {n} 200 1 1 50 {run}/out".format(run=run, dep=dep, gz=BOARD_RAW_GZ, n=args.n_total),
                  "rt", background=True)
    print("board run directory:", run, "pid", pid, flush=True)
    wait_pid(pid, poll=60)
    rc = ssh("tail -1 %s/rt.out" % run, check=False).stdout.strip()
    fetch(run + "/out", out)
    fetch(run + "/rt.out", out)
    fetch(run + "/rt_guard.log", out)
    print(rc)
    print(json.dumps(json.loads((out / "out" / "summary.json").read_text()), indent=1))


def cmd_rt_fetch(args):
    """Wait for (or just collect) an rt run started earlier: --run is the board
    run directory printed by `rt`."""
    # name only (raw_core_rt_<tag>): Git Bash rewrites a /home/... argument into a Windows path
    run = "/home/xilinx/" + args.run.replace("\\", "/").rstrip("/").split("/")[-1]
    pid = ssh("pgrep -f '[f]pga_rt_acceptance.py.*%s' | head -1" % run, check=False).stdout.strip()
    if pid:
        wait_pid(pid, poll=60)
    out = REPO / "artifacts" / "results" / Path(run).name.replace("raw_core_rt_", "raw_core_board_rt_")
    out.mkdir(parents=True, exist_ok=True)
    for f in ("out", "rt.out", "rt_guard.log"):
        fetch("%s/%s" % (run, f), out)
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
            p.add_argument("--vote-k", type=int, default=5)
            p.add_argument("--input", default=None,
                           help="local perturbed raw test set (tools/make_noisy_testset.py); "
                                "default: the clean raw test set already on the board")
            p.add_argument("--cmodel", type=Path, default=CMODEL,
                           help="C-model logits of the deployed model (default) or of a variant")
        if name == "rt":
            p.add_argument("--n-total", type=int, default=30200,
                           help="windows incl. 200 warm-up (30,200 = the 2026-10-24 criterion run)")
    v = sub.add_parser("recheck-votes")
    v.add_argument("--run", required=True, help="local results directory of an earlier fulltest")
    v.add_argument("--k", type=int, default=5)
    f = sub.add_parser("rt-fetch")
    f.add_argument("--run", required=True, help="board run directory name printed by rt (raw_core_rt_<tag>)")
    args = ap.parse_args()
    {"deploy": cmd_deploy, "golden": cmd_golden, "fulltest": cmd_fulltest, "rt": cmd_rt,
     "rt-fetch": cmd_rt_fetch, "recheck-votes": cmd_recheck_votes}[args.cmd](args)


if __name__ == "__main__":
    main()
