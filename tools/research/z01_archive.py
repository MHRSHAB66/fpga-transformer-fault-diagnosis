"""Z01: archive the scarce physical evidence and its reproduction identity, then make
and verify an off-repository copy.

  index    artifacts/research/Z01_20261114/archive_index.json: every archived file with
           size, sha256 and category (bitstreams/HWH/XSA + post-route reports, sources and
           generators, configs/contracts/goldens, ordered replay inputs and event/burst
           manifests, board logits/classes/status/timing/guard traces, failures and aborted
           runs, docs); plus environment.json (tool, runtime and board versions)
  backup   copy every indexed file (same relative path) and a git bundle of all refs to
           --dest, then re-hash every copy against the index (archive_verify.json)

    python -m tools.research.z01_archive index
    python -m tools.research.z01_archive backup --dest E:/fpga_fault_diagnosis_backup/Z01_20261114
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/research/Z01_20261114"

TRACKED_ROOTS = ["artifacts/research", "artifacts/deploy", "artifacts/calibration", "artifacts/preprocessing",
                 "artifacts/manifests", "artifacts/quantized/f_seq256_int10", "hls/research", "hls/f1_int10_seq256",
                 "hls/f1_int10/gpp_csim", "docs/research", "docs/reports", "tools", "pynq", "host", "src/quantize",
                 "src/data", "reports", "AGENTS.md", "CLAUDE.md", "requirements.txt"]
RESULT_PREFIXES = ["artifacts/results/raw_core_board_", "artifacts/results/onchip_preprocess_20261109",
                   "artifacts/results/fpga_rawinput_", "artifacts/results/noisy_testsets_", "artifacts/results/robustness_",
                   "artifacts/results/seq256_dma_vote_z7020_20261110", "artifacts/results/hls_rt_dataflow_FAILED"]
FAILURE_MARKERS = ("timing_fail", "_aborted", "lag_padded_artifact", "_failed")


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout


def category(rel):
    r = rel.lower()
    if any(m in r for m in FAILURE_MARKERS):
        return "failures_and_aborts"
    if r.endswith((".bit", ".hwh", ".xsa")) or "vivado_reports" in r or "hls_reports" in r or r.endswith(".rpt"):
        return "bit_hwh_xsa_and_route"
    if r.startswith(("artifacts/results/raw_core_board_", "artifacts/research/s01")) or r.endswith(("rt_raw.npz", "fulltest.out")):
        return "board_traces"
    if r.endswith(".i32.gz") or "manifest" in r and r.endswith((".json", ".txt")):
        return "replay_inputs_and_manifests"
    if r.startswith(("hls/", "tools/", "pynq/", "host/", "src/")):
        return "sources"
    if "golden" in r or "contract" in r or "config" in r or r.startswith("artifacts/calibration"):
        return "references_configs_goldens"
    return "docs_and_results"


def collect():
    files = set()
    for line in git("ls-files", "-z").split("\0"):
        if line and any(line == r or line.startswith(r.rstrip("/") + "/") for r in TRACKED_ROOTS + RESULT_PREFIXES):
            files.add(line)
        elif line and any(line.startswith(p) for p in RESULT_PREFIXES):
            files.add(line)
    man = json.loads((REPO / "artifacts/manifests/artifact_manifest.json").read_text())
    for a in man["artifacts"]:
        if (REPO / a["path"]).exists():
            files.add(a["path"])
    # untracked-but-kept research outputs (gitignored binaries, failure folders)
    for root in ("artifacts/research", "artifacts/deploy", "artifacts/results/seq256_dma_vote_z7020_20261110",
                 "artifacts/results/hls_rt_dataflow_FAILED"):
        for p in (REPO / root).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                files.add(p.relative_to(REPO).as_posix())
    return sorted(f for f in files if (REPO / f).is_file())


def environment():
    def cmd(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=120, shell=isinstance(c, str)).stdout.strip()[:400]
        except Exception as e:   # noqa: BLE001
            return "unavailable: %s" % e
    import numpy
    env = {"generated": datetime.now().isoformat(timespec="seconds"), "git_head": git("rev-parse", "HEAD").strip(),
           "git_status_short": git("status", "--short"), "python": platform.python_version(), "numpy": numpy.__version__,
           "platform": platform.platform(),
           "gpp": cmd(["C:/Users/mine_pc/scoop/apps/gcc/current/bin/g++", "--version"]).splitlines()[0:1],
           "vivado": "Vivado/Vitis HLS 2020.2 at G:/Xilinx (version banner of the build logs in tmp/*_core_build)",
           "board": cmd('ssh -i %s -o BatchMode=yes -o ConnectTimeout=5 root@192.168.2.99 '
                        '"uname -a; python3.6 --version; python3.6 -c \'import pynq; print(pynq.__version__)\'; '
                        'sha256sum /home/xilinx/raw_testset_20261109/fpga_testset_raw.i32.gz"'
                        % (Path.home() / ".ssh/id_zturn").as_posix())}
    return env


def cmd_index(_args):
    OUT.mkdir(parents=True, exist_ok=True)
    files = collect()
    idx = []
    for f in files:
        p = REPO / f
        idx.append({"path": f, "bytes": p.stat().st_size, "sha256": sha256(p), "category": category(f)})
    cats = {}
    for e in idx:
        c = cats.setdefault(e["category"], {"files": 0, "bytes": 0})
        c["files"] += 1
        c["bytes"] += e["bytes"]
    (OUT / "archive_index.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                        "files": len(idx), "bytes": sum(e["bytes"] for e in idx),
                                                        "categories": cats, "entries": idx}, indent=0))
    (OUT / "environment.json").write_text(json.dumps(environment(), indent=1))
    print(json.dumps(cats, indent=1), len(idx), "files")


def cmd_backup(args):
    dest = Path(args.dest)
    index = json.loads((OUT / "archive_index.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    for e in index["entries"]:
        d = dest / "files" / e["path"]
        d.parent.mkdir(parents=True, exist_ok=True)
        if not (d.exists() and d.stat().st_size == e["bytes"]):
            shutil.copy2(REPO / e["path"], d)
    for name in ("archive_index.json", "environment.json"):
        shutil.copy2(OUT / name, dest / name)
    bundle = dest / "repository.bundle"
    subprocess.run(["git", "-C", str(REPO), "bundle", "create", str(bundle), "--all"], check=True,
                   capture_output=True)
    ok, bad = 0, []
    for e in index["entries"]:
        d = dest / "files" / e["path"]
        if d.exists() and d.stat().st_size == e["bytes"] and sha256(d) == e["sha256"]:
            ok += 1
        else:
            bad.append(e["path"])
    bv = subprocess.run(["git", "bundle", "verify", str(bundle)], capture_output=True, text=True, cwd=str(REPO))
    res = {"dest": str(dest), "verified": datetime.now().isoformat(timespec="seconds"), "files_ok": ok,
           "files_total": len(index["entries"]), "mismatches": bad, "git_bundle": bundle.name,
           "git_bundle_sha256": sha256(bundle), "git_bundle_verify_rc": bv.returncode,
           "git_bundle_verify": (bv.stdout + bv.stderr).strip()[-300:],
           "same_physical_disk_as_repo": args.same_disk,
           "note": "copy on another partition of the SAME physical SSD protects against repository loss, not "
                   "against disk failure; an external drive or cloud copy is still required" if args.same_disk else ""}
    (OUT / "archive_verify.json").write_text(json.dumps(res, indent=1))
    shutil.copy2(OUT / "archive_verify.json", dest / "archive_verify.json")
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=1), "mismatches", len(bad))
    if bad or bv.returncode:
        raise SystemExit("backup verification FAILED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    b = sub.add_parser("backup")
    b.add_argument("--dest", required=True)
    b.add_argument("--same-disk", action="store_true", help="destination is on the repository's physical disk")
    args = ap.parse_args()
    {"index": cmd_index, "backup": cmd_backup}[args.cmd](args)


if __name__ == "__main__":
    main()
