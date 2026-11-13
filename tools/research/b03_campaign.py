"""B03: physical logical-fault-injection campaign on the B03 research overlay.

Held-out manifest (seed 20261115, different from the B01/B02 software campaigns):
256 raw test windows (64 per class), 3 injection sites x 3 bit bands x 128 events,
plus one injection-off row per window. Each row is run (1) through the bit-exact
B03 C-model and (2) on the board; the board must equal the C-model row by row
(logits, class register, integrity_status), and detection is scored against the
canonical clean class of the window.
  site 1 input-buffer word (10-bit code)        bands 0-2 / 3-5 / 6-9
  site 2 projection accumulator (32 bit)        bands 0-9 / 10-20 / 21-31
  site 3 output logit of tail copy A (10 bit)   bands 0-2 / 3-5 / 6-9
Logical emulation only (no radiation / configuration-memory claims).

    python -m tools.research.b03_campaign manifest --out artifacts/research/B03_20261113
    python -m tools.research.b03_campaign cmodel   --out artifacts/research/B03_20261113
    python -m tools.research.b03_campaign board    --out artifacts/research/B03_20261113 --name research_b03_z7020_20261113
    python -m tools.research.b03_campaign analyze  --out artifacts/research/B03_20261113
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "artifacts/results/fpga_rawinput_testset_20261109/fpga_testset_raw.i32.gz"
LABELS = REPO / "artifacts/results/fpga_rawinput_testset_20261109/labels.txt"
CANON = REPO / "artifacts/results/onchip_preprocess_20261109/csim_raw_argmax/csim_logits.txt"
EXE = (REPO / "tmp/csim/batch_raw_b03.exe").resolve()
SEED, EVENTS = 20261115, 128
SITES = {1: (2048, [(0, 3), (3, 6), (6, 10)]), 2: (256 * 32, [(0, 10), (10, 21), (21, 32)]),
         3: (4, [(0, 3), (3, 6), (6, 10)])}
BANDS = ["low", "mid", "high"]


def cmd_manifest(args):
    rng = np.random.default_rng(SEED)
    y = np.loadtxt(LABELS, dtype=np.int64)
    wins = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), 64, replace=False) for c in range(4)]))
    rows = [(int(w), 0, 0, 0) for w in wins]
    for site, (size, bands) in SITES.items():
        for lo, hi in bands:
            for _ in range(EVENTS):
                rows.append((int(rng.choice(wins)), site, int(rng.integers(size)), int(rng.integers(lo, hi))))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.txt").write_text("".join("%d %d %d %d\n" % r for r in rows))
    print("rows", len(rows), "windows", len(wins))


def load_windows(idx):
    want, out, k = set(idx), {}, 0
    with gzip.open(RAW, "rb") as f:
        while len(out) < len(want):
            b = f.read(32768)
            if not b:
                break
            if k in want:
                out[k] = np.frombuffer(b, dtype="<i4").reshape(1024, 8).copy()
            k += 1
    return out


def read_manifest(out):
    return [tuple(int(v) for v in l.split()) for l in (out / "manifest.txt").read_text().splitlines() if l.strip()]


def cmd_cmodel(args):
    rows = read_manifest(args.out)
    win = load_windows([r[0] for r in rows])
    stream = np.stack([win[r[0]] for r in rows])
    inj = args.out / "cmodel_injections.txt"
    inj.write_text("".join("%d %d %d\n" % r[1:] for r in rows))
    res = subprocess.run([str(EXE), "0", str(len(rows)), str(inj.resolve())], input=stream.astype("<i4").tobytes(),
                         capture_output=True, check=True).stdout.decode()
    (args.out / "cmodel_rows.txt").write_text(res)
    print("C-model rows", len([l for l in res.splitlines() if l.strip()]))


def cmd_board(args):
    from tools.raw_core_board_test import (BOARD_RAW_GZ, DEPLOYMENTS, REPO as R, fetch, guarded, scp, ssh, wait_pid)
    from datetime import datetime
    dep = "%s/%s" % (DEPLOYMENTS, args.name)
    run = "/home/xilinx/b03_%s" % datetime.now().strftime("%Y%m%d_%H%M%S")
    ssh("mkdir %s" % run)
    scp([R / "tools/board/b03_inject_runner.py", R / "tools/board/temp_guard.sh", args.out / "manifest.txt"], run + "/")
    ssh("sed -i 's/\\r$//' %s/temp_guard.sh; chmod +x %s/temp_guard.sh" % (run, run))
    pid = guarded(dep, run, "python3.6 -u {run}/b03_inject_runner.py {dep}/overlay/design_1_wrapper.bit {gz} "
                  "{run}/manifest.txt {run}/board_rows.txt".format(run=run, dep=dep, gz=BOARD_RAW_GZ), "b03", background=True)
    wait_pid(pid, poll=15)
    for f in ("board_rows.txt", "b03.out", "b03_guard.log"):
        fetch("%s/%s" % (run, f), args.out)
    print((args.out / "b03.out").read_text()[-400:])


def parse_board(path):
    out = []
    for l in path.read_text().splitlines():
        if l.startswith("R "):
            kv = dict(p.split("=", 1) for p in l.split()[1:])
            out.append((int(kv["cls"]), [int(v) for v in kv["logits"].split(",")], int(kv["integrity"])))
    return out


def cmd_analyze(args):
    rows = read_manifest(args.out)
    cm = [[int(v) for v in l.split()] for l in (args.out / "cmodel_rows.txt").read_text().splitlines() if l.strip()]
    canon = np.loadtxt(CANON, dtype=np.int64)
    res = {"id": "B03", "seed": SEED, "rows": len(rows),
           "scope": "logical single-bit injection in the FPGA datapath (research overlay); not radiation or configuration-memory testing"}
    board = parse_board(args.out / "board_rows.txt") if (args.out / "board_rows.txt").exists() else None
    if board is not None:
        eq = sum(b[0] == c[5] and b[1] == c[1:5] and b[2] == c[6] for b, c in zip(board, cm))
        res["board_equals_cmodel_rows"] = int(eq)
        res["board_rows"] = len(board)
    src = board if board is not None else [(c[5], c[1:5], c[6]) for c in cm]
    res["source"] = "board" if board is not None else "cmodel"
    off = [(r, s) for r, s in zip(rows, src) if r[1] == 0]
    res["injection_off"] = {"rows": len(off), "class_equal_canonical": int(sum(s[0] == canon[r[0], 5] for r, s in off)),
                            "logits_equal_canonical": int(sum(s[1] == list(canon[r[0], 1:5]) for r, s in off)),
                            "integrity_nonzero": int(sum(s[2] != 0 for r, s in off))}
    by = {}
    for (w, site, word, bit), s in zip(rows, src):
        if site == 0:
            continue
        band = BANDS[[i for i, (lo, hi) in enumerate(SITES[site][1]) if lo <= bit < hi][0]]
        key = "site%d/%s" % (site, band)
        d = by.setdefault(key, {"events": 0, "logits_changed": 0, "class_changed": 0, "detected": 0, "silent": 0,
                                "detected_without_class_change": 0})
        cls_changed = s[0] != canon[w, 5]
        det = s[2] != 0
        d["events"] += 1
        d["logits_changed"] += int(s[1] != list(canon[w, 1:5]))
        d["class_changed"] += int(cls_changed)
        d["detected"] += int(det)
        d["silent"] += int(cls_changed and not det)
        d["detected_without_class_change"] += int(det and not cls_changed)
    res["by_site_band"] = by
    tot = {k: sum(v[k] for v in by.values()) for k in ("events", "class_changed", "detected", "silent")}
    res["totals"] = tot
    (args.out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k not in ("by_site_band",)}, indent=1))
    for k, v in by.items():
        print(k, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("manifest", "cmodel", "board", "analyze"):
        p = sub.add_parser(n)
        p.add_argument("--out", type=Path, required=True)
        if n == "board":
            p.add_argument("--name", required=True)
    args = ap.parse_args()
    {"manifest": cmd_manifest, "cmodel": cmd_cmodel, "board": cmd_board, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
