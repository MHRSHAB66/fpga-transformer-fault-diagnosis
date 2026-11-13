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
    python -m tools.research.b03_campaign cmodel   --out artifacts/research/B03_20261113 [--lp64]
    python -m tools.research.b03_campaign board    --out artifacts/research/B03_20261113 --name research_b03_z7020_20261113
    python -m tools.research.b03_campaign analyze  --out artifacts/research/B03_20261113
B04 (same sites on the B04i overlay, new held-out seed):
    python -m tools.research.b03_campaign manifest --out artifacts/research/B04_20261114/heldout --seed 20261116
    python -m tools.research.b03_campaign cmodel   --out artifacts/research/B04_20261114/heldout --exe tmp/csim/batch_raw_b04i.exe
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "artifacts/results/fpga_rawinput_testset_20261109/fpga_testset_raw.i32.gz"
LABELS = REPO / "artifacts/results/fpga_rawinput_testset_20261109/labels.txt"
CANON = REPO / "artifacts/results/onchip_preprocess_20261109/csim_raw_argmax/csim_logits.txt"
EXE = (REPO / "tmp/csim/batch_raw_b03.exe").resolve()
# 64-bit-long build (hls/research/b03_inject/make_lp64_cmodel.py): MinGW long is 32 bit, the synthesized long 64 bit
EXE_LP64 = (REPO / "tmp/csim/batch_raw_b03_lp64.exe").resolve()
SEED, EVENTS = 20261115, 128
SITES = {1: (2048, [(0, 3), (3, 6), (6, 10)]), 2: (256 * 32, [(0, 10), (10, 21), (21, 32)]),
         3: (4, [(0, 3), (3, 6), (6, 10)])}
BANDS = ["low", "mid", "high"]


def cmd_manifest(args):
    rng = np.random.default_rng(args.seed)
    y = np.loadtxt(LABELS, dtype=np.int64)
    wins = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), 64, replace=False) for c in range(4)]))
    rows = [(int(w), 0, 0, 0) for w in wins]
    if args.recovery:
        # recovery probes: every injected window is followed by a clean window of another
        # test window; the clean one must be exact and unflagged (no state carried over)
        rows = []
        for site, (size, bands) in SITES.items():
            for lo, hi in bands:
                for _ in range(args.recovery):
                    w_inj, w_clean = (int(v) for v in rng.choice(wins, 2, replace=False))
                    rows.append((w_inj, site, int(rng.integers(size)), int(rng.integers(lo, hi))))
                    rows.append((w_clean, 0, 0, 0))
    else:
        for site, (size, bands) in SITES.items():
            for lo, hi in bands:
                for _ in range(EVENTS):
                    rows.append((int(rng.choice(wins)), site, int(rng.integers(size)), int(rng.integers(lo, hi))))
    args.out.mkdir(parents=True, exist_ok=True)
    text = "".join("%d %d %d %d\n" % r for r in rows)
    (args.out / "manifest.txt").write_text(text)
    (args.out / "manifest_meta.json").write_text(json.dumps({
        "seed": args.seed, "windows": len(wins), "rows": len(rows),
        "events_per_site_band": args.recovery or EVENTS, "recovery_pairs": bool(args.recovery),
        "sites": {str(k): {"words": v[0], "bit_bands": v[1]} for k, v in SITES.items()},
        "window_selection": "64 test windows per class drawn with this seed; overlap with other campaigns is "
                            "reported separately (a different seed alone does not make the samples independent)",
        "sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=1))
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
    exe, name = (EXE_LP64, "cmodel_rows_lp64.txt") if args.lp64 else (EXE, "cmodel_rows.txt")
    if args.exe is not None:
        exe = args.exe.resolve()
    res = subprocess.run([str(exe), "0", str(len(rows)), str(inj.resolve())], input=stream.astype("<i4").tobytes(),
                         capture_output=True, check=True).stdout.decode()
    (args.out / name).write_text(res)
    m = meta(args.out, "cmodel_meta.json") or {}
    m[name] = {"exe": str(exe), "exe_sha256": hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
               "long_model": "LP64 (64-bit long, synthesized semantics)" if args.lp64 else "MinGW LLP64 (32-bit long)",
               "rows": len([l for l in res.splitlines() if l.strip()])}
    (args.out / "cmodel_meta.json").write_text(json.dumps(m, indent=1))
    print("C-model rows", m[name]["rows"])


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
    bit = ssh("sha256sum %s/overlay/design_1_wrapper.bit" % dep, check=False).stdout.split()
    (args.out / "board_meta.json").write_text(json.dumps({
        "deployment": dep, "board_run_dir": run, "board_bitstream_sha256": bit[0] if bit else None,
        "runner": "tools/board/b03_inject_runner.py"}, indent=1))
    print((args.out / "b03.out").read_text()[-400:])


def parse_board(path, rows):
    """Board rows in manifest order; every field of every row must match the manifest."""
    out = []
    for l in path.read_text().splitlines():
        if l.startswith("R "):
            kv = dict(p.split("=", 1) for p in l.split()[1:])
            k = int(kv["row"])
            assert k == len(out), "board row ids not consecutive at %d" % k
            assert (int(kv["idx"]), int(kv["site"]), int(kv["word"]), int(kv["bit"])) == rows[k], \
                "board row %d does not match the manifest" % k
            out.append((int(kv["cls"]), [int(v) for v in kv["logits"].split(",")], int(kv["integrity"])))
    assert len(out) == len(rows), "board rows %d != manifest rows %d (incomplete run)" % (len(out), len(rows))
    return out


def read_cmodel(path, rows):
    cm = [[int(v) for v in l.split()] for l in path.read_text().splitlines() if l.strip()]
    assert len(cm) == len(rows), "%s: %d rows != manifest %d" % (path.name, len(cm), len(rows))
    assert [c[0] for c in cm] == list(range(len(rows))), "%s: row ids not 0..n-1" % path.name
    return cm


def meta(out, name):
    p = out / name
    return json.loads(p.read_text()) if p.exists() else None


def cmd_analyze(args):
    rows = read_manifest(args.out)
    cm = read_cmodel(args.out / "cmodel_rows.txt", rows)
    canon = np.loadtxt(CANON, dtype=np.int64)
    mm, cmeta = meta(args.out, "manifest_meta.json"), meta(args.out, "cmodel_meta.json")
    res = {"id": args.id, "rows": len(rows), "manifest_meta": mm, "cmodel_meta": cmeta,
           "seed": mm["seed"] if mm else "unrecorded in this directory (B03 manifest: seed %d)" % SEED,
           "scope": "logical single-bit injection at the three defined sites, transient (one window, one bit); "
                    "not radiation, configuration-memory or field upset-rate testing",
           "canonical_reference": "artifacts/results/onchip_preprocess_20261109/csim_raw_argmax/csim_logits.txt"}
    lp = args.out / "cmodel_rows_lp64.txt"
    cl = read_cmodel(lp, rows) if lp.exists() else None
    if cl:
        ref, res["selected_reference"] = cl, "cmodel_rows_lp64.txt (64-bit long = synthesized semantics)"
        res["cmodel_vs_lp64_differing_rows"] = [k for k, (a, c) in enumerate(zip(cm, cl)) if a != c]
    else:
        ref, res["selected_reference"] = cm, "cmodel_rows.txt"
    board = parse_board(args.out / "board_rows.txt", rows) if (args.out / "board_rows.txt").exists() else None
    if board is not None:
        res["board_rows"] = len(board)
        res["board_equals_cmodel_rows"] = int(sum(b[0] == c[5] and b[1] == c[1:5] and b[2] == c[6]
                                                  for b, c in zip(board, cm)))
        res["board_equals_selected_reference_rows"] = int(sum(b[0] == c[5] and b[1] == c[1:5] and b[2] == c[6]
                                                              for b, c in zip(board, ref)))
        res["board_meta"] = meta(args.out, "board_meta.json")
    src = board if board is not None else [(c[5], c[1:5], c[6]) for c in ref]
    res["source"] = "board" if board is not None else "cmodel (%s)" % res["selected_reference"]
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
    keys = ("events", "logits_changed", "class_changed", "detected", "silent", "detected_without_class_change")
    tot = {k: sum(v[k] for v in by.values()) for k in keys}
    tot["invalid_result_coverage"] = (1 - tot["silent"] / tot["class_changed"]) if tot["class_changed"] else None
    changed = [s for r, s in zip(rows, src) if r[1] and s[1] != list(canon[r[0], 1:5])]
    tot["corrupted_logit_coverage"] = (sum(s[2] != 0 for s in changed) / len(changed)) if changed else None
    tot["availability_cost_flags_without_class_change"] = tot["detected_without_class_change"]
    tot["note"] = ("a flag marks the window's result invalid before the deadline; it is NOT a correct "
                   "classification of that window")
    res["totals"] = tot
    (args.out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k not in ("by_site_band", "manifest_meta", "cmodel_meta", "board_meta")},
                     indent=1))
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
        if n == "analyze":
            p.add_argument("--id", default="B03", help="campaign id recorded in summary.json (B03, B04i, ...)")
        if n == "manifest":
            p.add_argument("--seed", type=int, default=SEED, help="manifest seed (B03: %d)" % SEED)
            p.add_argument("--recovery", type=int, default=0,
                           help="recovery manifest: N (injected, clean) pairs per site/band instead of a campaign")
        if n == "cmodel":
            p.add_argument("--lp64", action="store_true", help="use the 64-bit-long C-model build")
            p.add_argument("--exe", type=Path, default=None,
                           help="C-model of another injection overlay (e.g. tmp/csim/batch_raw_b04i.exe)")
    args = ap.parse_args()
    {"manifest": cmd_manifest, "cmodel": cmd_cmodel, "board": cmd_board, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
