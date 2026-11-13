"""Q02: checks of the research quality/decision-status core (hls/research/q02_quality).

  cmodel   Q02 C-model on the Q01 v2 sample windows under every condition: logits/class
           must equal the canonical C-model, and quality_detail / quality_status /
           logit_margin must equal the independent Python reference (q01 features +
           the frozen thresholds); the 'flags' and 'flags+margin' decisions must equal Q01 v2
  replay   ordered digital replay for the board: the frozen Q01 v2 TEST bursts
           (96 bursts x 40 windows, onset 10, faults applied to the raw codes) written as a
           raw24 gz in burst order + its expected Q02 C-model rows
  compare  board fulltest.out (fpga_live_runner with the Q02 overlay) vs the Q02 C-model rows:
           logits, class and all three registers row by row; burst onset / rejection /
           recovery from the BOARD outputs; clean false flags for the clean full test

    python -m tools.research.q02_check cmodel  --out artifacts/research/Q02_20261114
    python -m tools.research.q02_check replay  --out artifacts/research/Q02_20261114
    python -m tools.research.q02_check compare --out artifacts/research/Q02_20261114 \
        --board artifacts/results/raw_core_board_fulltest_<run>/fulltest.out --kind replay|clean
Digital replay proves the integer path, not an analog front end or a real sensor.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tools.research.q01_quality import CODE_MAX, CODE_MIN, TEST, VAL, quality_features, read_windows
from tools.research.q01_quality_v2 import BURST_FAULTS, CONDITIONS, perturb

REPO = Path(__file__).resolve().parents[2]
EXE_Q02 = (REPO / "tmp/csim/batch_raw_q02.exe").resolve()
EXE_CANON = (REPO / "tmp/csim/batch_raw_argmax.exe").resolve()
Q01V2 = REPO / "artifacts/research/Q01v2_20261114"
REC = 244


def thresholds():
    from hls.research.q02_quality.make_q02_top import thresholds as t   # noqa: E402
    return t()


def run_exe(exe, windows, shards=4):
    def one(chunk):
        if not len(chunk):
            return np.zeros((0, 10), np.int64)
        out = subprocess.run([str(exe), "0", str(len(chunk))], input=chunk.astype("<i4").tobytes(),
                             capture_output=True, check=True).stdout.decode().split("\n")
        r = np.array([[int(v) for v in l.split()] for l in out if l.strip()], np.int64)
        assert len(r) == len(chunk)
        return r
    with ThreadPoolExecutor(shards) as ex:
        return np.concatenate(list(ex.map(one, np.array_split(windows, shards))))


def reference(wins, logits):
    """Independent Python reference of quality_detail / quality_status / logit_margin."""
    run_max, low_min, margin_min = thresholds()
    det, st, mg = [], [], []
    for w, lg in zip(wins, logits):
        clipped, longest, _ = quality_features(w)
        low_n = ((w & 0xFFFF) != 0).sum(axis=0)
        clip_n = ((w >= CODE_MAX) | (w <= CODE_MIN)).sum(axis=0)
        d = 0
        for c in range(8):
            d |= int(clip_n[c] > 0) << c
            d |= int(longest[c] > run_max[c]) << (8 + c)
            d |= int(low_n[c] < low_min[c]) << (16 + c)
        srt = sorted(int(v) for v in lg)
        m = srt[-1] - srt[-2]
        s = int(bool(d & 0xFF)) | int(bool(d & 0xFF00)) << 1 | int(bool(d & 0xFF0000)) << 2 | int(m < margin_min) << 3
        det.append(d), st.append(s), mg.append(m)
    return np.array(st), np.array(det), np.array(mg)


def cmd_cmodel(args):
    smp = json.loads((Q01V2 / "sample_manifest.json").read_text())
    res = {}
    for split, d, gz in (("val", VAL, "fpga_valset_raw.i32.gz"), ("test", TEST, "fpga_testset_raw.i32.gz")):
        idx, chans = np.array(smp[split]["idx"]), np.array(smp[split]["chans"])
        w = read_windows(d / gz, idx)
        y = np.loadtxt(d / "labels.txt", dtype=np.int64)[idx]
        res[split] = {}
        for cond in CONDITIONS:
            wc = np.stack([perturb(w[i], cond, chans[i], int(idx[i])) for i in range(len(w))])
            q = run_exe(EXE_Q02, wc)
            c = run_exe(EXE_CANON, wc)
            st, det, mg = reference(wc, q[:, 1:5])
            flags_ok = (det == 0)
            res[split][cond] = {
                "windows": len(wc),
                "logits_class_equal_canonical_cmodel": int((q[:, 1:6] == c[:, 1:6]).all(1).sum()),
                "status_equal_reference": int((q[:, 6] == st).sum()), "detail_equal_reference": int((q[:, 7] == det).sum()),
                "margin_equal_reference": int((q[:, 8] == mg).sum()),
                "flagged": int((~flags_ok).sum()), "low_margin": int((st & 8 != 0).sum()),
                "accepted_flags_and_margin": int((st == 0).sum()), "wrong_class": int((q[:, 5] != y).sum())}
            print(split, cond, res[split][cond], flush=True)
    # equality with the Q01 v2 'flags' and 'flags+margin' decisions
    s = json.loads((Q01V2 / "summary.json").read_text())["report"]
    agree = {sp: {c: {"flags_accepted_q01v2": s[sp]["flags"][c]["accepted"],
                      "flags_accepted_q02": res[sp][c]["windows"] - res[sp][c]["flagged"],
                      "flags+margin_accepted_q01v2": s[sp]["flags+margin"][c]["accepted"],
                      "flags+margin_accepted_q02": res[sp][c]["accepted_flags_and_margin"]} for c in CONDITIONS}
             for sp in res}
    ok = all(v["flags_accepted_q01v2"] == v["flags_accepted_q02"] and
             v["flags+margin_accepted_q01v2"] == v["flags+margin_accepted_q02"] for sp in agree.values() for v in sp.values())
    exact = all(r["logits_class_equal_canonical_cmodel"] == r["windows"] == r["status_equal_reference"] ==
                r["detail_equal_reference"] == r["margin_equal_reference"] for sp in res.values() for r in sp.values())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cmodel_check.json").write_text(json.dumps({"exact_all": exact, "decisions_equal_q01v2": ok,
                                                             "per_condition": res, "q01v2_agreement": agree}, indent=1))
    print("exact_all", exact, "decisions_equal_q01v2", ok)


def cmd_replay(args):
    bm = json.loads((Q01V2 / "burst_manifest.json").read_text())["bursts"]["test"]
    labels = np.loadtxt(TEST / "labels.txt", dtype=np.int64)
    need = sorted({b["start"] + k for b in bm for k in range(b["length"])})
    src = dict(zip(need, read_windows(TEST / "fpga_testset_raw.i32.gz", need)))
    wins, rows = [], []
    for b in bm:
        for k in range(b["length"]):
            x = src[b["start"] + k]
            faulted = b["onset"] <= k < b["onset"] + b["duration"]
            if faulted:
                x = perturb(x, b["fault"], b["channel"], b["start"] + k)
            wins.append(x)
            rows.append({"replay_index": len(rows), "burst_id": b["burst_id"], "k": k, "test_window": b["start"] + k,
                         "label": int(labels[b["start"] + k]), "faulted": bool(faulted)})
    wins = np.stack(wins)
    args.out.mkdir(parents=True, exist_ok=True)
    gz = args.out / "q02_burst_replay_test.i32.gz"
    with gzip.open(gz, "wb", compresslevel=6) as f:
        f.write(wins.astype("<i4").tobytes())
    q = run_exe(EXE_Q02, wins)
    np.savetxt(args.out / "replay_cmodel_rows.txt", q, fmt="%d")
    (args.out / "replay_manifest.json").write_text(json.dumps({
        "source": "artifacts/research/Q01v2_20261114/burst_manifest.json (test bursts)", "windows": len(wins),
        "gz": gz.name, "rows": rows}, indent=0))
    np.savetxt(args.out / "replay_labels.txt", [r["label"] for r in rows], fmt="%d")
    print("replay windows", len(wins))


def parse_fulltest(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("R "):
            kv = dict(p.split("=", 1) for p in line.split()[1:])
            rows[int(kv["idx"])] = [int(v) for v in kv["logits"].split(",")] + [int(kv["cls"]), int(kv["quality_status"]),
                                                                              int(kv["quality_detail"]), int(kv["logit_margin"])]
    return rows


def cmd_compare(args):
    rows = parse_fulltest(args.board)
    if args.kind == "replay":
        ref = np.loadtxt(args.out / "replay_cmodel_rows.txt", dtype=np.int64)
        man = json.loads((args.out / "replay_manifest.json").read_text())
    else:
        ref = np.loadtxt(args.cmodel, dtype=np.int64)
    assert sorted(rows) == list(range(len(ref))), "board rows incomplete: %d of %d" % (len(rows), len(ref))
    b = np.array([rows[i] for i in range(len(ref))], np.int64)
    eq = (b == ref[:, 1:9]).all(1)
    res = {"kind": args.kind, "board": str(args.board), "rows": len(ref), "rows_equal_q02_cmodel": int(eq.sum()),
           "logits_class_equal": int((b[:, :5] == ref[:, 1:6]).all(1).sum()),
           "registers_equal": int((b[:, 5:8] == ref[:, 6:9]).all(1).sum())}
    st = b[:, 5]
    if args.kind == "clean":
        y = np.loadtxt(TEST / "labels.txt", dtype=np.int64)
        res.update({"input_flagged_windows": int(((st & 7) != 0).sum()), "low_margin_windows": int(((st & 8) != 0).sum()),
                    "accepted_flags_and_margin": int((st == 0).sum()), "accuracy": float((b[:, 4] == y).mean()),
                    "accepted_wrong_flags_and_margin": int(((st == 0) & (b[:, 4] != y)).sum())})
    else:
        y = np.array([r["label"] for r in man["rows"]])
        acc_f = (st & 7) == 0
        acc_fm = st == 0
        per = {}
        for name, acc in (("flags", acc_f), ("flags+margin", acc_fm)):
            out = {}
            for f in BURST_FAULTS:
                for d in (20, 3):
                    sel = [r for r in man["rows"] if r["burst_id"] in {x["burst_id"] for x in
                           json.loads((Q01V2 / "burst_manifest.json").read_text())["bursts"]["test"]
                           if x["fault"] == f and x["duration"] == d}]
                    if not sel:
                        continue
                    ids = sorted({r["burst_id"] for r in sel})
                    det, rec, accw, pre = 0, 0, 0, 0
                    for bid in ids:
                        rr = [r for r in sel if r["burst_id"] == bid]
                        ix = np.array([r["replay_index"] for r in rr])
                        fl = np.array([r["faulted"] for r in rr])
                        a = acc[ix]
                        det += int((~a[fl]).any())
                        pre += int((~a[:10]).sum())
                        post = a[np.flatnonzero(fl)[-1] + 1:]
                        rec += int(post.any())
                        accw += int((a[fl] & (b[ix[fl], 4] != y[ix[fl]])).sum())
                    out["%s/d%d" % (f, d)] = {"bursts": len(ids), "detected": det, "false_reject_pre": pre,
                                              "recovered": rec, "accepted_wrong_in_burst": accw}
            per[name] = out
        res["bursts_from_board"] = per
    (args.out / ("board_compare_%s.json" % args.kind)).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "bursts_from_board"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("cmodel", "replay", "compare"):
        p = sub.add_parser(n)
        p.add_argument("--out", type=Path, required=True)
        if n == "compare":
            p.add_argument("--board", type=Path, required=True)
            p.add_argument("--kind", choices=["replay", "clean"], required=True)
            p.add_argument("--cmodel", type=Path, default=None, help="Q02 C-model full-test rows (kind clean)")
    args = ap.parse_args()
    {"cmodel": cmd_cmodel, "replay": cmd_replay, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
