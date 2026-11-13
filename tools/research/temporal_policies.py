"""T01: temporal decision policies on ordered per-window logits (offline).

Policies run over ONE continuous stream (recordings concatenated in order, as
the FPGA vote sees them):
  raw          decision = window class
  vote_k       majority of the last k classes, ties -> newest (host/temporal_vote.py, = FPGA)
  hyst_h       switch to class c only after h consecutive windows of c
  leaky_l      E <- l*E + onehot(class); switch when argmax E changes (ties keep current)
  evid_T       integer evidence with a finite horizon: per class e_c <- min(Emax, e_c + margin)
               if the window says c, else max(0, e_c - margin); switch to c when
               e_c >= T and c != current (margin = top1 - top2 integer logit, capped
               at 64; Emax = 4*T bounds the horizon)
Metrics: system accuracy and macro-F1 (steady state included), switches of the
stable decision, false switches (new decision != true label), delay after each
true-class change (searched inside the segment; unresolved counted separately),
and synthetic short events A->B(L windows)->A spliced from real windows of the
stream (controlled synthetic transitions, not measured fault onsets).
Selection on validation logits (A02 INT10 = canonical model); the chosen
operating points are then reported once on the board's test logits.

    python -m tools.research.temporal_policies --out artifacts/research/T01_20261113
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
VAL_LOGITS = REPO / "artifacts/research/A02_20261113/int10/val_outputs.npz"
TEST_RUNS = {"test_board_clean": REPO / "artifacts/results/raw_core_board_fulltest_20261111_130927/fulltest.out",
             "test_board_0dB": REPO / "artifacts/results/raw_core_board_fulltest_20261111_170226/fulltest.out"}
LABELS = REPO / "artifacts/results/fpga_rawinput_testset_20261109/labels.txt"


def policy_stream(name, param, cls, logits):
    n = len(cls)
    out = np.empty(n, np.int64)
    if name == "raw":
        return cls.copy()
    if name == "vote":
        from host.temporal_vote import vote_sequence
        return np.array([c for c, _ in vote_sequence(list(cls), param)])
    cur = int(cls[0])
    if name == "hyst":
        run_c, run_n = cur, 0
        for t, c in enumerate(cls):
            c = int(c)
            run_n = run_n + 1 if c == run_c else 1
            run_c = c
            if c != cur and run_n >= param:
                cur = c
            out[t] = cur
        return out
    if name == "leaky":
        e = np.zeros(4)
        for t, c in enumerate(cls):
            e *= param
            e[int(c)] += 1.0
            best = int(np.argmax(e))
            if e[best] > e[cur]:
                cur = best
            out[t] = cur
        return out
    if name == "evid":
        T = param
        emax = 4 * T
        e = np.zeros(4, np.int64)
        srt = np.sort(logits, axis=1)
        margin = np.minimum(srt[:, -1] - srt[:, -2], 64)
        for t, c in enumerate(cls):
            c = int(c)
            m = int(margin[t])
            for k in range(4):
                e[k] = min(emax, e[k] + m) if k == c else max(0, e[k] - m)
            if c != cur and e[c] >= T:
                cur = c
            out[t] = cur
        return out
    raise ValueError(name)


def metrics(dec, y):
    acc = float((dec == y).mean())
    f1 = []
    for c in range(4):
        tp = ((dec == c) & (y == c)).sum()
        p = tp / max((dec == c).sum(), 1)
        r = tp / max((y == c).sum(), 1)
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    sw = np.flatnonzero(dec[1:] != dec[:-1]) + 1
    false_sw = int((dec[sw] != y[sw]).sum())
    changes = [j for j in range(1, len(y)) if y[j] != y[j - 1]]
    bounds = changes + [len(y)]
    delays, unresolved = [], 0
    for j, end in zip(changes, bounds[1:]):
        hit = np.flatnonzero(dec[j:end] == y[j])
        if len(hit):
            delays.append(int(hit[0]))
        else:
            unresolved += 1
    return {"accuracy": acc, "macro_f1": float(np.mean(f1)), "switches": int(len(sw)), "false_switches": false_sw,
            "false_switches_per_10k": 1e4 * false_sw / len(y), "true_changes": len(changes),
            "delay_p50": float(np.median(delays)) if delays else None,
            "delay_p95": float(np.percentile(delays, 95)) if delays else None,
            "delay_max": int(max(delays)) if delays else None, "unresolved": unresolved}


def short_events(name, param, cls, logits, y, rng, lengths=(2, 4, 8, 16), n=200):
    """Splice L windows of a class-B recording into a stable class-A stretch."""
    res = {}
    idx_by = {c: np.flatnonzero(y == c) for c in range(4)}
    for L in lengths:
        detected, delays = 0, []
        for _ in range(n):
            a, b = rng.choice(4, 2, replace=False)
            sa = rng.choice(idx_by[a][:-60])
            sb = rng.choice(idx_by[b][:-L])
            seq = np.concatenate([np.arange(sa, sa + 40), np.arange(sb, sb + L), np.arange(sa + 40, sa + 60)])
            dec = policy_stream(name, param, cls[seq], logits[seq])
            ev = dec[40:40 + L + 20]
            hit = np.flatnonzero(ev == b)
            if len(hit):
                detected += 1
                delays.append(int(hit[0]))
        res[str(L)] = {"detected_fraction": detected / n, "delay_p50": float(np.median(delays)) if delays else None}
    return res


FAMILIES = {"raw": [None], "vote": [3, 5, 9, 15], "hyst": [2, 3, 4, 5, 6, 8],
            "leaky": [0.5, 0.6, 0.7, 0.8, 0.9], "evid": [32, 64, 96, 128, 192, 256, 384]}
STATE_BITS = {"raw": 0, "vote": 15 * 2 + 4, "hyst": 2 + 2 + 4, "leaky": "4 accumulators (fixed point)",
              "evid": "4 x 10-11 bit counters + 2-bit decision"}


def load_board(path, labels):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("R "):
            kv = dict(p.split("=", 1) for p in line.split()[1:])
            rows.append((int(kv["idx"]), int(kv["cls"]), [int(v) for v in kv["logits"].split(",")]))
    rows.sort()
    idx = np.array([r[0] for r in rows])
    return np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), labels[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from tools.research.precision_ablation import split_rows
    z = np.load(VAL_LOGITS)
    _, val_rows, yv, _ = split_rows()
    assert (z["rows"] == val_rows).all()
    lv = z["logits"].astype(np.int64)
    cv = lv.argmax(1)
    labels = np.loadtxt(LABELS, dtype=np.int64)
    streams = {"validation": (cv, lv, yv)}
    for k, p in TEST_RUNS.items():
        streams[k] = load_board(p, labels)
    result = {"id": "T01", "selection_stream": "validation", "streams": {}}
    for sname, (cls, lg, y) in streams.items():
        table = {}
        for fam, params in FAMILIES.items():
            for prm in params:
                key = fam if prm is None else "%s_%s" % (fam, prm)
                table[key] = metrics(policy_stream(fam, prm, cls, lg), y)
        result["streams"][sname] = table
    # matched comparison on validation: false-switch budget = deployed vote_5
    val = result["streams"]["validation"]
    budget = val["vote_5"]["false_switches_per_10k"]
    best = {}
    for fam in ("vote", "hyst", "leaky", "evid"):
        ok = [(k, v) for k, v in val.items() if k.startswith(fam) and v["false_switches_per_10k"] <= budget
              and v["unresolved"] <= val["vote_5"]["unresolved"]]
        if ok:
            k, v = min(ok, key=lambda kv: (kv[1]["delay_p95"], -kv[1]["macro_f1"]))
            best[fam] = {"policy": k, **v}
    simple = [best[f] for f in ("vote", "hyst") if f in best]
    ref = min(simple, key=lambda v: v["delay_p95"]) if simple else None
    cand = best.get("evid")
    gate = None
    if ref and cand:
        gate = {"best_simple": ref["policy"], "candidate": cand["policy"],
                "p95_reduction": 1 - cand["delay_p95"] / ref["delay_p95"] if ref["delay_p95"] else None,
                "macro_f1_drop_pp": 100 * (ref["macro_f1"] - cand["macro_f1"]),
                "unresolved_not_increased": cand["unresolved"] <= ref["unresolved"]}
        gate["passed"] = bool(gate["p95_reduction"] is not None and gate["p95_reduction"] >= 0.2
                              and gate["macro_f1_drop_pp"] <= 0.5 and gate["unresolved_not_increased"])
    result["matched_on_validation"] = {"false_switch_budget_per_10k": budget, "best_per_family": best, "gate_T01": gate}
    # report the selected operating points once on the board test streams + synthetic short events
    rng = np.random.default_rng(20261113)
    chosen = sorted({v["policy"] for v in best.values()} | {"raw", "vote_5"})
    result["test_report_selected"] = {s: {k: result["streams"][s][k] for k in chosen} for s in TEST_RUNS}
    cls, lg, y = streams["test_board_clean"]
    result["short_events_test_clean_synthetic"] = {}
    for k in chosen:
        fam, _, prm = k.partition("_")
        prm = None if not prm else (float(prm) if fam == "leaky" else int(prm))
        result["short_events_test_clean_synthetic"][k] = short_events(fam, prm, cls, lg, y, rng)
    result["state"] = STATE_BITS
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    print("budget (vote_5 false switches / 10k, validation): %.2f" % budget)
    for f, v in best.items():
        print("%-7s %-10s acc %.4f F1 %.4f fsw/10k %.2f p95 %s unres %d" % (f, v["policy"], v["accuracy"], v["macro_f1"],
              v["false_switches_per_10k"], v["delay_p95"], v["unresolved"]))
    print("gate:", gate)
    for s in TEST_RUNS:
        for k in chosen:
            v = result["test_report_selected"][s][k]
            print("%-17s %-10s acc %.4f F1 %.4f fsw/10k %.2f p95 %s unres %d" % (s, k, v["accuracy"], v["macro_f1"],
                  v["false_switches_per_10k"], v["delay_p95"], v["unresolved"]))
    for k, v in result["short_events_test_clean_synthetic"].items():
        print("short events", k, {L: round(e["detected_fraction"], 2) for L, e in v.items()})


if __name__ == "__main__":
    main()
