"""T01 v2 (corrected, 2026-11-14): temporal decision policies on ordered integer logits.

Supersedes the short-event part and the gate of tools/research/temporal_policies.py
(artifacts/research/T01_20261113, kept unchanged). Corrections after the
2026-11-13 audit (reports/claude_execution_review_20261113.md, findings 1 and 4):

* one immutable event manifest per split, generated and saved BEFORE any policy
  is evaluated (seed, recording ids, start windows, labels, lengths, sha256);
  every policy runs on exactly the same events from the same fresh state;
* every segment lies inside ONE recording (244 windows per recording, recording
  = window index // 244 for the raw exports, file_of for the validation cache)
  and every window of every segment is asserted to carry the intended label;
* short events are scored separately as detected DURING the B segment, detected
  only AFTER it ended (late), or not detected; false switches outside the event
  and recovery to A are reported separately;
* an explicit maximum decision horizon H_MAX (windows) is applied to every policy:
  a step transition counts as resolved only if the decision reaches the new class
  within H_MAX windows, otherwise it is unresolved (timeout);
* "evid_T" is renamed bounded-state evidence: the 4*T cap bounds memory, NOT the
  decision time (a zero-margin stream never switches). "evidto_T_h" adds an
  explicit timeout fallback: switch to c if the raw class was c on the last h
  windows, so any persistent change with a constant raw class resolves within h.

Selection uses validation only (A02 INT10 logits = canonical numerics on the 195
validation files). The test streams (board logits) were explored before and are
reported once for the selected policies; they are not an independent confirmation.

    python -m tools.research.temporal_policies_v2 selfcheck --out artifacts/research/T01v2_20261114
    python -m tools.research.temporal_policies_v2 run --out artifacts/research/T01v2_20261114
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.research.temporal_policies import LABELS, TEST_RUNS, VAL_LOGITS, load_board

REC = 244                 # windows per MAFAULDA recording in both raw exports and the cache
H_MAX = 16                # maximum decision horizon, windows (16 x 20.48 ms = 328 ms)
PRE, STEP_B, POST = 40, 60, 60
SHORT_L = (1, 2, 4, 8, 16)
N_STEP, N_SHORT = 400, 300
SEEDS = {"validation": 20261114, "test_board_clean": 20261115, "test_board_0dB": 20261116}

FAMILIES = {"raw": [None], "vote": [3, 5, 7, 9, 15], "hyst": [2, 3, 4, 5, 6, 8],
            "leaky": [0.5, 0.6, 0.7, 0.8, 0.9], "evid": [16, 32, 64, 96, 128, 192, 256],
            "evidto": [(T, h) for T in (32, 64, 128, 256) for h in (4, 6, 8)]}


# --------------------------------------------------------------------------- policies
def policy_stream(fam, prm, cls, logits):
    """Decision sequence from a fresh state (decision = first raw class)."""
    n = len(cls)
    if fam == "raw":
        return cls.astype(np.int64).copy()
    if fam == "vote":
        from host.temporal_vote import vote_sequence
        return np.array([c for c, _ in vote_sequence(list(map(int, cls)), prm)], np.int64)
    out = np.empty(n, np.int64)
    cur = int(cls[0])
    if fam == "hyst":
        run_c, run_n = cur, 0
        for t in range(n):
            c = int(cls[t])
            run_n = run_n + 1 if c == run_c else 1
            run_c = c
            if c != cur and run_n >= prm:
                cur = c
            out[t] = cur
        return out
    if fam == "leaky":
        e = np.zeros(4)
        for t in range(n):
            e *= prm
            e[int(cls[t])] += 1.0
            best = int(np.argmax(e))
            if e[best] > e[cur]:
                cur = best
            out[t] = cur
        return out
    if fam in ("evid", "evidto"):
        T, h = (prm, None) if fam == "evid" else prm
        emax = 4 * T                       # bounds the state (memory), not the decision time
        e = np.zeros(4, np.int64)
        srt = np.sort(logits, axis=1)
        margin = np.minimum(srt[:, -1] - srt[:, -2], 64)
        run_c, run_n = cur, 0
        for t in range(n):
            c, m = int(cls[t]), int(margin[t])
            for k in range(4):
                e[k] = min(emax, e[k] + m) if k == c else max(0, e[k] - m)
            run_n = run_n + 1 if c == run_c else 1
            run_c = c
            if c != cur and (e[c] >= T or (h is not None and run_n >= h)):
                cur = c
            out[t] = cur
        return out
    raise ValueError(fam)


def key_of(fam, prm):
    if prm is None:
        return fam
    if fam == "evidto":
        return "evidto_%d_%d" % prm
    return "%s_%s" % (fam, prm)


def parse_key(k):
    fam, _, rest = k.partition("_")
    if not rest:
        return fam, None
    if fam == "evidto":
        T, h = rest.split("_")
        return fam, (int(T), int(h))
    return fam, (float(rest) if fam == "leaky" else int(rest))


# --------------------------------------------------------------------------- manifests
def recordings(rec_id, y):
    out = {}
    for r in np.unique(rec_id):
        idx = np.flatnonzero(rec_id == r)
        assert (np.diff(idx) == 1).all(), "recording %d not contiguous" % r
        lab = np.unique(y[idx])
        assert len(lab) == 1, "recording %d has several labels" % r
        out[int(r)] = (int(idx[0]), len(idx), int(lab[0]))
    return out


def make_manifest(rec_id, y, seed, split):
    rng = np.random.default_rng(seed)
    recs = recordings(rec_id, y)
    by_cls = {c: [r for r, v in recs.items() if v[2] == c] for c in range(4)}
    events = []

    def seg(c, length):
        r = int(rng.choice(by_cls[c]))
        first, n, _ = recs[r]
        s = int(rng.integers(0, n - length + 1))
        return r, first + s

    def add(kind, L, post):
        a, b = (int(v) for v in rng.choice(4, 2, replace=False))
        ra, sa = seg(a, PRE + post)
        rb, sb = seg(b, L)
        events.append({"event_id": len(events), "kind": kind, "L": L, "a": a, "b": b,
                       "rec_a": ra, "start_a": sa, "rec_b": rb, "start_b": sb, "post": post})

    for _ in range(N_STEP):
        add("step", STEP_B, 0)
    for L in SHORT_L:
        for _ in range(N_SHORT):
            add("short", L, POST)
    man = {"split": split, "seed": seed, "recording_windows": REC, "pre": PRE, "post_short": POST,
           "step_b": STEP_B, "h_max": H_MAX, "events": events}
    check_manifest(man, rec_id, y)
    man["sha256_events"] = hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()
    return man


def event_windows(ev):
    pre = np.arange(ev["start_a"], ev["start_a"] + PRE)
    b = np.arange(ev["start_b"], ev["start_b"] + ev["L"])
    post = np.arange(ev["start_a"] + PRE, ev["start_a"] + PRE + ev["post"])
    return pre, b, post


def check_manifest(man, rec_id, y):
    """Recording boundaries and intended labels A -> B (-> A) for every window."""
    ids = [e["event_id"] for e in man["events"]]
    assert ids == list(range(len(ids))), "event ids not unique/ordered"
    for ev in man["events"]:
        pre, b, post = event_windows(ev)
        assert ev["a"] != ev["b"]
        for w, rec, lab in ((pre, ev["rec_a"], ev["a"]), (b, ev["rec_b"], ev["b"]), (post, ev["rec_a"], ev["a"])):
            if len(w):
                assert (rec_id[w] == rec).all(), "segment crosses a recording boundary: event %d" % ev["event_id"]
                assert (y[w] == lab).all(), "segment label mismatch: event %d" % ev["event_id"]


# --------------------------------------------------------------------------- scoring
def score_step(dec, ev):
    """Persistent change A -> B at index PRE; resolved iff decision == b within H_MAX windows."""
    b, a = ev["b"], ev["a"]
    pre_dec = dec[:PRE]
    sw = np.flatnonzero(pre_dec[1:] != pre_dec[:-1]) + 1
    false_pre = int((pre_dec[sw] != a).sum())
    hit = np.flatnonzero(dec[PRE:] == b)
    delay = int(hit[0]) if len(hit) else None
    resolved = delay is not None and delay < H_MAX
    bseg = dec[PRE:]
    sw_b = np.flatnonzero(bseg[1:] != bseg[:-1]) + 1
    wrong_b = int(((bseg[sw_b] != b)).sum())
    return {"resolved": bool(resolved), "delay": delay if resolved else None, "false_switches_pre": false_pre,
            "wrong_switches_in_b": wrong_b}


def categorize_short(dec, ev):
    """during / after_only / none for the B event; false switches outside it; recovery to A."""
    a, b, L = ev["a"], ev["b"], ev["L"]
    during = bool((dec[PRE:PRE + L] == b).any())
    late = (not during) and bool((dec[PRE + L:PRE + L + H_MAX] == b).any())
    cat = "during" if during else ("after_only" if late else "none")
    det_delay = int(np.flatnonzero(dec[PRE:PRE + L] == b)[0]) if during else None
    pre = dec[:PRE]
    sw = np.flatnonzero(pre[1:] != pre[:-1]) + 1
    false_pre = int((pre[sw] != a).sum())
    post = dec[PRE + L:]
    sw = np.flatnonzero(post[1:] != post[:-1]) + 1
    false_post = int((post[sw] != a).sum()) + int(post[0] != a and dec[PRE + L - 1] != post[0])
    left_a = bool((dec[PRE:] != a).any())
    rec = None
    if left_a:
        back = np.flatnonzero(post == a)
        if len(back):
            # recovered = decision is a from this point to the end of the post segment
            for k in back:
                if (post[k:] == a).all():
                    rec = int(k)
                    break
    return {"category": cat, "delay": det_delay, "false_switches_pre": false_pre, "false_switches_post": false_post,
            "left_a": left_a, "recovery_delay": rec, "unrecovered": bool(left_a and rec is None)}


def run_events(man, cls, logits, fam, prm):
    res = []
    for ev in man["events"]:
        pre, b, post = event_windows(ev)
        seq = np.concatenate([pre, b, post])
        dec = policy_stream(fam, prm, cls[seq], logits[seq])
        r = score_step(dec, ev) if ev["kind"] == "step" else categorize_short(dec, ev)
        r["event_id"] = ev["event_id"]
        res.append(r)
    assert [r["event_id"] for r in res] == [e["event_id"] for e in man["events"]]
    return res


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def summarize_events(res, man):
    evs = man["events"]
    step = [r for r, e in zip(res, evs) if e["kind"] == "step"]
    d = [r["delay"] for r in step if r["resolved"]]
    out = {"step": {"events": len(step), "resolved": len(d), "unresolved_timeout": len(step) - len(d),
                    "delay_p50": pct(d, 50), "delay_p95": pct(d, 95), "delay_max": max(d) if d else None,
                    # unresolved counted at the horizon so a policy cannot gain by not deciding
                    "delay_p95_with_timeouts_at_hmax": pct([x if x is not None else H_MAX for x in
                                                            [r["delay"] for r in step]], 95),
                    "false_switches_pre": sum(r["false_switches_pre"] for r in step),
                    "wrong_switches_in_b": sum(r["wrong_switches_in_b"] for r in step)}, "short": {}}
    for L in SHORT_L:
        rs = [r for r, e in zip(res, evs) if e["kind"] == "short" and e["L"] == L]
        cats = {c: sum(r["category"] == c for r in rs) for c in ("during", "after_only", "none")}
        assert sum(cats.values()) == len(rs)
        dl = [r["delay"] for r in rs if r["delay"] is not None]
        rec = [r["recovery_delay"] for r in rs if r["recovery_delay"] is not None]
        out["short"][str(L)] = {"events": len(rs), **cats, "detected_during_fraction": cats["during"] / len(rs),
                                "delay_p50_during": pct(dl, 50),
                                "false_switches_outside_event": sum(r["false_switches_pre"] + r["false_switches_post"]
                                                                    for r in rs),
                                "left_a": sum(r["left_a"] for r in rs), "recovered": len(rec),
                                "unrecovered": sum(r["unrecovered"] for r in rs),
                                "recovery_delay_p50": pct(rec, 50), "recovery_delay_p95": pct(rec, 95)}
    return out


def stream_metrics(dec, y):
    f1 = []
    for c in range(4):
        tp = ((dec == c) & (y == c)).sum()
        p, r = tp / max((dec == c).sum(), 1), tp / max((y == c).sum(), 1)
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    sw = np.flatnonzero(dec[1:] != dec[:-1]) + 1
    false_sw = int((dec[sw] != y[sw]).sum())
    return {"accuracy": float((dec == y).mean()), "macro_f1": float(np.mean(f1)), "switches": int(len(sw)),
            "false_switches": false_sw, "false_switches_per_10k": 1e4 * false_sw / len(y)}


# --------------------------------------------------------------------------- data
def load_streams():
    from tools.research.precision_ablation import split_rows
    z = np.load(VAL_LOGITS)
    _, val_rows, yv, file_of = split_rows()
    assert (z["rows"] == val_rows).all()
    lv = z["logits"].astype(np.int64)
    labels = np.loadtxt(LABELS, dtype=np.int64)
    streams = {"validation": (lv.argmax(1), lv, yv, np.asarray(file_of))}
    for k, p in TEST_RUNS.items():
        cls, lg, y = load_board(p, labels)
        assert len(cls) == len(labels) == 95404
        streams[k] = (cls, lg, y, np.arange(len(y)) // REC)
    return streams


# --------------------------------------------------------------------------- self-check
def cmd_selfcheck(args):
    checks = {}
    # zero-margin ties: decision 1, then raw class 0 by a tie (lowest index) with margin 0
    lg = np.array([[0, 30, 0, 0]] * 10 + [[5, 5, 0, 0]] * 1000, np.int64)
    cls = lg.argmax(1)
    assert cls[-1] == 0
    d_ev = policy_stream("evid", 64, cls, lg)
    d_to = policy_stream("evidto", (64, 8), cls, lg)
    checks["zero_margin_evid_never_switches"] = bool((d_ev[10:] == 1).all())
    checks["zero_margin_timeout_switches_after_h"] = int(np.flatnonzero(d_to == 0)[0]) == 10 + 8 - 1
    # categorizer: late, during, none, recovery
    ev = {"a": 0, "b": 2, "L": 2, "kind": "short", "post": POST}
    dec = np.zeros(PRE + 2 + POST, np.int64)
    dec[PRE + 3] = 2
    c_late = categorize_short(dec, ev)
    dec2 = np.zeros_like(dec)
    dec2[PRE + 1:PRE + 5] = 2
    c_dur = categorize_short(dec2, ev)
    checks["late_detection_is_after_only"] = c_late["category"] == "after_only"
    checks["during_detection"] = c_dur["category"] == "during" and c_dur["recovery_delay"] == 3
    dec3 = np.zeros_like(dec)
    dec3[PRE:] = 2
    c_stuck = categorize_short(dec3, ev)
    checks["unrecovered_counted"] = c_stuck["unrecovered"] and c_stuck["recovery_delay"] is None
    checks["none_when_never"] = categorize_short(np.zeros_like(dec), ev)["category"] == "none"
    # events shorter than the required evidence: hyst_3 never detects L=2
    lg2 = np.array([[40, 0, 0, 0]] * PRE + [[0, 0, 40, 0]] * 2 + [[40, 0, 0, 0]] * POST, np.int64)
    d = policy_stream("hyst", 3, lg2.argmax(1), lg2)
    checks["short_event_below_required_evidence_none"] = categorize_short(d, ev)["category"] == "none"
    # step timeout: a policy that never switches is unresolved
    sev = {"a": 0, "b": 1, "kind": "step"}
    checks["step_unresolved_timeout"] = score_step(np.zeros(PRE + STEP_B, np.int64), sev)["resolved"] is False
    late_step = np.zeros(PRE + STEP_B, np.int64)
    late_step[PRE + H_MAX:] = 1
    checks["step_beyond_horizon_unresolved"] = score_step(late_step, sev)["resolved"] is False
    # manifest: recording boundaries, labels, identical identities, reproducibility
    streams = load_streams()
    cls, lg, y, rec = streams["validation"]
    m1 = make_manifest(rec, y, SEEDS["validation"], "validation")
    m2 = make_manifest(rec, y, SEEDS["validation"], "validation")
    checks["manifest_reproducible"] = m1["sha256_events"] == m2["sha256_events"]
    bad = json.loads(json.dumps(m1))
    bad["events"][0]["start_a"] = int(np.flatnonzero(rec == bad["events"][0]["rec_a"])[-1]) - 5
    try:
        check_manifest(bad, rec, y)
        checks["boundary_violation_detected"] = False
    except AssertionError:
        checks["boundary_violation_detected"] = True
    ids = {k: [r["event_id"] for r in run_events(m1, cls, lg, *parse_key(k))[:50]] for k in ("vote_5", "evid_64")}
    checks["identical_event_ids_across_policies"] = ids["vote_5"] == ids["evid_64"]
    ok = all(bool(v) for v in checks.values())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selfcheck.json").write_text(json.dumps({"passed": ok, "checks": checks}, indent=1))
    print(json.dumps(checks, indent=1))
    if not ok:
        raise SystemExit("self-check failed")


# --------------------------------------------------------------------------- main run
def cmd_run(args):
    streams = load_streams()
    args.out.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for s, (cls, lg, y, rec) in streams.items():
        p = args.out / ("event_manifest_%s.json" % s)
        man = make_manifest(rec, y, SEEDS[s], s)
        if p.exists():   # immutable: an existing manifest must be identical
            old = json.loads(p.read_text())
            assert old["sha256_events"] == man["sha256_events"], "manifest changed: %s" % p
        else:
            p.write_text(json.dumps(man, indent=0))
        manifests[s] = man
    keys = [key_of(f, prm) for f, ps in FAMILIES.items() for prm in ps]
    result = {"id": "T01v2", "h_max_windows": H_MAX, "window_ms": 20.48,
              "manifests": {s: {"path": "event_manifest_%s.json" % s, "seed": m["seed"], "sha256_events": m["sha256_events"],
                                "events": len(m["events"])} for s, m in manifests.items()},
              "supersedes": "artifacts/research/T01_20261113 short events and gate_T01", "streams": {}, "events": {}}
    for s, (cls, lg, y, rec) in streams.items():
        if s != "validation" and not args.all_on_test:
            continue
        result["streams"][s] = {k: stream_metrics(policy_stream(*parse_key(k), cls, lg), y) for k in keys}
        result["events"][s] = {k: summarize_events(run_events(manifests[s], cls, lg, *parse_key(k)), manifests[s])
                               for k in keys}
        print("done", s, flush=True)
    val_s, val_e = result["streams"]["validation"], result["events"]["validation"]
    budget = val_s["vote_5"]["false_switches_per_10k"]          # the deployed policy sets the budget

    def admissible(k):
        return val_s[k]["false_switches_per_10k"] <= budget

    best = {}
    for fam in FAMILIES:
        if fam == "raw":
            continue
        ok = [k for k in keys if parse_key(k)[0] == fam and admissible(k)]
        if ok:
            k = min(ok, key=lambda k: (val_e[k]["step"]["unresolved_timeout"],
                                       val_e[k]["step"]["delay_p95_with_timeouts_at_hmax"], -val_s[k]["macro_f1"]))
            best[fam] = k
    simple = [best[f] for f in ("vote", "hyst") if f in best]
    ref = min(simple, key=lambda k: (val_e[k]["step"]["unresolved_timeout"],
                                     val_e[k]["step"]["delay_p95_with_timeouts_at_hmax"])) if simple else None
    gates = {}
    for cand_fam in ("evid", "evidto", "leaky"):
        cand = best.get(cand_fam)
        if not (ref and cand):
            continue
        r_p95 = val_e[ref]["step"]["delay_p95_with_timeouts_at_hmax"]
        c_p95 = val_e[cand]["step"]["delay_p95_with_timeouts_at_hmax"]
        g = {"best_simple": ref, "candidate": cand, "simple_p95": r_p95, "candidate_p95": c_p95,
             "p95_reduction": (1 - c_p95 / r_p95) if r_p95 else (0.0 if c_p95 == 0 else -1.0),
             "macro_f1_drop_pp": 100 * (val_s[ref]["macro_f1"] - val_s[cand]["macro_f1"]),
             "unresolved_simple": val_e[ref]["step"]["unresolved_timeout"],
             "unresolved_candidate": val_e[cand]["step"]["unresolved_timeout"]}
        hy = best.get("hyst")
        g["simple_hysteresis_equal"] = bool(hy and val_e[hy]["step"]["delay_p95_with_timeouts_at_hmax"] <= c_p95)
        g["passed"] = bool(g["p95_reduction"] >= 0.2 and g["macro_f1_drop_pp"] <= 0.5
                           and g["unresolved_candidate"] <= g["unresolved_simple"] and not g["simple_hysteresis_equal"])
        gates[cand_fam] = g
    result["selection_validation"] = {"false_switch_budget_per_10k": budget, "best_per_family": best,
                                      "gate_T01": gates, "fpga_go": any(g["passed"] for g in gates.values())}
    chosen = sorted(set(best.values()) | {"raw", "vote_5"})
    result["test_report_selected_previously_explored"] = {}
    for s in streams:
        if s == "validation":
            continue
        cls, lg, y, rec = streams[s]
        result["test_report_selected_previously_explored"][s] = {
            k: {"stream": stream_metrics(policy_stream(*parse_key(k), cls, lg), y),
                "events": summarize_events(run_events(manifests[s], cls, lg, *parse_key(k)), manifests[s])}
            for k in chosen}
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    print("budget (vote_5 false switches/10k, validation): %.2f" % budget)
    for f, k in best.items():
        st, ev = val_s[k], val_e[k]
        print("%-7s %-14s F1 %.4f fsw/10k %5.2f step p95(h) %s unres %d | short L=2 during %.2f L=4 %.2f" % (
            f, k, st["macro_f1"], st["false_switches_per_10k"], ev["step"]["delay_p95_with_timeouts_at_hmax"],
            ev["step"]["unresolved_timeout"], ev["short"]["2"]["detected_during_fraction"],
            ev["short"]["4"]["detected_during_fraction"]))
    print("gates:", json.dumps(gates, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("selfcheck", "run"):
        p = sub.add_parser(n)
        p.add_argument("--out", type=Path, required=True)
        if n == "run":
            p.add_argument("--all-on-test", action="store_true",
                           help="also tabulate every policy on the (previously explored) test streams")
    args = ap.parse_args()
    {"selfcheck": cmd_selfcheck, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
