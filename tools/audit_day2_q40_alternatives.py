"""Float64-reference vs Q40-oracle equivalence for EVERY Day-2 calibration scale set.

Software-only and read-only. For each of the 12 scale sets in calibration_sets.json
it derives the constants exactly as export_hls_seq256_int10.py would (multipliers,
folded biases, residual ratios, LayerNorm affine) and converts them to Q40 with the
gen_weights_header.py rule (``int10_contract.to_q40`` on ``repr``), WITHOUT writing
any HLS data or header. It then compares the float64 reference arithmetic
(``requant_float``, ``IntegerTransformer._add``/``_layernorm``) with the Q40 oracle:

  requant   exhaustive over |acc| <= worst-case bound, all 12 single multipliers
  residual  exhaustive over all 1023^2 INT10 code pairs, both residual adds
  LayerNorm 2,000 random rows per LN (sampled)

This is NOT a hardware validation of the alternative models: no HLS export, C-model
or board run exists for them. It only shows whether the Python numbers reported for
them would equal a Q40 implementation of the same constants.

    python -m tools.audit_day2_q40_alternatives
writes artifacts/experiments/day2_calibration_sensitivity_20261120/audit/q40_equivalence_alternatives.json
"""
from __future__ import annotations

import json
import math
import argparse
from fractions import Fraction
import time
from pathlib import Path

import numpy as np

from src.quantize import int10_contract as c
from src.quantize.integer_ops import requant_float
from src.quantize.calibration_sensitivity import _new_model, _load_state, DEFAULT_CHECKPOINT

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts" / "experiments" / "day2_calibration_sensitivity_20261120"
SEQ, HD, D = 256, 8, 32
LIN = {"proj": ("input", "embed", 8), "l0.wq": ("embed", "l0.q", 32), "l0.wk": ("embed", "l0.k", 32),
       "l0.wv": ("embed", "l0.v", 32), "l0.out": ("l0.ctx", "l0.attn_out", 32),
       "l0.ff0": ("l0.ln1", "l0.ff1", 32), "l0.ff3": ("l0.ff_act", "l0.ff2", 64),
       "cls1": ("cls_ln", "cls1", 32), "cls4": ("cls_act", "logits", 16)}


def check_config(m, s, rng):
    res = {}
    # single multipliers + worst-case accumulator bounds (as in the export plan)
    mults = {}
    for n, (si, so, din) in LIN.items():
        M = (s[si] * m.sW[n]) / s[so]
        qb = np.round(m.bias[n] / (s[si] * m.sW[n])).astype(np.int64)
        amax = int((511 * np.abs(m.qW[n]).reshape(-1, din).sum(1) + np.abs(qb)).max())
        mults[n] = (M, amax)
    mults["l0.scores"] = ((s["l0.q"] * s["l0.k"]) / (math.sqrt(HD) * s["l0.scores"]), HD * 511 * 511)
    mults["l0.ctx"] = (((1.0 / 511) * s["l0.v"]) / s["l0.ctx"], SEQ * 511 * 511)
    mults["pool"] = (s["l0.ln2"] / (SEQ * s["pool"]), SEQ * 511)
    mism = {}
    for n, (M, amax) in mults.items():
        mq = c.to_q40(repr(M))
        assert amax < (1 << 31) and amax * mq < (1 << 63) - 1, n     # same width bounds
        bad = 0
        for lo in range(-amax, amax + 1, 1 << 24):
            acc = np.arange(lo, min(lo + (1 << 24), amax + 1), dtype=np.int64)
            hit = acc[c.requant_q40_np(acc, mq) != requant_float(acc, M, 10)]
            bad += int(hit.size)
            for a in hit.tolist():                         # minimal reproducible cases
                res.setdefault("cases", []).append({
                    "layer": n, "acc": a, "M_double": repr(M), "Mq40": mq,
                    "exact_acc_x_M": float(Fraction(a) * Fraction(M)),
                    "exact_acc_x_Mq_over_2^40": float(Fraction(a * mq, c.Q40_ONE)),
                    "float_reference": int(requant_float(np.array([a]), M, 10)[0]),
                    "q40": c.requant_q40(a, mq)})
        mism[n] = bad
    res["requant_mismatches"] = mism
    res["requant_values_checked"] = int(sum(2 * a + 1 for _, a in mults.values()))
    # residual adds, exhaustive
    a, b = np.meshgrid(np.arange(-511, 512), np.arange(-511, 512))
    a, b = a.ravel().astype(np.int64), b.ravel().astype(np.int64)
    rm = {}
    for name, sa, sb, so in (("res1", "embed", "l0.attn_out", "l0.res1"), ("res2", "l0.ln1", "l0.ff2", "l0.res2")):
        r0q, r1q = c.to_q40(repr(s[sa] / s[so])), c.to_q40(repr(s[sb] / s[so]))
        q = np.clip(c.div_round_even_np(a * r0q + b * r1q, c.Q40_ONE), -511, 511)
        rm[name] = int((q != m._add(a, s[sa], b, s[sb], s[so])).sum())
    res["residual_mismatches"] = rm
    # LayerNorm, sampled
    lm = {}
    for name, key, si, so in (("ln1", "encoder_layers.0.attention_norm", "l0.res1", "l0.ln1"),
                              ("ln2", "encoder_layers.0.feedforward_norm", "l0.res2", "l0.ln2"),
                              ("cls_ln", "classifier.0", "pool", "cls_ln")):
        gq = [c.to_q40(repr(float(v))) for v in m.f[key + ".weight"] / ((1 << 15) * s[so])]
        bq = [c.to_q40(repr(float(v))) for v in m.f[key + ".bias"] / s[so]]
        epsq = int(round(1e-5 / (s[si] ** 2)))
        amp = rng.choice([1, 5, 30, 150, 511], size=(2000, 1))
        rows = np.clip(np.round(rng.uniform(-1, 1, (2000, D)) * amp), -511, 511).astype(np.int64)
        ref = m._layernorm(rows, s[si], m.f[key + ".weight"], m.f[key + ".bias"], s[so])
        ora = np.array([c.layernorm_q40(r, gq, bq, epsq, m.rsqrt_lut) for r in rows])
        lm[name] = {"mismatches": int((ref != ora).sum()), "epsq": epsq}
    res["layernorm_sampled"] = lm
    res["all_equal"] = (not any(mism.values()) and not any(rm.values())
                        and not any(v["mismatches"] for v in lm.values()))
    return res


def scan(state, calib, cases_by_cfg, n_files):
    """Count how often the mismatching accumulators occur in real inference
    (validation files, hardware ingress order), per config and layer."""
    from src.config import DATA_RAW_DIR
    from src.data.make_dataloaders import build_X_y_from_file_index
    from src.data.token_aggregation import aggregate_round_div
    from src.quantize.calibration_sensitivity import _split
    _, va, _ = _split(Path(DATA_RAW_DIR))
    picks = np.linspace(0, len(va) - 1, n_files).round().astype(int)
    df = va.iloc[picks]
    X, _, _, _, _ = build_X_y_from_file_index(df, window_size=1024, stride=1024, aggregation_factor=1)
    out = {"files": [str(p) for p in df["filepath"]], "windows": int(len(X)), "configs": {}}
    for k, cases in cases_by_cfg.items():
        m = _new_model(state, calib["configs"][k]["scales"])
        targets = {}
        for e in cases:
            targets.setdefault(e["layer"], set()).add(e["acc"])
        seen = {n: 0 for n in targets}
        orig = m._linear

        def tap(q_in, name, s_in, s_out, _orig=orig):
            if name in targets:
                qW, sW, b = m.qW[name], m.sW[name], m.bias[name]
                acc = q_in @ qW.T + np.round(b / (s_in * sW)).astype(np.int64)
                seen[name] += int(np.isin(acc, list(targets[name])).sum())
            return _orig(q_in, name, s_in, s_out)
        m._linear = tap
        s_in = calib["configs"][k]["scales"]["input"]
        for i in range(0, len(X), 16):
            q = np.clip(np.round(X[i:i + 16].astype(np.float64) / s_in), -511, 511).astype(np.int64)
            m.infer_int(aggregate_round_div(q, 4).astype(np.float64) * s_in)
        out["configs"][k] = seen
        print("scan", k, seen, flush=True)
    return out


_QW: dict = {}


def _q40_init(scales):
    """Reference model whose _linear requantises with the Q40 rule (the only op
    with float/Q40 mismatches; scores/ctx/pool/residual are exhaustively equal)."""
    m = _new_model(_load_state(DEFAULT_CHECKPOINT), scales)

    def lin(q_in, name, s_in, s_out):
        qW, sW, b = m.qW[name], m.sW[name], m.bias[name]
        acc = q_in @ qW.T + np.round(b / (s_in * sW)).astype(np.int64)
        return c.requant_q40_np(acc, c.to_q40(repr((s_in * sW) / s_out)))
    m._linear = lin
    _QW["m"], _QW["s_in"] = m, scales["input"]


def _q40_file(job):
    from src.data.make_dataloaders import build_X_y_from_file_index
    from src.data.token_aggregation import aggregate_round_div
    import pandas as pd
    path, label = job
    X, _, _, _, _ = build_X_y_from_file_index(pd.DataFrame([{"filepath": path, "label": label}]),
                                              window_size=1024, stride=1024, aggregation_factor=1)
    s_in, m = _QW["s_in"], _QW["m"]
    out = []
    for i in range(0, len(X), 16):
        q = np.clip(np.round(X[i:i + 16].astype(np.float64) / s_in), -511, 511).astype(np.int64)
        out.append(m.infer_int(aggregate_round_div(q, 4).astype(np.float64) * s_in).argmax(1))
    return path, np.concatenate(out).tolist() if out else []


def q40_full_val(calib, k, workers):
    """Q40-exact predictions for config k on the full validation split vs the
    float-reference predictions saved from Kaggle."""
    import gzip
    from concurrent.futures import ProcessPoolExecutor
    from src.config import DATA_RAW_DIR
    from src.quantize.calibration_sensitivity import _split, _rel
    _, va, _ = _split(Path(DATA_RAW_DIR))
    kag = {}
    for sh in range(4):
        d = json.loads(gzip.decompress((EXP / "kaggle_val" / f"val_shard{sh}.json.gz").read_bytes()))
        for f in d["files"]:
            kag[f["path"]] = (f["pred"][k], f["y"])
    with ProcessPoolExecutor(max_workers=workers, initializer=_q40_init,
                             initargs=(calib["configs"][k]["scales"],)) as ex:
        res = list(ex.map(_q40_file, list(zip(va["filepath"], va["label"]))))
    changed = correct_q = correct_f = n = 0
    for path, pq in res:
        pf, y = kag[_rel(path, DATA_RAW_DIR)]
        pq, pf, y = np.array(pq), np.array(pf), np.array(y)
        changed += int((pq != pf).sum())
        correct_q += int((pq == y).sum())
        correct_f += int((pf == y).sum())
        n += len(y)
    return {"config": k, "windows": n, "prediction_changes_q40_vs_float_reference": changed,
            "accuracy_q40_pct": 100 * correct_q / n, "accuracy_float_reference_pct": 100 * correct_f / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q40-full-val", nargs="*", default=[],
                    help="configs to re-evaluate on the FULL validation split with Q40 linear requant")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--scan-files", type=int, default=12,
                    help="validation files scanned for the mismatching accumulators (0 = skip)")
    ap.add_argument("--append", action="store_true",
                    help="only run --q40-full-val and add it to the existing JSON")
    args = ap.parse_args()
    t0 = time.perf_counter()
    dst = EXP / "audit" / "q40_equivalence_alternatives.json"
    calib = json.loads((EXP / "calibration_sets.json").read_text())
    if args.append:
        out = json.loads(dst.read_text())
        out["q40_full_validation"] = [q40_full_val(calib, k, args.workers) for k in args.q40_full_val]
        out["q40_full_validation_seconds"] = time.perf_counter() - t0
        dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(out["q40_full_validation"])
        return
    state = _load_state(DEFAULT_CHECKPOINT)
    rng = np.random.default_rng(11)
    out = {"note": "software-only; no HLS export / C-model / board run of the alternative constants",
           "configs": {}}
    for k, cfg in calib["configs"].items():
        m = _new_model(state, cfg["scales"])               # set_bits(10) inside
        out["configs"][k] = check_config(m, cfg["scales"], rng)
        r = out["configs"][k]
        print(k, "all_equal" if r["all_equal"] else {"requant_mismatches": {n: v for n, v in r["requant_mismatches"].items() if v}}, flush=True)
    cases = {k: v["cases"] for k, v in out["configs"].items() if v.get("cases")}
    if args.scan_files and cases:
        out["occurrence_scan"] = scan(state, calib, cases, args.scan_files)
    if args.q40_full_val:
        out["q40_full_validation"] = [q40_full_val(calib, k, args.workers) for k in args.q40_full_val]
        print(out["q40_full_validation"], flush=True)
    out["seconds"] = time.perf_counter() - t0
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
