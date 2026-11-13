"""A01/A02: attention-probability precision versus weight/activation precision.

Research-only numerics (the canonical INT10 model, export and goldens are not
touched). A variant = (weight bits W, activation bits A, attention-probability
bits P, signed or unsigned probability codes). Every variant:
  * quantizes the SAME float checkpoint (the deployed SEQ=256 F_1layer);
  * recalibrates activation scales on the SAME 20 training files as the INT10
    export (artifacts/research/F00_20261113: split.calibration_int10_export);
  * is evaluated on the canonical validation split (195 files);
  * keeps EXP_FRAC=15, the rsqrt LUT and the scaling rules of integer_model.py.
P-unsigned uses codes 0..2^P-1 (qmax 255 for P=8) instead of the signed range.
Integer matmuls run as float64 BLAS products, which are exact here (all
|accumulators| < 2^52, asserted); `selfcheck` proves the INT10 variant equals
the canonical IntegerTransformer logit for logit before any ablation result.

    python -m tools.research.precision_ablation toy --out artifacts/research/A01_20261113
    python -m tools.research.precision_ablation selfcheck --out artifacts/research/A02_20261113
    python -m tools.research.precision_ablation run --out artifacts/research/A02_20261113 [--variants ...]
    python -m tools.research.precision_ablation summarize --out artifacts/research/A02_20261113
Validation windows come from the window cache of tools/generalization_study.py
(D:/fpga_cache/seq256_all, identical windowing/normalization/4:1 aggregation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CACHE = Path("D:/fpga_cache/seq256_all")
F00 = REPO / "artifacts" / "research" / "F00_20261113" / "frozen_baseline_manifest.json"
CKPT = REPO / "artifacts/experiments/transformer_f1_seq256_pool4/kaggle_v3/seq256_pool4/scratch/best.pt"
CANON_SCALES = REPO / "artifacts/quantized/f_seq256_int10/integer_config.json"
MODEL_CFG = {"input_dim": 8, "seq_len": 256, "num_classes": 4, "d_model": 32, "nhead": 4,
             "num_layers": 1, "dim_feedforward": 64, "dropout": 0.1}

# id: (W, A, P, P_unsigned)
VARIANTS = {
    "int10": (10, 10, 10, False), "int8": (8, 8, 8, False),
    "w8": (8, 10, 10, False), "a8": (10, 8, 10, False),
    "p8s": (10, 10, 8, False), "p9s": (10, 10, 9, False), "p8u": (10, 10, 8, True),
    "w8a8_p9s": (8, 8, 9, False), "w8a8_p10s": (8, 8, 10, False), "w8a8_p8u": (8, 8, 8, True),
}
EXACT_LIMIT = float(2 ** 52)
DIAG_EVERY = 8   # attention/context error on every 8th batch of 16 windows (~1/8 of validation)


# ---------------------------------------------------------------------------
# research model
# ---------------------------------------------------------------------------
def set_global_bits(w, a, p):
    import src.quantize.integer_model as im
    im.W_BITS, im.ACT_BITS, im.ATTN_BITS = w, a, p


def _exact_matmul(a, b):
    """Integer matmul via float64 BLAS; exact while every partial sum < 2^52."""
    bound = float(np.abs(a).max(initial=0)) * float(np.abs(b).max(initial=0)) * a.shape[-1]
    assert bound < EXACT_LIMIT, bound
    return np.rint(np.matmul(a.astype(np.float64), b.astype(np.float64))).astype(np.int64)


def make_model(state, w, a, p, unsigned, scales=None, diag=False):
    import src.quantize.integer_model as im
    from src.quantize.integer_ops import exp_lookup, qint_max, requant_float
    set_global_bits(w, a, p)

    class ResearchIntegerTransformer(im.IntegerTransformer):
        p_qmax = (1 << p) - 1 if unsigned else qint_max(p)

        def _linear(self, q_in, name, s_in, s_out):
            qW, sW, b = self.qW[name], self.sW[name], self.bias[name]
            acc = _exact_matmul(q_in, qW.T) + np.round(b / (s_in * sW)).astype(np.int64)
            return requant_float(acc, (s_in * sW) / s_out, im.ACT_BITS)

        def _softmax(self, q_scores, s_out):
            lut, clamp = self.exp_lut[self._cur_layer]
            d = np.clip(q_scores - q_scores.max(axis=-1, keepdims=True), -clamp, 0)
            num = exp_lookup(d, lut, clamp)
            denom = np.sum(num, axis=-1, keepdims=True)
            q_attn = ((num * self.p_qmax + denom // 2) // np.maximum(denom, 1)).astype(np.int64)
            if self.diag is not None:
                rows = q_attn.sum(axis=-1)                                 # (N,H,S)
                self.diag["zero_rows"] = (rows == 0).sum(axis=2)            # (N,H)
                self.diag["row_mass"] = (rows / self.p_qmax).mean(axis=2)   # (N,H)
                self.diag["nonzero_per_row"] = (q_attn > 0).sum(axis=-1).mean(axis=2)
                self.diag["prob_abs_err"] = np.full(q_attn.shape[:2], np.nan)
            if self.diag is not None and self.diag_full:
                s_sc = self.scales[im.layer_positions(self._cur_layer)["scores"]]
                z = q_scores * s_sc
                z = np.exp(z - z.max(axis=-1, keepdims=True))
                exact = z / z.sum(axis=-1, keepdims=True)
                self.diag["prob_abs_err"] = np.abs(q_attn / self.p_qmax - exact).mean(axis=(2, 3))
            return q_attn

        def _attention(self, x, s_in, l):
            pos = im.layer_positions(l)
            qh = self._split(self._linear(x, f"l{l}.wq", s_in, self.scales[pos["q"]]))
            kh = self._split(self._linear(x, f"l{l}.wk", s_in, self.scales[pos["k"]]))
            vh = self._split(self._linear(x, f"l{l}.wv", s_in, self.scales[pos["v"]]))
            acc = _exact_matmul(qh, kh.transpose(0, 1, 3, 2))
            M = (self.scales[pos["q"]] * self.scales[pos["k"]]) / (math.sqrt(self.head_dim) * self.scales[pos["scores"]])
            q_scores = requant_float(acc, M, im.ACT_BITS)
            self._cur_layer = l
            q_attn = self._softmax(q_scores, None)
            s_attn = 1.0 / self.p_qmax
            q_ctx = requant_float(_exact_matmul(q_attn, vh), (s_attn * self.scales[pos["v"]]) / self.scales[pos["ctx"]],
                                  im.ACT_BITS)
            if self.diag is not None:
                self.diag["ctx_deq"] = q_ctx * self.scales[pos["ctx"]]
            return self._linear(self._merge(q_ctx), f"l{l}.out", self.scales[pos["ctx"]], self.scales[pos["attn_out"]])

    m = ResearchIntegerTransformer(state, MODEL_CFG)
    m.diag = {} if diag else None
    m.diag_full = False   # attention/context error only on the diagnostic subset
    if scales is not None:
        m.set_scales(scales)
    return m


def float_context(m, x):
    """Float attention context of the float model (reference for context error)."""
    from src.quantize.integer_model import _softmax_np
    f, d = m.f, m.d_model
    h = x @ f["input_projection.weight"].T + f["input_projection.bias"] + m.pe[None]
    w, b = f["encoder_layers.0.self_attention.in_proj_weight"], f["encoder_layers.0.self_attention.in_proj_bias"]
    q, k, v = (m._fsplit(h @ w[i * d:(i + 1) * d].T + b[i * d:(i + 1) * d]) for i in range(3))
    return _softmax_np((q @ k.transpose(0, 1, 3, 2)) / math.sqrt(m.head_dim)) @ v


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_state():
    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    return {k: v.cpu().numpy() for k, v in ck["model_state_dict"].items()}


def split_rows():
    """Cache rows of the calibration files (export order) and of the validation files."""
    meta = np.load(CACHE / "meta.npz")
    rel = [p.replace("\\", "/").split("mafaulda/", 1)[1] for p in meta["file_path"]]
    pos = {r: i for i, r in enumerate(rel)}
    f00 = json.loads(F00.read_text())["split"]
    file_of = meta["file_of"]
    rows_of = lambda files: np.concatenate([np.flatnonzero(file_of == pos[r]) for r in files])
    calib_files = f00["calibration_int10_export"]["relpaths_in_sample_order"]
    val_files = f00["validation_files_relpaths"]
    assert hashlib.sha256("\n".join(sorted(val_files)).encode()).hexdigest() == f00["val"]["sha256_sorted_relpaths"]
    val_rows = rows_of(val_files)
    return rows_of(calib_files), val_rows, meta["file_label"][file_of[val_rows]], file_of[val_rows]


def calibrate(state, w, a, p, unsigned, x_cal):
    from src.quantize.integer_model import CalibStats
    m = make_model(state, w, a, p, unsigned)
    stats = CalibStats.empty()
    for s in range(0, len(x_cal), 8):
        m.collect(x_cal[s:s + 8].astype(np.float64), stats)
    return stats.to_scales(act_bits=a)


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
_W = {}


def _init(spec, scales):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["state"] = load_state()
    _W["model"] = make_model(_W["state"], *spec, scales=scales, diag=True)
    _W["x"] = np.load(CACHE / "x.npy", mmap_mode="r")


def _run_chunk(rows):
    m, x = _W["model"], _W["x"]
    out = {"logits": [], "zero_rows": [], "row_mass": [], "nonzero_per_row": [], "prob_abs_err": [], "ctx_rel_err": []}
    for b, i in enumerate(range(0, len(rows), 16)):
        xb = np.ascontiguousarray(x[rows[i:i + 16]]).astype(np.float64)
        m.diag_full = b % DIAG_EVERY == 0      # deterministic diagnostic subset
        out["logits"].append(m.infer_int(xb))
        for k in ("zero_rows", "row_mass", "nonzero_per_row", "prob_abs_err"):
            out[k].append(m.diag[k])
        if m.diag_full:
            cf = float_context(m, xb)
            out["ctx_rel_err"].append(np.abs(m.diag["ctx_deq"] - cf).mean(axis=(1, 2, 3)) / np.abs(cf).mean(axis=(1, 2, 3)))
        else:
            out["ctx_rel_err"].append(np.full(len(xb), np.nan))
    return {k: np.concatenate(v) for k, v in out.items()}


def run_variant(name, spec, out, workers):
    state = load_state()
    x = np.load(CACHE / "x.npy", mmap_mode="r")
    cal_rows, val_rows, _, _ = split_rows()
    t0 = time.time()
    scales = calibrate(state, *spec, np.ascontiguousarray(x[cal_rows]))
    chunks = np.array_split(val_rows, workers * 8)
    with Pool(workers, initializer=_init, initargs=(spec, scales)) as pool:
        parts = pool.map(_run_chunk, chunks)
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "scales.json").write_text(json.dumps(scales, indent=1))
    np.savez_compressed(d / "val_outputs.npz", rows=val_rows, logits=res["logits"].astype(np.int32),
                        zero_rows=res["zero_rows"].astype(np.int16), row_mass=res["row_mass"].astype(np.float32),
                        nonzero_per_row=res["nonzero_per_row"].astype(np.float32),
                        prob_abs_err=res["prob_abs_err"].astype(np.float32), ctx_rel_err=res["ctx_rel_err"].astype(np.float32))
    W, A, P, U = spec
    (d / "variant.json").write_text(json.dumps({
        "id": name, "weight_bits": W, "activation_bits": A, "prob_bits": P, "prob_unsigned": U,
        "prob_qmax": (1 << P) - 1 if U else (1 << (P - 1)) - 1, "exp_frac": 15,
        "calibration": "20 export calibration files (F00), max-abs per position, recalibrated for this variant",
        "validation_windows": int(len(val_rows)), "seconds": round(time.time() - t0, 1)}, indent=1))
    print(name, "done in %.0f s" % (time.time() - t0), flush=True)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_toy(args):
    """A01: actual _softmax on equal scores (reproduces the 2026-11-11 pilot, adds unsigned codes)."""
    from src.quantize.integer_model import IntegerTransformer
    import src.quantize.integer_model as im
    rows = []
    state = None
    for tokens in (128, 256, 512, 1024):
        for p, uns in ((8, False), (9, False), (10, False), (8, True)):
            set_global_bits(10, 10, p)
            class Dummy:
                _cur_layer = 0
                exp_lut = {0: (np.array([32768], dtype=np.int64), 0)}
            q = IntegerTransformer._softmax(Dummy(), np.zeros((1, 1, tokens, tokens), np.int64), None) if not uns else None
            if uns:   # same arithmetic with qmax = 2^P - 1
                num = np.full((1, 1, tokens, tokens), 32768, np.int64)
                denom = num.sum(axis=-1, keepdims=True)
                q = (num * 255 + denom // 2) // denom
            rows.append({"tokens": tokens, "prob_bits": p, "unsigned": uns, "code": int(q[0, 0, 0, 0]),
                         "row_sum": int(q[0, 0, 0].sum()), "zero_rows": int((q.sum(-1) == 0).sum())})
    set_global_bits(16, 16, 16)
    args.out.mkdir(parents=True, exist_ok=True)
    prior = REPO / "research_notes" / "مسیرهای پژوهشی تازه برای برد" / "hardware_pilot.json"
    agree = None
    if prior.exists():
        old = {(r["tokens"], r["signed_attention_bits"]): (r["first_code"], r["zero_rows"])
               for r in json.loads(prior.read_text(encoding="utf-8"))["actual_softmax_equal_score_pilot"]}
        agree = all(old[(r["tokens"], r["prob_bits"])] == (r["code"], r["zero_rows"]) for r in rows if not r["unsigned"])
    (args.out / "softmax_equal_scores.json").write_text(json.dumps(
        {"id": "A01", "kind": "synthetic numerical observation (not a cause of any accuracy loss)",
         "rows": rows, "matches_2026_11_11_pilot": agree}, indent=1))
    for r in rows:
        print(r)
    print("matches the 2026-11-11 pilot:", agree)


def cmd_selfcheck(args):
    """The research INT10 variant must equal the canonical model exactly."""
    from src.quantize.integer_model import IntegerTransformer
    state = load_state()
    x = np.load(CACHE / "x.npy", mmap_mode="r")
    cal_rows, val_rows, _, _ = split_rows()
    scales = calibrate(state, 10, 10, 10, False, np.ascontiguousarray(x[cal_rows]))
    canon = json.loads(CANON_SCALES.read_text())["scales"]
    scales_equal = scales == canon
    rng = np.random.default_rng(20261113)
    pick = np.sort(rng.choice(val_rows, size=args.n, replace=False))
    xb = np.ascontiguousarray(x[pick]).astype(np.float64)
    set_global_bits(10, 10, 10)
    ref = IntegerTransformer(state, MODEL_CFG, scales=canon)
    lr = np.concatenate([ref.infer_int(xb[i:i + 16]) for i in range(0, len(xb), 16)])
    res = make_model(state, 10, 10, 10, False, scales=canon, diag=True)
    lx = np.concatenate([res.infer_int(xb[i:i + 16]) for i in range(0, len(xb), 16)])
    out = {"scales_equal_canonical_export": scales_equal,
           "windows": int(len(pick)), "logits_equal_canonical": int((lr == lx).all(axis=1).sum())}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selfcheck.json").write_text(json.dumps(out, indent=1))
    print(out)
    if not scales_equal or out["logits_equal_canonical"] != len(pick):
        raise SystemExit("research INT10 variant does NOT reproduce the canonical model")


def cmd_run(args):
    if not (args.out / "selfcheck.json").exists():
        raise SystemExit("run selfcheck first")
    for name in args.variants:
        if (args.out / name / "val_outputs.npz").exists() and not args.force:
            print(name, "exists, skipped")
            continue
        run_variant(name, VARIANTS[name], args.out, args.workers)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("toy", "selfcheck", "run", "summarize"):
        p = sub.add_parser(n)
        p.add_argument("--out", type=Path, required=True)
        if n == "selfcheck":
            p.add_argument("--n", type=int, default=512)
        if n == "run":
            p.add_argument("--variants", nargs="+", default=list(VARIANTS))
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "summarize":
        from tools.research.precision_summary import summarize
        summarize(args.out)
        return
    {"toy": cmd_toy, "selfcheck": cmd_selfcheck, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
