"""
Day-2 calibration-sensitivity sweep for the deployed SEQ=256 / pool4 / INT10 model.

EVALUATION ONLY. It never writes the deployed scales, HLS data, headers, golden
vectors or the checkpoint; the deployed 20-file calibration stays the reference.

Three sub-commands:

  calibrate  (train split only) For every (seed, n_files) draw
             ``train_df.sample(n=n_files, random_state=seed)`` -- the exact rule of
             export_hls_seq256_int10.py (seed 42, 20 files) and of the historical
             12-file evaluator (seed 42, 12 files). pandas sampling is a prefix of
             one permutation, so for a fixed seed the sets are NESTED (12 c 20 c 40
             c 80); this is asserted. Scales come from the approved float path:
             1024-sample z-scored windows -> float 4:1 mean -> IntegerTransformer.collect
             -> max-abs / 511. The seed-42/20-file scales are asserted bit-identical
             to artifacts/quantized/f_seq256_int10/integer_config.json.

  evaluate   One shard of the VALIDATION split (by file), streaming file by file:
             1024 normalized samples -> INT10 quantize (pre-clip counted) ->
             aggregate_round_div 4:1 -> 256-token IntegerTransformer, once per scale
             set, plus the float reference (float mean -> float model). An
             instrumented subclass records PRE-CLIP saturation at every requant
             position; its logits are asserted identical to the stock infer_int.

  combine    Merge shards -> accuracy, macro-F1, per-class recall, agreement and
             changed predictions vs the canonical model, saturation rates, scale
             deltas. Writes JSON + CSV.

Run (see docs/reports/day2_quantization_validation_20261120.md):
    python -m src.quantize.calibration_sensitivity calibrate --out <dir>
    python -m src.quantize.calibration_sensitivity evaluate --scales <dir>/calibration_sets.json \
        --shard 0 --num-shards 1 --workers 4 --out <dir>/val_shard0.json
    python -m src.quantize.calibration_sensitivity combine --calib <dir>/calibration_sets.json \
        --shards <dir>/val_shard*.json --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (PROJECT_ROOT / "artifacts" / "experiments" / "transformer_f1_seq256_pool4"
                      / "kaggle_v3" / "seq256_pool4" / "scratch" / "best.pt")
EXPORT_CONFIG = PROJECT_ROOT / "artifacts" / "quantized" / "f_seq256_int10" / "integer_config.json"
MODEL_CFG = {"input_dim": 8, "seq_len": 256, "num_classes": 4, "d_model": 32, "nhead": 4,
             "num_layers": 1, "dim_feedforward": 64, "dropout": 0.1}
BITS = 10
QMAX = 511
CANONICAL = "s42_n20"
CLASS_NAMES = ["normal", "imbalance", "misalignment", "bearing_fault"]
POSITIONS = ["input", "embed_proj", "embed", "l0.q", "l0.k", "l0.v", "l0.scores", "l0.ctx",
             "l0.attn_out", "l0.res1", "l0.ln1", "l0.ff1", "l0.ff_act", "l0.ff2", "l0.res2",
             "l0.ln2", "pool", "cls_ln", "cls1", "cls_act", "logits"]


def peak_rss_mb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if sys.platform != "darwin" else r / 2 ** 20
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2 ** 20


def _split(raw_dir: Path):
    from src.config import LABEL_MAP
    from src.data.load_data import build_file_index
    from src.data.make_dataloaders import split_file_index
    idx = build_file_index(raw_dir)
    idx = idx[idx["label"].isin(LABEL_MAP.keys())].copy()
    return split_file_index(idx)


def _rel(p, raw_dir: Path) -> str:
    return Path(p).resolve().relative_to(Path(raw_dir).resolve()).as_posix()


def _load_state(checkpoint: Path) -> dict:
    import torch
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {k: v.cpu().numpy() for k, v in ck["model_state_dict"].items()}


def _new_model(state: dict, scales: dict | None = None, instrumented: bool = False):
    from src.quantize.evaluate_integer_bits import set_bits
    from src.quantize.integer_model import IntegerTransformer
    set_bits(BITS)                                   # before construction: 10-bit weights
    cls = _instrumented_class() if instrumented else IntegerTransformer
    m = cls(state, MODEL_CFG)
    if scales is not None:
        m.set_scales(scales)
    return m


# ---------------------------------------------------------------------------
# instrumented reference (diagnostic only; stock IntegerTransformer untouched)
# ---------------------------------------------------------------------------

def _instrumented_class():
    import src.quantize.integer_model as im
    from src.quantize.integer_model import IntegerTransformer, layer_positions
    from src.quantize.integer_ops import quantize_sym, exp_lookup, build_gelu_lut  # noqa: F401

    class InstrumentedIntegerTransformer(IntegerTransformer):
        """Same arithmetic as IntegerTransformer.infer_int, with every pre-clip
        integer value counted against +-511 before it is clipped."""

        def reset_counts(self):
            self.sat = {p: np.zeros(5, dtype=np.int64) for p in POSITIONS}
            # [values, pre-clip > +511, pre-clip < -511, |post-clip| == 511, windows with any]

        def _cnt(self, tag, y):
            # y is integer-valued, so |clip(y)| == 511  <=>  |y| >= 511
            mag = np.abs(y)
            beyond = mag > QMAX
            a = self.sat[tag]
            a[0] += y.size
            a[3] += int(np.count_nonzero(mag >= QMAX))
            nb = int(np.count_nonzero(beyond))
            if nb:
                npos = int(np.count_nonzero(y[beyond] > 0))
                a[1] += npos
                a[2] += nb - npos
                a[4] += int(beyond.reshape(y.shape[0], -1).any(axis=1).sum())

        def _rq(self, tag, acc, m):
            y = np.round(acc.astype(np.float64) * m)
            self._cnt(tag, y)
            return np.clip(y, -QMAX, QMAX).astype(np.int64)

        def _lin(self, tag, q_in, name, s_in, s_out):
            qW, sW, b = self.qW[name], self.sW[name], self.bias[name]
            acc = q_in @ qW.T + np.round(b / (s_in * sW)).astype(np.int64)
            return self._rq(tag, acc, (s_in * sW) / s_out)

        def _addi(self, tag, qa, sa, qb, sb, s_out):
            y = np.round(qa.astype(np.float64) * (sa / s_out) + qb.astype(np.float64) * (sb / s_out))
            self._cnt(tag, y)
            return np.clip(y, -QMAX, QMAX).astype(np.int64)

        def _lni(self, tag, q, s_in, gamma, beta, s_out):
            from src.quantize.integer_ops import rsqrt_fxp
            D = q.shape[-1]
            mean = np.round(np.sum(q, axis=-1, keepdims=True).astype(np.float64) / D).astype(np.int64)
            centered = q - mean
            var = np.round(np.sum(centered.astype(np.int64) ** 2, axis=-1, keepdims=True)
                           .astype(np.float64) / D).astype(np.int64)
            epsq = int(round(im.LN_EPS / (s_in * s_in)))
            inv, frac = rsqrt_fxp(var + epsq, self.rsqrt_lut)
            norm = centered.astype(np.int64) * inv
            y = np.round(norm.astype(np.float64) * (gamma / ((1 << frac) * s_out)) + beta / s_out)
            self._cnt(tag, y)
            return np.clip(y, -QMAX, QMAX).astype(np.int64)

        def _gelu_raw(self, si, so):
            import math as _m
            codes = np.arange(-QMAX, QMAX + 1, dtype=np.float64)
            x = codes * si
            return np.round(0.5 * x * (1.0 + np.vectorize(_m.erf)(x / _m.sqrt(2.0))) / so)

        def _build_luts(self):
            super()._build_luts()
            s = self.scales
            self._gelu_raw_l0 = self._gelu_raw(s["l0.ff1"], s["l0.ff_act"])
            self._gelu_raw_cls = self._gelu_raw(s["cls1"], s["cls_act"])

        def _gelu_i(self, tag, q, lut, raw):
            idx = np.clip(q, -QMAX, QMAX) + QMAX
            self._cnt(tag, raw[idx])                # LUT entry before its build-time clip
            return lut[idx]

        def infer_counted(self, x_real, x_codes_pre):
            """x_codes_pre: np.round(x1024 / s_in) BEFORE clipping (input saturation)."""
            s = self.scales
            self._cnt("input", x_codes_pre)
            x = self._lin("embed_proj", np.clip(np.round(x_real / s["input"]), -QMAX, QMAX)
                          .astype(np.int64), "proj", s["input"], s["embed"])
            q_pe = np.round(self.pe / s["embed"]).astype(np.int64)
            x = self._addi("embed", x, s["embed"], q_pe[None], s["embed"], s["embed"])
            s_in = s["embed"]
            pos = layer_positions(0)
            pfx = "encoder_layers.0."
            # attention
            qh = self._split(self._lin("l0.q", x, "l0.wq", s_in, s[pos["q"]]))
            kh = self._split(self._lin("l0.k", x, "l0.wk", s_in, s[pos["k"]]))
            vh = self._split(self._lin("l0.v", x, "l0.wv", s_in, s[pos["v"]]))
            acc = qh @ kh.transpose(0, 1, 3, 2)
            M = (s[pos["q"]] * s[pos["k"]]) / (math.sqrt(self.head_dim) * s[pos["scores"]])
            q_scores = self._rq("l0.scores", acc, M)
            self._cur_layer = 0
            q_attn = self._softmax(q_scores, None)
            s_attn = 1.0 / im.qint_max(im.ATTN_BITS)
            q_ctx = self._rq("l0.ctx", q_attn @ vh, (s_attn * s[pos["v"]]) / s[pos["ctx"]])
            attn = self._lin("l0.attn_out", self._merge(q_ctx), "l0.out", s[pos["ctx"]], s[pos["attn_out"]])
            # encoder tail
            res1 = self._addi("l0.res1", x, s_in, attn, s[pos["attn_out"]], s[pos["res1"]])
            ln1 = self._lni("l0.ln1", res1, s[pos["res1"]], self.f[pfx + "attention_norm.weight"],
                            self.f[pfx + "attention_norm.bias"], s[pos["ln1"]])
            ff1 = self._lin("l0.ff1", ln1, "l0.ff0", s[pos["ln1"]], s[pos["ff1"]])
            ff_act = self._gelu_i("l0.ff_act", ff1, self.gelu_lut[0], self._gelu_raw_l0)
            ff2 = self._lin("l0.ff2", ff_act, "l0.ff3", s[pos["ff_act"]], s[pos["ff2"]])
            res2 = self._addi("l0.res2", ln1, s[pos["ln1"]], ff2, s[pos["ff2"]], s[pos["res2"]])
            x = self._lni("l0.ln2", res2, s[pos["res2"]], self.f[pfx + "feedforward_norm.weight"],
                          self.f[pfx + "feedforward_norm.bias"], s[pos["ln2"]])
            s_in = s[pos["ln2"]]
            q_pool = self._rq("pool", np.sum(x, axis=1), s_in / (x.shape[1] * s["pool"]))
            cls_ln = self._lni("cls_ln", q_pool, s["pool"], self.f["classifier.0.weight"],
                               self.f["classifier.0.bias"], s["cls_ln"])
            c1 = self._lin("cls1", cls_ln, "cls1", s["cls_ln"], s["cls1"])
            c_act = self._gelu_i("cls_act", c1, self.cls_gelu_lut, self._gelu_raw_cls)
            return self._lin("logits", c_act, "cls4", s["cls_act"], s["logits"])

    return InstrumentedIntegerTransformer


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def _calibrate_one(job):
    name, seed, n, raw_dir, files, checkpoint = job
    import pandas as pd
    from src.data.make_dataloaders import build_X_y_from_file_index
    from src.quantize.evaluate_integer_bits import calibrate_bits
    t0 = time.perf_counter()
    df = pd.DataFrame(files)
    df["filepath"] = [str(Path(raw_dir) / p) for p in df["relpath"]]
    Xc, _, kept, dropped, _ = build_X_y_from_file_index(
        df[["filepath", "label"]], window_size=1024, stride=1024, aggregation_factor=4)
    t_load = time.perf_counter() - t0
    m = _new_model(_load_state(checkpoint))
    t1 = time.perf_counter()
    scales = {k: float(v) for k, v in calibrate_bits(m, Xc, BITS).items()}
    return {"name": name, "seed": seed, "n_files": n, "files_kept": int(kept),
            "files_dropped": int(dropped), "windows": int(len(Xc)),
            "load_seconds": t_load, "collect_seconds": time.perf_counter() - t1,
            "peak_rss_mb": peak_rss_mb(), "scales": scales, "files": files}


def cmd_calibrate(args):
    from src.config import DATA_RAW_DIR
    raw_dir = Path(args.raw_dir or DATA_RAW_DIR)
    train_df, val_df, test_df = _split(raw_dir)
    jobs, draws = [], {}
    for seed in args.seeds:
        prev = None
        for n in sorted(args.sizes):
            sub = train_df.sample(n=n, random_state=seed).reset_index(drop=True)
            rel = [_rel(p, raw_dir) for p in sub["filepath"]]
            if prev is not None:
                assert rel[:len(prev)] == prev, f"seed {seed}: {n}-file draw is not nested"
            prev = rel
            files = [{"relpath": r, "label": l} for r, l in zip(rel, sub["label"])]
            name = f"s{seed}_n{n}"
            draws[name] = files
            jobs.append((name, seed, n, str(raw_dir), files, str(args.checkpoint)))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
        results = list(ex.map(_calibrate_one, jobs))
    out = {r["name"]: r for r in results}

    canon = json.loads(EXPORT_CONFIG.read_text())
    if CANONICAL in out:
        same = out[CANONICAL]["scales"] == canon["scales"]
        out[CANONICAL]["matches_deployed_integer_config"] = bool(same)
        assert same, "seed-42/20-file recalibration does not reproduce the deployed scales"
        assert out[CANONICAL]["windows"] == canon["calibration"]["windows"]
    payload = {
        "stage": "day2_calibration_sensitivity_calibrate",
        "split": "train only (calibration); evaluation uses validation",
        "rule": "train_df.sample(n, random_state=seed); float 4:1 mean; max-abs/511",
        "canonical": CANONICAL,
        "deployed_integer_config": str(EXPORT_CONFIG.relative_to(PROJECT_ROOT)),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "val_files": [_rel(p, raw_dir) for p in val_df["filepath"]],
        "wall_seconds": time.perf_counter() - t0,
        "configs": out,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calibration_sets.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for r in results:
        print(f"[calib] {r['name']:8s} files={r['files_kept']} windows={r['windows']} "
              f"load={r['load_seconds']:.0f}s collect={r['collect_seconds']:.0f}s "
              f"rss={r['peak_rss_mb']:.0f}MB")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

_W: dict = {}


def _worker_init(checkpoint, scale_sets):
    state = _load_state(Path(checkpoint))
    _W["models"] = {k: _new_model(state, s, instrumented=True) for k, s in scale_sets.items()}
    _W["stock"] = {k: _new_model(state, s) for k, s in scale_sets.items()}
    _W["float"] = _new_model(state, next(iter(scale_sets.values())))
    _W["checked"] = False


def _eval_file(job):
    from src.data.make_dataloaders import build_X_y_from_file_index
    from src.data.token_aggregation import aggregate_round_div, aggregate_mean_float
    import pandas as pd
    path, label, batch = job
    X, y, kept, _, _ = build_X_y_from_file_index(
        pd.DataFrame([{"filepath": path, "label": label}]),
        window_size=1024, stride=1024, aggregation_factor=1)
    res = {"path": path, "n": int(len(y)), "y": y.astype(int).tolist(), "pred": {}, "sat": {},
           "seconds": {}}
    if len(y) == 0:
        return res
    X = X.astype(np.float64)
    t = time.perf_counter()
    fl = np.concatenate([_W["float"].float_logits(aggregate_mean_float(X[i:i + batch], 4, np.float64))
                         for i in range(0, len(X), batch)])
    res["pred"]["float"] = fl.argmax(1).tolist()
    res["seconds"]["float"] = time.perf_counter() - t
    for k, m in _W["models"].items():
        t = time.perf_counter()
        m.reset_counts()
        s_in = m.scales["input"]
        logits = []
        for i in range(0, len(X), batch):
            pre = np.round(X[i:i + batch] / s_in)
            q256 = aggregate_round_div(np.clip(pre, -QMAX, QMAX).astype(np.int64), 4)
            xin = q256.astype(np.float64) * s_in
            lg = m.infer_counted(xin, pre)
            if not _W["checked"]:                    # instrumented == stock, bit for bit
                assert np.array_equal(lg, _W["stock"][k].infer_int(xin)), k
            logits.append(lg)
        _W["checked"] = _W["checked"] or k == list(_W["models"])[-1]
        res["pred"][k] = np.concatenate(logits).argmax(1).tolist()
        res["sat"][k] = {p: v.tolist() for p, v in m.sat.items()}
        res["seconds"][k] = time.perf_counter() - t
    return res


def cmd_evaluate(args):
    from src.config import DATA_RAW_DIR
    raw_dir = Path(args.raw_dir or DATA_RAW_DIR)
    calib = json.loads(Path(args.scales).read_text())
    names = args.configs or list(calib["configs"])
    scale_sets = {k: calib["configs"][k]["scales"] for k in names}
    _, val_df, test_df = _split(raw_dir)
    df = {"val": val_df, "test": test_df}[args.split]
    rel = [_rel(p, raw_dir) for p in df["filepath"]]
    if args.split == "val":
        assert rel == calib["val_files"], "validation split differs from the calibration record"
    df = df.iloc[args.shard::args.num_shards]
    if args.limit_files:
        df = df.iloc[:args.limit_files]
    jobs = [(p, l, args.batch) for p, l in zip(df["filepath"], df["label"])]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                             initargs=(str(args.checkpoint), scale_sets)) as ex:
        files = list(ex.map(_eval_file, jobs))
    for f in files:
        f["path"] = _rel(f["path"], raw_dir)
    out = {"stage": "day2_calibration_sensitivity_evaluate", "split": args.split,
           "shard": args.shard, "num_shards": args.num_shards, "configs": names,
           "numeric_order": "z-score 1024 -> INT10 quantize -> aggregate_round_div 4:1 -> "
                            "IntegerTransformer(256); float ref: float mean -> float model",
           "workers": args.workers, "wall_seconds": time.perf_counter() - t0,
           "peak_rss_mb_parent": peak_rss_mb(), "files": files}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(f"[eval] {args.split} shard {args.shard}/{args.num_shards}: {len(files)} files, "
          f"{sum(f['n'] for f in files)} windows, {out['wall_seconds']:.0f}s")


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

def _metrics(y, p, k=4):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-300), 0.0)
    return {"windows": int(len(y)), "correct": int(tp.sum()), "accuracy": float(tp.sum() / len(y)),
            "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()),
            "macro_f1": float(f1.mean()),
            "per_class_recall": dict(zip(CLASS_NAMES, map(float, rec))),
            "per_class_f1": dict(zip(CLASS_NAMES, map(float, f1))),
            "confusion_matrix": cm.tolist()}


def _file_bootstrap(y, p, ref, n_per_file, reps=2000, seed=0):
    """Paired FILE-level bootstrap of (config - canonical) accuracy and macro-F1.
    Windows of one file are correlated, so files are the resampling unit."""
    rng = np.random.default_rng(seed)
    starts = np.r_[0, np.cumsum(n_per_file)[:-1]]
    nf = len(n_per_file)
    dacc, df1 = [], []
    for _ in range(reps):
        pick = rng.integers(0, nf, nf)
        idx = np.concatenate([np.arange(starts[i], starts[i] + n_per_file[i]) for i in pick])
        a, b = _metrics(y[idx], p[idx]), _metrics(y[idx], ref[idx])
        dacc.append(a["accuracy"] - b["accuracy"])
        df1.append(a["macro_f1"] - b["macro_f1"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"reps": reps, "unit": "file", "accuracy_diff_ci95": q(dacc), "macro_f1_diff_ci95": q(df1)}


def cmd_combine(args):
    calib = json.loads(Path(args.calib).read_text())
    shards = [json.loads(gzip.decompress(Path(p).read_bytes()) if str(p).endswith(".gz")
                         else Path(p).read_text()) for p in args.shards]
    files = sorted((f for s in shards for f in s["files"]), key=lambda f: f["path"])
    paths = [f["path"] for f in files]
    assert len(paths) == len(set(paths)), "duplicate files across shards"
    split = shards[0]["split"]
    if split == "val":
        assert sorted(calib["val_files"]) == paths, "shards do not cover the validation split"
    names = shards[0]["configs"]
    y = np.array([v for f in files for v in f["y"]])
    preds = {k: np.array([v for f in files for v in f["pred"][k]]) for k in ["float"] + names}
    canon = preds[CANONICAL]
    cs = calib["configs"][CANONICAL]["scales"]
    rows, scale_rows, sat_rows, summary = [], [], [], {}
    for k in ["float"] + names:
        m = _metrics(y, preds[k])
        m["agreement_with_canonical"] = float((preds[k] == canon).mean())
        m["changed_predictions_vs_canonical"] = int((preds[k] != canon).sum())
        changed = preds[k] != canon
        m["changed_fixed_vs_broken"] = {
            "canonical_wrong_now_right": int((changed & (preds[k] == y)).sum()),
            "canonical_right_now_wrong": int((changed & (canon == y)).sum())}
        m["seconds_cpu_sum"] = float(sum(f["seconds"].get(k, 0.0) for f in files))
        if k != CANONICAL:
            m["bootstrap_vs_canonical"] = _file_bootstrap(
                y, preds[k], canon, np.array([f["n"] for f in files]), reps=args.bootstrap)
        if k != "float":
            c = calib["configs"][k]
            m["calibration"] = {x: c[x] for x in ("seed", "n_files", "files_kept", "windows",
                                                  "load_seconds", "collect_seconds", "peak_rss_mb")}
            sat = np.zeros((len(POSITIONS), 5), dtype=np.int64)
            for f in files:
                sat += np.array([f["sat"][k][p] for p in POSITIONS])
            m["saturation"] = {p: {"values": int(a[0]), "preclip_over": int(a[1]),
                                   "preclip_under": int(a[2]), "rate": float((a[1] + a[2]) / max(a[0], 1)),
                                   "post_clip_at_limit": int(a[3]),
                                   "windows_with_any": int(a[4])} for p, a in zip(POSITIONS, sat)}
            for p, a in zip(POSITIONS, sat):
                sat_rows.append({"config": k, "position": p, "values": int(a[0]),
                                 "preclip_over": int(a[1]), "preclip_under": int(a[2]),
                                 "rate": (a[1] + a[2]) / max(a[0], 1), "post_clip_at_limit": int(a[3]),
                                 "windows_with_any": int(a[4]), "window_rate": a[4] / len(y)})
            for p, v in c["scales"].items():
                scale_rows.append({"config": k, "position": p, "scale": v, "canonical": cs[p],
                                   "abs_change": v - cs[p], "rel_change": (v - cs[p]) / cs[p]})
        summary[k] = m
        rows.append({"config": k, "seed": m.get("calibration", {}).get("seed", ""),
                     "n_files": m.get("calibration", {}).get("n_files", ""),
                     "calib_windows": m.get("calibration", {}).get("windows", ""),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     **{f"recall_{n}": r for n, r in m["per_class_recall"].items()},
                     "acc_diff_ci95_lo": m.get("bootstrap_vs_canonical", {}).get("accuracy_diff_ci95", ["", ""])[0],
                     "acc_diff_ci95_hi": m.get("bootstrap_vs_canonical", {}).get("accuracy_diff_ci95", ["", ""])[1],
                     "agreement_with_canonical": m["agreement_with_canonical"],
                     "changed_vs_canonical": m["changed_predictions_vs_canonical"],
                     "calib_collect_s": m.get("calibration", {}).get("collect_seconds", ""),
                     "calib_peak_rss_mb": m.get("calibration", {}).get("peak_rss_mb", ""),
                     "eval_cpu_s": m["seconds_cpu_sum"]})
    args.out.mkdir(parents=True, exist_ok=True)
    tag = split
    np.savez_compressed(args.out / f"{tag}_predictions.npz", y=y.astype(np.int8),
                        paths=np.array(paths), n_per_file=np.array([f["n"] for f in files]),
                        **{k: v.astype(np.int8) for k, v in preds.items()})
    (args.out / f"{tag}_summary.json").write_text(json.dumps({
        "stage": "day2_calibration_sensitivity", "split": split,
        "label": ("validation (model-selection split; never used for calibration)" if split == "val"
                  else "EXPLORATORY: already-used test split, not independent confirmation"),
        "canonical": CANONICAL, "files": len(files), "windows": int(len(y)),
        "shard_wall_seconds": [s["wall_seconds"] for s in shards],
        "configs": summary}, indent=1), encoding="utf-8")
    for fname, data in ((f"{tag}_metrics.csv", rows), ("scales_vs_canonical.csv", scale_rows),
                        (f"{tag}_saturation.csv", sat_rows)):
        with open(args.out / fname, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    for r in rows:
        print(f"{r['config']:8s} acc={r['accuracy']:.5f} f1={r['macro_f1']:.5f} "
              f"normal={r['recall_normal']:.4f} changed={r['changed_vs_canonical']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--raw-dir", type=Path, default=None)
        p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
        p.add_argument("--workers", type=int, default=4)
    pc = sub.choices["calibrate"]
    pc.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    pc.add_argument("--sizes", type=int, nargs="+", default=[12, 20, 40])
    pc.add_argument("--out", type=Path, required=True)
    pe = sub.choices["evaluate"]
    pe.add_argument("--scales", type=Path, required=True)
    pe.add_argument("--configs", nargs="*", default=None)
    pe.add_argument("--split", choices=["val", "test"], default="val")
    pe.add_argument("--shard", type=int, default=0)
    pe.add_argument("--num-shards", type=int, default=1)
    pe.add_argument("--limit-files", type=int, default=0)
    pe.add_argument("--batch", type=int, default=16)
    pe.add_argument("--out", type=Path, required=True)
    pm = sub.add_parser("combine")
    pm.add_argument("--calib", type=Path, required=True)
    pm.add_argument("--shards", type=Path, nargs="+", required=True)
    pm.add_argument("--out", type=Path, required=True)
    pm.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    {"calibrate": cmd_calibrate, "evaluate": cmd_evaluate, "combine": cmd_combine}[args.cmd](args)


if __name__ == "__main__":
    main()
