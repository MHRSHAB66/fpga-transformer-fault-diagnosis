"""Independent audit of the Day-2 calibration-sensitivity evidence (read-only).

Re-derives everything from the four committed Kaggle shard outputs, with code
that does not reuse calibration_sensitivity.combine's metric functions:

  1. coverage   shard files == val_df.iloc[s::4]; union == validation split
                exactly once; window counts; every config predicted every window
  2. calib      every calibration file is a TRAIN file (none in val/test);
                nesting; s42_n20 == deployed integer_config.json;
                s42_n12 == historical numpy_calib12_scales.json
  3. metrics    accuracy / macro-F1 / per-class recall / confusion / changed
                predictions recomputed from per-file confusion matrices and
                compared with val_summary.json; val_predictions.npz checked
  4. saturation counters re-summed; denominators == windows x elements/window;
                pre-clip beyond <= post-clip-at-limit; windows_with_any <= windows
  5. bootstrap  combine's unstratified paired file-level CI reproduced exactly
                (same RNG stream), plus a class-STRATIFIED paired bootstrap
                (files resampled within their class) incl. normal recall
  6. local      --reproduce-files K: recompute predictions of K validation files
                for all 12 scale sets with the STOCK IntegerTransformer.infer_int
                (normalize -> INT10 -> aggregate_round_div -> 256 tokens) and
                compare with Kaggle; also LayerNorm integer-variance statistics

    python -m tools.audit_day2_calibration --reproduce-files 2
writes artifacts/experiments/day2_calibration_sensitivity_20261120/audit/audit_recompute.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts" / "experiments" / "day2_calibration_sensitivity_20261120"
DEPLOYED = ROOT / "artifacts" / "quantized" / "f_seq256_int10" / "integer_config.json"
CALIB12 = ROOT / "artifacts" / "results" / "board_vs_reference_20261022" / "numpy_reference" / "numpy_calib12_scales.json"
CANON = "s42_n20"
ELEMS = {"input": 1024 * 8, "embed_proj": 256 * 32, "embed": 256 * 32, "l0.q": 256 * 32,
         "l0.k": 256 * 32, "l0.v": 256 * 32, "l0.scores": 4 * 256 * 256, "l0.ctx": 256 * 32,
         "l0.attn_out": 256 * 32, "l0.res1": 256 * 32, "l0.ln1": 256 * 32, "l0.ff1": 256 * 64,
         "l0.ff_act": 256 * 64, "l0.ff2": 256 * 32, "l0.res2": 256 * 32, "l0.ln2": 256 * 32,
         "pool": 32, "cls_ln": 32, "cls1": 16, "cls_act": 16, "logits": 4}


def cm_metrics(cm):
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), 0.0)
    return tp.sum() / cm.sum(), f1.mean(), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce-files", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.perf_counter()
    out, fails = {}, []

    def check(name, ok, detail=None):
        out.setdefault("checks", {})[name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
        if not ok:
            fails.append(name)

    calib = json.loads((EXP / "calibration_sets.json").read_text())
    shards = [json.loads(gzip.decompress((EXP / "kaggle_val" / f"val_shard{s}.json.gz").read_bytes()))
              for s in range(4)]
    summ = json.loads((EXP / "val_summary.json").read_text())
    cfgs = shards[0]["configs"]
    allc = ["float"] + cfgs

    # ---- 1. coverage ------------------------------------------------------
    from src.config import DATA_RAW_DIR
    from src.quantize.calibration_sensitivity import _split, _rel
    tr, va, te = _split(Path(DATA_RAW_DIR))
    rel = lambda df: [_rel(p, DATA_RAW_DIR) for p in df["filepath"]]
    val_rel, tr_rel, te_rel = rel(va), set(rel(tr)), set(rel(te))
    check("val_split_equals_calibration_record", val_rel == calib["val_files"])
    for s, sh in enumerate(shards):
        check(f"shard{s}_files_are_val_iloc_s_mod_4",
              [f["path"] for f in sh["files"]] == val_rel[s::4],
              {"files": len(sh["files"]), "windows": sum(f["n"] for f in sh["files"])})
        check(f"shard{s}_same_config_list", sh["configs"] == cfgs and sh["split"] == "val")
    # same order as combine (sorted by path): needed to replay its bootstrap RNG stream
    files = sorted((f for sh in shards for f in sh["files"]), key=lambda f: f["path"])
    paths = [f["path"] for f in files]
    check("union_covers_val_exactly_once", sorted(paths) == sorted(val_rel) and len(set(paths)) == len(paths),
          {"files": len(paths)})
    nwin = sum(f["n"] for f in files)
    check("window_count_47580", nwin == 47580, nwin)
    check("every_config_predicts_every_window",
          all(len(f["y"]) == f["n"] and all(len(f["pred"][k]) == f["n"] for k in allc) for f in files))
    check("saturation_recorded_for_all_int10_configs", all(set(f["sat"]) == set(cfgs) for f in files))

    # ---- 2. calibration provenance ----------------------------------------
    C = calib["configs"]
    allcal = {fl["relpath"] for k in C for fl in C[k]["files"]}
    check("calibration_files_all_train", allcal <= tr_rel, {"distinct_files": len(allcal)})
    check("calibration_files_disjoint_val_test", not (allcal & (set(val_rel) | te_rel)))
    for seed in (42, 1, 2):
        lists = [[fl["relpath"] for fl in C[f"s{seed}_n{n}"]["files"]] for n in (12, 20, 40, 80)]
        check(f"seed{seed}_nested", all(lists[i + 1][:len(lists[i])] == lists[i] for i in range(3)))
    check("s42_n20_equals_deployed_scales", C[CANON]["scales"] == json.loads(DEPLOYED.read_text())["scales"])
    check("s42_n20_window_count_equals_deployed",
          C[CANON]["windows"] == json.loads(DEPLOYED.read_text())["calibration"]["windows"], C[CANON]["windows"])
    if CALIB12.exists():
        check("s42_n12_equals_historical_calib12", C["s42_n12"]["scales"] == json.loads(CALIB12.read_text()))

    # ---- 3. metrics from per-file confusion matrices ------------------------
    y = np.concatenate([f["y"] for f in files])
    P = {k: np.concatenate([f["pred"][k] for f in files]) for k in allc}
    fcm = {k: np.stack([np.bincount(4 * np.array(f["y"]) + np.array(f["pred"][k]), minlength=16).reshape(4, 4)
                        for f in files]) for k in allc}
    recomputed = {}
    worst = 0.0
    for k in allc:
        cm = fcm[k].sum(0)
        acc, f1, rec = cm_metrics(cm)
        s = summ["configs"][k]
        ch = int((P[k] != P[CANON]).sum())
        recomputed[k] = {"accuracy_pct": 100 * acc, "macro_f1_pct": 100 * f1,
                         "recall": dict(zip(["normal", "imbalance", "misalignment", "bearing_fault"], rec.tolist())),
                         "correct": int(np.trace(cm)), "changed_vs_canonical": ch}
        worst = max(worst, abs(acc - s["accuracy"]), abs(f1 - s["macro_f1"]),
                    *[abs(a - b) for a, b in zip(rec, s["per_class_recall"].values())])
        check(f"{k}_confusion_and_changes_match_summary",
              cm.tolist() == s["confusion_matrix"] and ch == s["changed_predictions_vs_canonical"]
              and int(np.trace(cm)) == s["correct"])
    check("float_metrics_match_summary_1e-12", worst < 1e-12, worst)
    z = np.load(EXP / "val_predictions.npz")
    order = np.argsort(paths, kind="stable")
    starts = np.r_[0, np.cumsum([f["n"] for f in files])]
    idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in order])
    check("npz_matches_shards", list(z["paths"]) == sorted(paths) and np.array_equal(z["y"], y[idx])
          and all(np.array_equal(z[k], P[k][idx]) for k in allc))
    out["recomputed_metrics"] = recomputed

    # ---- 4. saturation --------------------------------------------------------
    satchk = {}
    for k in cfgs:
        tot = {p: np.sum([f["sat"][k][p] for f in files], axis=0) for p in ELEMS}
        for p, a in tot.items():
            s = summ["configs"][k]["saturation"][p]
            ok = (a[0] == nwin * ELEMS[p] and a[0] == s["values"] and a[1] == s["preclip_over"]
                  and a[2] == s["preclip_under"] and a[3] == s["post_clip_at_limit"]
                  and a[4] == s["windows_with_any"] and a[1] + a[2] <= a[3] and a[4] <= nwin)
            if not ok:
                satchk[f"{k}:{p}"] = a.tolist()
    check("saturation_counters_denominators_and_bounds", not satchk, satchk or None)
    check("saturation_positions_complete", set(summ["configs"][CANON]["saturation"]) == set(ELEMS))
    out["pool_saturation_ppm"] = {k: 1e6 * summ["configs"][k]["saturation"]["pool"]["rate"] for k in cfgs}

    # ---- 5. bootstrap ------------------------------------------------------
    labels = np.array([f["y"][0] for f in files])
    nf = len(files)
    boot = {}
    for k in allc:
        if k == CANON:
            continue
        # (a) exactly combine's RNG stream (default_rng(0), one integers(0,nf,nf) per rep)
        rng = np.random.default_rng(0)
        da = []
        df = []
        for _ in range(args.reps):
            pick = rng.integers(0, nf, nf)
            a1, f11, _ = cm_metrics(fcm[k][pick].sum(0))
            a0, f10, _ = cm_metrics(fcm[CANON][pick].sum(0))
            da.append(a1 - a0)
            df.append(f11 - f10)
        # (b) class-stratified paired bootstrap
        rng = np.random.default_rng(1)
        groups = [np.flatnonzero(labels == c) for c in range(4)]
        sa, sf, sn = [], [], []
        for _ in range(args.reps):
            pick = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
            a1, f11, r1 = cm_metrics(fcm[k][pick].sum(0))
            a0, f10, r0 = cm_metrics(fcm[CANON][pick].sum(0))
            sa.append(a1 - a0)
            sf.append(f11 - f10)
            sn.append(r1[0] - r0[0])
        q = lambda v: [round(100 * float(np.percentile(v, 2.5)), 4), round(100 * float(np.percentile(v, 97.5)), 4)]
        rep_ci = summ["configs"][k]["bootstrap_vs_canonical"]
        same = (np.allclose(np.percentile(da, [2.5, 97.5]), rep_ci["accuracy_diff_ci95"], atol=1e-12)
                and np.allclose(np.percentile(df, [2.5, 97.5]), rep_ci["macro_f1_diff_ci95"], atol=1e-12))
        boot[k] = {"unstratified_acc_pp": q(da), "unstratified_f1_pp": q(df), "reproduces_combine": bool(same),
                   "stratified_acc_pp": q(sa), "stratified_f1_pp": q(sf), "stratified_normal_recall_pp": q(sn)}
        check(f"{k}_bootstrap_reproduced", same)
    out["bootstrap_vs_canonical_pp"] = boot
    out["files_per_class"] = {c: int((labels == i).sum()) for i, c in
                              enumerate(["normal", "imbalance", "misalignment", "bearing_fault"])}

    # ---- 6. local re-execution with the STOCK reference path -----------------
    if args.reproduce_files:
        out["local_reproduction"] = reproduce(files, calib, val_rel, va, args.reproduce_files, check)

    out["failed_checks"] = fails
    out["n_checks"] = len(out["checks"])
    out["seconds"] = time.perf_counter() - t0
    (EXP / "audit").mkdir(exist_ok=True)
    (EXP / "audit" / "audit_recompute.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"checks: {len(out['checks']) - len(fails)}/{len(out['checks'])} ok; failed: {fails}")


def reproduce(files, calib, val_rel, va, n_files, check):
    import torch
    from src.data.make_dataloaders import build_X_y_from_file_index
    from src.data.token_aggregation import aggregate_round_div
    from src.quantize.calibration_sensitivity import _new_model, DEFAULT_CHECKPOINT
    import src.quantize.integer_model as im
    ck = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    state = {k: v.cpu().numpy() for k, v in ck["model_state_dict"].items()}
    # one normal file (rarest class) + the first file of shard 1
    by_path = {f["path"]: f for f in files}
    normal = next(p for p in val_rel if p.startswith("normal/"))
    picks = [normal, val_rel[1]][:n_files]
    res = {"files": picks, "configs": {}}
    var_stats = {}

    class VarTap(im.IntegerTransformer):
        def _layernorm(self, q, s_in, gamma, beta, s_out):
            D = q.shape[-1]
            mean = np.round(np.sum(q, axis=-1, keepdims=True).astype(np.float64) / D).astype(np.int64)
            var = np.round(np.sum((q - mean) ** 2, axis=-1).astype(np.float64) / D).astype(np.int64)
            key = "cls_ln" if q.ndim == 2 else ("ln1_or_ln2")
            a = var_stats.setdefault(key, np.zeros(4, dtype=np.int64))
            a += [var.size, int((var == 0).sum()), int((var < 32).sum()), 0]
            return super()._layernorm(q, s_in, gamma, beta, s_out)

    all_ok = True
    for k, cfg in calib["configs"].items():
        m = _new_model(state, cfg["scales"])
        if k == CANON:
            m.__class__ = VarTap
        s_in = cfg["scales"]["input"]
        agree = 0
        tot = 0
        for p in picks:
            row = va.iloc[val_rel.index(p)]
            X, yy, _, _, _ = build_X_y_from_file_index(row.to_frame().T, window_size=1024,
                                                       stride=1024, aggregation_factor=1)
            q1024 = np.clip(np.round(X.astype(np.float64) / s_in), -511, 511).astype(np.int64)
            xin = aggregate_round_div(q1024, 4).astype(np.float64) * s_in
            pred = np.concatenate([m.infer_int(xin[i:i + 16]) for i in range(0, len(xin), 16)]).argmax(1)
            kag = np.array(by_path[p]["pred"][k])
            agree += int((pred == kag).sum())
            tot += len(pred)
        res["configs"][k] = {"windows": tot, "identical_to_kaggle": agree}
        all_ok &= agree == tot
    res["layernorm_integer_variance_canonical"] = {
        k: {"rows": int(v[0]), "var_eq_0": int(v[1]), "var_lt_32": int(v[2])} for k, v in var_stats.items()}
    check("local_stock_path_reproduces_kaggle_predictions", all_ok,
          {k: v["identical_to_kaggle"] for k, v in res["configs"].items()})
    return res


if __name__ == "__main__":
    main()
