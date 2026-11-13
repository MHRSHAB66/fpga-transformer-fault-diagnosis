"""A02 summary: metrics, paired file-level bootstrap against INT10, attention
diagnostics and the frozen gates of artifacts/research/F00_20261113/preregistered_protocol.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = ["normal", "imbalance", "misalignment", "bearing_fault"]
BOOT, SEED = 2000, 20261113
SEQ, HEADS = 256, 4


def _metrics(y, p):
    rec = [float(((p == c) & (y == c)).sum() / max((y == c).sum(), 1)) for c in range(4)]
    f1 = []
    for c in range(4):
        tp = ((p == c) & (y == c)).sum()
        prec = tp / max((p == c).sum(), 1)
        r = tp / max((y == c).sum(), 1)
        f1.append(0.0 if prec + r == 0 else 2 * prec * r / (prec + r))
    return {"accuracy": float((p == y).mean()), "macro_f1": float(np.mean(f1)), "recall": rec}


def _paired_boot(y, p_base, p_var, files):
    """File-level paired bootstrap of (variant - baseline) accuracy and macro-F1 (pp)."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(files)
    members = {f: np.flatnonzero(files == f) for f in uniq}
    da, df = [], []
    for _ in range(BOOT):
        idx = np.concatenate([members[f] for f in rng.choice(uniq, size=len(uniq), replace=True)])
        mb, mv = _metrics(y[idx], p_base[idx]), _metrics(y[idx], p_var[idx])
        da.append(100 * (mv["accuracy"] - mb["accuracy"]))
        df.append(100 * (mv["macro_f1"] - mb["macro_f1"]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {"delta_accuracy_pp_ci95": ci(da), "delta_macro_f1_pp_ci95": ci(df)}


def summarize(out: Path):
    from tools.research.precision_ablation import split_rows
    _, val_rows, y, files = split_rows()
    variants = {}
    for d in sorted(p for p in out.iterdir() if (p / "val_outputs.npz").exists()):
        z = np.load(d / "val_outputs.npz")
        assert (z["rows"] == val_rows).all()
        variants[d.name] = (json.loads((d / "variant.json").read_text()), z)
    if "int10" not in variants:
        raise SystemExit("int10 baseline missing")
    base_pred = variants["int10"][1]["logits"].argmax(1)
    base = _metrics(y, base_pred)
    rows_per_window = HEADS * SEQ
    table = {}
    for name, (spec, z) in variants.items():
        pred = z["logits"].argmax(1)
        m = _metrics(y, pred)
        zr = z["zero_rows"].astype(np.int64)                       # (N, H)
        win_zero_rate = zr.sum(1) / rows_per_window
        changed = pred != base_pred
        per_file = {}
        for f in np.unique(files):
            sel = files == f
            per_file[int(f)] = float(win_zero_rate[sel].mean())
        top_files = sorted(per_file.items(), key=lambda kv: -kv[1])[:5]
        row = {
            "spec": spec, **m,
            "delta_vs_int10_pp": {"accuracy": round(100 * (m["accuracy"] - base["accuracy"]), 3),
                                  "macro_f1": round(100 * (m["macro_f1"] - base["macro_f1"]), 3),
                                  "recall": [round(100 * (a - b), 3) for a, b in zip(m["recall"], base["recall"])]},
            "prediction_changes_vs_int10": int(changed.sum()),
            "attention": {
                "zero_row_rate": float(zr.sum() / (len(zr) * rows_per_window)),
                "zero_row_rate_per_head": [float(v) for v in zr.sum(0) / (len(zr) * SEQ)],
                "windows_with_any_zero_row": int((zr.sum(1) > 0).sum()),
                "zero_row_rate_changed_windows": float(win_zero_rate[changed].mean()) if changed.any() else None,
                "zero_row_rate_unchanged_windows": float(win_zero_rate[~changed].mean()),
                "files_with_highest_zero_row_rate": [[f, round(v, 4)] for f, v in top_files],
                "mean_row_mass": float(z["row_mass"].mean()), "min_row_mass": float(z["row_mass"].min()),
                "mean_nonzero_per_row": float(z["nonzero_per_row"].mean()),
                "mean_prob_abs_err_subset": float(np.nanmean(z["prob_abs_err"])),
                "mean_ctx_rel_err_subset": float(np.nanmean(z["ctx_rel_err"])),
                "error_subset_windows": int(np.isfinite(z["ctx_rel_err"]).sum()),
            },
        }
        if name != "int10":
            row["paired_file_bootstrap"] = _paired_boot(y, base_pred, pred, files)
        table[name] = row

    gates = {}
    if all(k in table for k in ("p8s", "p8u", "p9s")):
        p8 = table["p8s"]
        loss = -p8["delta_vs_int10_pp"]["accuracy"]
        rec = {k: (loss - (-table[k]["delta_vs_int10_pp"]["accuracy"])) / loss if loss > 0 else None for k in ("p8u", "p9s")}
        a = p8["attention"]
        gates["A02_flat_row_mechanism"] = {
            "zero_row_rate_p8s": a["zero_row_rate"], "criterion_1_rate_ge_1pct": a["zero_row_rate"] >= 0.01,
            "criterion_2_changed_gt_unchanged": (a["zero_row_rate_changed_windows"] or 0) > a["zero_row_rate_unchanged_windows"],
            "p8s_accuracy_loss_pp": loss, "recovered_fraction": rec,
            "criterion_3_recovery_ge_50pct": any(v is not None and v >= 0.5 for v in rec.values()),
        }
        g = gates["A02_flat_row_mechanism"]
        g["supported"] = bool(g["criterion_1_rate_ge_1pct"] and g["criterion_2_changed_gt_unchanged"] and g["criterion_3_recovery_ge_50pct"])
    gates["A02_precision_finalist_accuracy_gate"] = {
        n: bool(r["delta_vs_int10_pp"]["macro_f1"] >= -0.3 and min(r["delta_vs_int10_pp"]["recall"]) >= -1.0)
        for n, r in table.items() if n != "int10"}
    result = {"id": "A02", "data": "canonical validation split, %d windows, %d files" % (len(y), len(np.unique(files))),
              "baseline": "int10 (research INT10 = canonical model, selfcheck.json)", "variants": table, "gates": gates}
    (out / "summary.json").write_text(json.dumps(result, indent=1))
    print("%-10s %7s %7s %7s %7s  %8s %8s %7s" % ("variant", "acc", "dF1pp", "minRec", "chg", "zeroRow", "ctxErr", "gate"))
    for n, r in table.items():
        print("%-10s %7.4f %7.3f %7.3f %7d  %8.5f %8.4f %7s" % (
            n, r["accuracy"], r["delta_vs_int10_pp"]["macro_f1"], min(r["delta_vs_int10_pp"]["recall"]),
            r["prediction_changes_vs_int10"], r["attention"]["zero_row_rate"], r["attention"]["mean_ctx_rel_err_subset"],
            gates["A02_precision_finalist_accuracy_gate"].get(n, "-")))
    print(json.dumps(gates.get("A02_flat_row_mechanism"), indent=1))
