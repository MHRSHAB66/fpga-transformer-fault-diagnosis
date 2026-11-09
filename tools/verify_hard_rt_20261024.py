"""Independent check of the 2026-10-24 hard-RT runs from their raw npz (no board).

For each run: recompute response = done - release, deadline misses after warm-up
(criterion file), class vs recorded 2026-10-08 board predictions, physical logits
vs the Python INT10 reference (export scales), and calibrated CONF recomputed
from the physical logits with the frozen temperature.
Usage: python -m tools.verify_hard_rt_20261024
       python -m tools.verify_hard_rt_20261024 --base <runs dir> --out-name verification.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from host.confidence import confidence_from_logits, display_confidence, load_calibration

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/results/hard_rt_20261024"
REC = np.loadtxt(ROOT / "artifacts/results/fpga_fulltest_20261008_141335/fpga_pred.txt", dtype=np.int64)
REF = np.loadtxt(ROOT / "artifacts/results/board_vs_reference_20261022/numpy_reference/numpy_export_logits.txt",
                 dtype=np.int64)[:, 1:]
P_MS, WARMUP = 20.48, 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--out-name", default="verification_v2.json")
    args = ap.parse_args()
    base = args.base if args.base.is_absolute() else ROOT / args.base
    cal = load_calibration()
    out = {"criterion": "docs/reports/hard_rt_acceptance_criterion_20261024.md",
           "calibration_T": cal["T"] if cal else None, "runs": {}}
    for d in sorted(p for p in base.iterdir() if (p / "remote/out/rt_raw.npz").exists()):
        z = np.load(d / "remote/out/rt_raw.npz")
        resp = (z["done"] - z["release"]) * 1e3
        lag = (z["start"] - z["release"]) * 1e3
        dma = (z["done"] - z["start"]) * 1e3
        acc = resp[WARMUP:]
        wi = z["window_index"]
        cls, lg = z["cls"].astype(np.int64), z["logits"].astype(np.int64)
        uniq = sorted(set(wi.tolist()))
        conf = [display_confidence(lg[k], int(cls[k]))[0] for k in range(0, len(cls), 97)]
        conf_ref = [confidence_from_logits(REF[wi[k]], int(REF[wi[k]].argmax()), temperature=cal["T"])
                    for k in range(0, len(cls), 97)]
        run = json.loads((d / "run.json").read_text())
        board_summary = json.loads((d / "remote/out/summary.json").read_text())
        expected_acc = int(board_summary["n_total"]) - WARMUP
        dur_s = float(z["done"][-1] - z["release"][0])
        r = {"executed": int(len(resp)), "acceptance_windows": int(len(acc)),
             "deadline_misses": int((acc > P_MS).sum()),
             "response_ms": {"p50": round(float(np.percentile(acc, 50)), 3), "p95": round(float(np.percentile(acc, 95)), 3),
                             "p99": round(float(np.percentile(acc, 99)), 3), "p99.9": round(float(np.percentile(acc, 99.9)), 3),
                             "p99.99": round(float(np.percentile(acc, 99.99)), 3),
                             "max": round(float(acc.max()), 3)},
             "start_lag_ms_max": round(float(lag[WARMUP:].max()), 3),
             "dma_fpga_ms": {"p50": round(float(np.percentile(dma[WARMUP:], 50)), 3), "max": round(float(dma[WARMUP:].max()), 3)},
             "release_period_ms_mean": round(float(np.diff(z["release"]).mean() * 1e3), 6),
             "duration_s": round(dur_s, 1),
             "throughput_windows_per_s": round(float(len(resp) / dur_s), 4),
             "miss_indices": [int(i) for i in np.where(acc > P_MS)[0] + WARMUP][:50],
             "windows_unique": len(uniq), "window_range": [uniq[0], uniq[-1]],
             "class_eq_recorded": int((cls == REC[wi]).sum()),
             "logits_eq_reference_all4": int((lg == REF[wi]).all(1).sum()),
             "calibrated_conf_checked": len(conf),
             "calibrated_conf_max_abs_diff_vs_reference": float(np.max(np.abs(np.array(conf) - np.array(conf_ref)))),
             "guard_rc": run.get("guard_rc"), "workload_rc": run.get("workload_rc"),
             "post_run_process_check": run.get("post_run_process_check")}
        r["hard_rt_pass"] = (r["deadline_misses"] == 0 and r["class_eq_recorded"] == r["executed"]
                             and r["logits_eq_reference_all4"] == r["executed"]
                             and r["guard_rc"] == "0" and r["workload_rc"] == "0"
                             and r["acceptance_windows"] == expected_acc)
        out["runs"][d.name] = r
        print(d.name, json.dumps({k: r[k] for k in ("acceptance_windows", "deadline_misses", "response_ms",
                                                    "throughput_windows_per_s", "class_eq_recorded",
                                                    "logits_eq_reference_all4", "calibrated_conf_max_abs_diff_vs_reference",
                                                    "hard_rt_pass")}))
    # verification.json (first version, R0-R2) is kept unchanged; v2 adds p99.99,
    # duration/throughput and longer runs.
    (base / args.out_name).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
