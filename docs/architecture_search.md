# Architecture Search — Transformer size vs accuracy

Consolidated record of every model configuration explored for the FPGA fault
classifier, why **F_1layer** was selected, and which configs still need a
documented run. Sources are cited so numbers are traceable.

Fixed settings for all runs: window 1024 / stride 1024, per-window z-score,
batch 64, max 12 epochs, patience 3, seed 42, Adam lr 1e-3, 8 input channels,
4 classes (`kaggle/tuning/output/..._tuning_final_report.json::fixed_settings`).

---

## 1. Configs with SAVED results (traceable artifacts)

Ranked by validation accuracy. Params are trainable-parameter counts.

| Config | d_model | layers | heads | ffn | params | val_acc | source |
|---|---|---|---|---|---|---|---|
| D_d64_slow | 64 | 2 | — | 128 | 69,860 | **99.66%** | results_merged.json |
| C_d48 | 48 | 2 | — | 96 | 39,724 | **99.27%** | results_merged.json |
| E_3layers | 32 | 3 | 4 | 64 | 26,580 | 98.73% | results_merged.json |
| **F_1layer** ⭐ | 32 | 1 | 4 | 64 | **9,492** | 98.28% | tuning result + full eval |
| A_baseline | 32 | 2 | 4 | 64 | 18,036 | 98.19% | tuning result |
| B_wide_ffn | 32 | 2 | 4 | 128 | 26,356 | 97.46% | tuning result |
| P_prior_like | 8 | 2 | — | 8 | 1,582 | 89.26% | results_merged.json |

Sources: `artifacts/runs/tuning_skipped_kaggle_20260801/results_merged.json`
(`all_completed_val_ranked`), `kaggle/tuning/output/tuning_20260616_051127__*__result.json`.

**F_1layer** additionally has full test-set metrics (the others have val only):
- float full-test **97.55%**, macro P/R/F1 0.928/0.980/0.951
  (`artifacts/results/f_1layer_test_metrics.json`).
- int10 full-test **97.11%** (`kaggle/int10_shard/int10_fulltest_combined.json`);
  int10 subset macro P/R/F1 0.973 (`artifacts/results/f_1layer_int10_test_metrics.json`).
- A_baseline (2-layer) is the "official software baseline": full-test **98.22%**.

---

## 2. Configs DEFINED but with NO saved result

Defined in `kaggle/tuning/fpga-fault-tuning.py` for a "sweet-spot" search below
d_model=32 (the FPGA-minimal region), but **no result JSON or log exists in the
repo**. The only trace is a code comment citing G_d16 ≈ 95.9% val — apparently
from an exploratory Kaggle run whose output was never saved.

| Config | d_model | layers | heads | ffn | ~params | accuracy | status |
|---|---|---|---|---|---|---|---|
| G_d16 | 16 | 1 | 2 | 32 | ~2,500 | ~95.9% | **comment only, unsaved** |
| H_d24 | 24 | 1 | 3 | 48 | ~5,500 | (estimate) | **not run / unsaved** |
| I_d32_ffn32 | 32 | 1 | 4 | 32 | ~7,400 | unknown | **not run / unsaved** |
| J_d16_ffn16 | 16 | 1 | 2 | 16 | ~2,000 | unknown | **not run / unsaved** |

These are exactly the FPGA-minimal candidates. G_d16 (~2,500 params, ~95.9%) is
~3.7x smaller than F_1layer for ~2.4% accuracy cost — a real option if the board
is too small for F. **To cite any of these in the thesis, re-run them on Kaggle
and save the artifacts** (see §4).

---

## 3. Findings

- **Accuracy increases monotonically with capacity** across the saved configs:
  d=8 (89%) ≪ d=32 (98%) < d=32/3L (98.7%) < d=48 (99.3%) < d=64 (99.7%).
- **Widening FFN alone does not help**: B_wide_ffn (d=32, ffn=128) is *worse*
  (97.5%) than the same model with ffn=64. Capacity should come from d_model or
  depth, not FFN width.
- **F_1layer was chosen for SIZE, not accuracy.** D_d64 is the most accurate
  (99.66%) but 7.4x larger (69,860 params) and does not fit the EBAZ4205 /
  Zynq-7010. F_1layer is the smallest config that still reaches ~98%, so it sits
  at the knee of the accuracy-vs-size Pareto curve. The ~1.4% accuracy given up
  (99.66 → 98.28 val) buys a 7.4x parameter reduction.

Accuracy vs size (saved configs):

```
val_acc
99.7% |                                              D_d64 (69.9k)
99.3% |                                    C_d48 (39.7k)
98.7% |                          E_3layers (26.6k)
98.3% |            F_1layer (9.5k)  A_baseline (18k)
97.5% |                          B_wide_ffn (26.4k)   <- wider FFN, worse
        ...
89.3% |  P_prior (1.6k)
        +------------------------------------------------> params
        (G_d16 ~2.5k @ ~95.9% and H_d24 ~5.5k would fill the 1.6k–9.5k gap
         once re-run)
```

---

## 4. Open gap / next run

The d_model=16 and 24 region (G_d16, H_d24, J_d16_ffn16, I_d32_ffn32) is the most
interesting for shrinking FPGA resources but has **no documented results**. This
is EXP-4 in `docs/future_experiments.md`.

To close it: re-run those four configs on Kaggle and save the result JSONs.
`kaggle/tuning/fpga-fault-tuning.py` already defines them; select them via the
`TUNING_RUN_ONLY` env var (comma-separated) or by editing `RUN_ONLY_DEFAULT`,
e.g. `RUN_ONLY_DEFAULT = ["G_d16", "H_d24", "I_d32_ffn32", "J_d16_ffn16"]`. Then
add a full test-set evaluation for the best small config, and append its row to §1.

_Last updated: 2026-08-19._
