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

## 2. FPGA-minimal sweet-spot configs (d_model 16/24 region)

Run on Kaggle 2026-08-13 to probe below F_1layer. Results archived in
`artifacts/runs/tuning_hij_20260813/` (recovered 2026-08-19; the run outputs had
lived only in a temp folder). Only the per-session winner was test-evaluated
(G_d16 in its run, I_d32_ffn32 in the H/I/J run); H_d24 and J_d16_ffn16 have val
metrics only.

| Config | d_model | layers | heads | ffn | params | val_acc | test_acc | val_false_alarm |
|---|---|---|---|---|---|---|---|---|
| I_d32_ffn32 | 32 | 1 | 4 | 32 | 7,412 | 96.89% | 96.45% | 1.97% |
| H_d24 | 24 | 1 | 3 | 48 | 5,488 | 96.29% | — | 2.62% |
| G_d16 | 16 | 1 | 2 | 32 | 2,572 | 95.84% | 95.88% | 1.97% |
| J_d16_ffn16 | 16 | 1 | 2 | 16 | 2,044 | 93.29% | — | **10.49%** |

Sources: `artifacts/runs/tuning_hij_20260813/{G_d16,H_d24,I_d32_ffn32,J_d16_ffn16}__result.json`
and `{g_d16,hij}_final_report.json` (`test_metrics_for_best`). Test false-alarm for
the tested winners: I_d32_ffn32 4.26%, G_d16 4.84% (vs F_1layer's much lower rate).

**Decision:** the H/I/J session winner was I_d32_ffn32, but the architecture was
NOT changed — I halves the FFN for only ~2k fewer params yet drops accuracy ~1.4%
and worsens the false-alarm/safety rate, so F_1layer (and the A/F family) was kept.

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
96.9% |        I_d32_ffn32 (7.4k)
96.3% |      H_d24 (5.5k)
95.8% |    G_d16 (2.6k)
93.3% |   J_d16_ffn16 (2.0k)   <- accuracy + safety fall off
89.3% |  P_prior (1.6k)
        +------------------------------------------------> params
```

The 1.6k–9.5k gap is now filled (G/H/I/J): the curve bends sharply right at
F_1layer. Below F, accuracy falls (F 98.3% -> I 96.9% -> G 95.8% -> J 93.3%) and,
below ~2.5k params (J), the false-alarm rate explodes (10.5%). So F_1layer is the
smallest config that stays in the "safe ~98%" band.

---

## 4. Conclusion (gap now closed)

The whole size spectrum from 1.6k to 69.9k params has now been explored (11
configs). The accuracy-vs-size curve has a clear knee at **F_1layer (9,492
params, ~98%)**:
- Above F: more accuracy but rapidly more params (up to 69.9k for +1.4%).
- Below F: I/H/G give up ~1.4–2.4% accuracy for modest size savings, and J
  (2,044 params) breaks the safety margin (false alarm 10.5%).

F_1layer is therefore the selected FPGA architecture — the smallest model that
keeps ~98% accuracy and a low false-alarm rate. If a future board is too small
even for F, **G_d16 (2,572 params, 95.88% test, fault-recall 99.6%)** is the
documented fallback (its higher false-alarm ~4.8% is the cost).

Optional future work (still EXP-4): only a coarse grid was tried below F; a finer
search or light regularization might recover some of G/H's accuracy at their size.

_Last updated: 2026-08-19._
