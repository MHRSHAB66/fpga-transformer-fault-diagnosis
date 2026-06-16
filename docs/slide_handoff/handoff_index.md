# Handoff Index — Read This First

## Purpose
This ZIP contains everything needed to revise and finalize the project presentation slides.
The software/model phase (Stages 0–5) is now **complete** with a final full-dataset training run.
All old 6-epoch results in the slides must be replaced with the results from this final run.

## WARNING — Do not use old results
Slides 7–10 currently show results from an older 6-epoch training run.
**Use only the files in this ZIP.** The authoritative numbers are in `metrics.json`.

---

## ZIP contents

| File / Folder | What it is |
|---|---|
| `presentation_progress.pptx` | **The current slide deck — edit this file** |
| `handoff_index.md` | This file |
| `slide_update_brief.md` | What to update and why — read second |
| `slide_by_slide_plan.md` | Exact per-slide edit instructions |
| `final_results_summary.md` | Clean human-readable summary of the final run |
| `figures_manifest.md` | Every figure in the ZIP with caption/slide guidance |
| `metrics.json` | Full metrics from the final training run |
| `transformer_experiment_config.json` | Model + training + preprocessing config |
| `history.json` | Per-epoch train/val loss and accuracy |
| `run_summary.json` | Artifact paths |
| `normalization_constants.json` | Normalization info |
| `final_runner_report.json` | Top-level run report (passed, mode, GPU) |
| `transformer_baseline_model_config.json` | Model architecture snapshot |
| `plots/` | All 6 training/evaluation plots (PNG) |
| `diagrams/` | 3 clean pipeline/architecture/stage diagrams (PNG) |

---

## Key numbers to use in slides

| Metric | Value |
|---|---|
| Test accuracy | **98.22%** |
| Val accuracy (best epoch) | **98.68%** |
| Best epoch | **10** (early stopped at epoch 16) |
| Best val loss | **0.0353** |
| Test macro F2 | **0.9780** |
| Test weighted F2 | **0.9822** |
| Fault recall | **99.75%** |
| False alarm rate | **2.34%** |
| Peak GPU memory | **0.28 GiB** |
| Checkpoint size | **380 KB** |

Checkpoint path: `artifacts/runs/20260615_223905_fpga_fault_final_t4/checkpoints/transformer_best.pt`

---

## What ChatGPT should do
1. Read `slide_update_brief.md` to understand the full context.
2. Open `slide_by_slide_plan.md` for exact per-slide instructions.
3. Open `presentation_progress.pptx` and apply the changes.
4. Replace all old figures with the PNGs from `plots/` and `diagrams/`.
5. Do **not** claim FPGA implementation is complete — only the software phase is done.
6. Save the revised deck as `FPGA_Transformer_Condition_Monitoring_Final_Software.pptx`.
