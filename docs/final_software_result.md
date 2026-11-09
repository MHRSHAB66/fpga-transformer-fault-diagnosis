# Final Software / Model-Phase Result

**Run ID:** `20260615_223905_fpga_fault_final_t4`  
**Date:** 2026-06-15  
**Stages complete:** 0 – 5

---

## Setup

| Item | Value |
|---|---|
| Dataset | MAFAULDA (full, 1,951 CSV files, 50 kHz, 8 channels) |
| Split | 1,365 train / 195 val / 391 test **files** (file-level, no window leakage) |
| Windows | 333,060 train / 47,580 val / 95,404 test (1024 samples × 8 channels, stride=1024) |
| Normalization | per-window Z-score (μ=0, σ=1 per window per channel) |
| Preprocessing flags | all OFF — no lowpass, Wiener, SMOTE, noise injection, resampling |
| Loss | Weighted cross-entropy (22× class imbalance ratio) |
| Batch size | 64 |
| Learning rate | 1e-3 (Adam) |
| Max epochs | 30, early stopping patience = 6, monitor = val_loss |
| GPU | Tesla T4 (Kaggle) |
| Seed | 42 |

## Model

| Parameter | Value |
|---|---|
| Input | [batch, 1024, 8] |
| Architecture | Linear(8→32) → Sinusoidal PE → 2× TransformerEncoder → Mean Pool → Classifier(32→16→4) |
| d_model | 32 |
| nhead | 4 |
| num_layers | 2 |
| dim_feedforward | 64 |
| dropout | 0.1 |
| Checkpoint size | 381 KB |
| Peak GPU memory | 0.28 GiB |

## Training Outcome

| Metric | Value |
|---|---|
| Best epoch | 10 |
| Stopped epoch | 16 (early stopped) |
| Best val loss | 0.0353 |
| Val accuracy at best | 98.68% |

## Test Set Results

| Metric | Value |
|---|---|
| Test accuracy | **98.22%** |
| Test macro F2 | **0.9780** |
| Test weighted F2 | **0.9822** |
| Fault recall | **99.75%** |
| False alarm rate | **2.34%** |

### Per-class test accuracy

| Class | Accuracy | F1 | F2 |
|---|---|---|---|
| Normal | 97.66% | 0.942 | 0.963 |
| Imbalance | 98.84% | 0.973 | 0.982 |
| Misalignment | 98.76% | 0.983 | 0.986 |
| Bearing Fault | 97.80% | 0.987 | 0.981 |

## Artifact Paths

> Local copy (2026-10-22 note): the paths below are the Kaggle run layout; in this
> repository the same run is stored under `artifacts/runs/2layer_baseline/`
> (`results/metrics.json` records `run_dir = …/20260615_223905_fpga_fault_final_t4`).
> This 2-layer model is the software baseline and is **not** the deployed FPGA model;
> see `docs/reports/consolidated_results_20261022.md`.

| Artifact | Path |
|---|---|
| Best checkpoint | `artifacts/runs/20260615_223905_fpga_fault_final_t4/checkpoints/transformer_best.pt` |
| Metrics JSON | `artifacts/runs/20260615_223905_fpga_fault_final_t4/results/metrics.json` |
| Experiment config | `artifacts/runs/20260615_223905_fpga_fault_final_t4/results/transformer_experiment_config.json` |
| History | `artifacts/runs/20260615_223905_fpga_fault_final_t4/results/history.json` |
| Normalization constants | `artifacts/runs/20260615_223905_fpga_fault_final_t4/results/normalization_constants.json` |
| Plots | `artifacts/runs/20260615_223905_fpga_fault_final_t4/plots/` |
