# Final Results Summary — Full Training Run

**Run ID:** `20260615_223905_fpga_fault_final_t4`  
**Date:** 2026-06-15  
**Kernel:** `mehrdadsheikhabbasi/fpga-fault-final-train` version 3  
**Mode:** full  
**GPU:** Tesla T4  

---

## Dataset Split

| Split | Files | Windows | Class breakdown |
|---|---|---|---|
| Train | 1,365 | 333,060 | normal:8,296 / imbalance:56,852 / misalignment:85,156 / bearing:182,756 |
| Val | 195 | 47,580 | normal:1,220 / imbalance:8,052 / misalignment:12,200 / bearing:26,108 |
| Test | 391 | 95,404 | normal:2,440 / imbalance:16,348 / misalignment:24,156 / bearing:52,460 |
| **Total** | **1,951** | **476,044** | |

- Window size: 1024 samples × 8 channels → shape [B, 1024, 8]
- Window stride: 1024 (no overlap)
- Normalization: per-window Z-score
- File-level split (all windows from same CSV file stay in same split — no leakage)
- Class imbalance ratio: 22.03×

---

## Model Configuration

| Parameter | Value |
|---|---|
| input_dim | 8 channels |
| seq_len | 1024 |
| d_model | 32 |
| nhead | 4 |
| num_layers | 2 |
| dim_feedforward | 64 |
| dropout | 0.1 |
| num_classes | 4 |
| Architecture | Input projection → Sinusoidal PE → 2× Encoder → Mean Pool → Classifier |
| Checkpoint size | 380 KB |

---

## Training Configuration

| Parameter | Value |
|---|---|
| batch_size | 64 |
| learning_rate | 1e-3 |
| optimizer | Adam |
| loss | Weighted cross-entropy |
| class_weights | normal=10.04, imbalance=1.46, misalignment=0.98, bearing=0.46 |
| max_epochs | 30 |
| early_stopping_patience | 6 |
| early_stopping_monitor | val_loss |
| random_seed | 42 |
| Preprocessing flags | all OFF (no lowpass, Wiener, SMOTE, noise, resampling) |

---

## Training Outcome

| Metric | Value |
|---|---|
| best_epoch | 10 |
| stopped_epoch | 16 |
| early_stopped | **True** |
| best_val_loss | 0.0353 |
| val_acc_at_best_epoch | **98.68%** |

### Per-epoch history (all 16 epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.3577 | 86.70% | 0.2946 | 88.94% |
| 2 | 0.1570 | 94.33% | 0.1580 | 94.13% |
| 3 | 0.1197 | 95.69% | 0.1994 | 93.36% |
| 4 | 0.0986 | 96.44% | 0.0777 | 97.24% |
| 5 | 0.0856 | 96.89% | 0.1700 | 94.12% |
| 6 | 0.0759 | 97.22% | 0.1549 | 94.43% |
| 7 | 0.0655 | 97.58% | 0.0542 | 98.11% |
| 8 | 0.0628 | 97.74% | 0.0394 | 98.65% |
| 9 | 0.0591 | 97.86% | 0.0839 | 97.24% |
| **10** | **0.0535** | **98.04%** | **0.0353** | **98.68%** ← BEST |
| 11 | 0.0499 | 98.17% | 0.0662 | 97.77% |
| 12 | 0.0461 | 98.29% | 0.0409 | 98.43% |
| 13 | 0.0446 | 98.32% | 0.0411 | 98.63% |
| 14 | 0.0433 | 98.45% | 0.0606 | 97.99% |
| 15 | 0.0402 | 98.53% | 0.0459 | 98.54% |
| 16 | 0.0400 | 98.53% | 0.0404 | 98.59% ← STOPPED |

---

## Test Set Results

### Overall
| Metric | Value |
|---|---|
| test_acc | **98.22%** |
| test_loss | 0.0468 |
| test_macro_F1 | 0.9711 |
| test_weighted_F1 | 0.9822 |
| test_macro_F2 | **0.9780** |
| test_weighted_F2 | **0.9822** |

### Per-class test accuracy
| Class | Accuracy | Precision | Recall | F1 | F2 | Support |
|---|---|---|---|---|---|---|
| Normal (0) | 97.66% | 0.910 | 0.977 | 0.942 | 0.963 | 2,440 |
| Imbalance (1) | 98.84% | 0.958 | 0.988 | 0.973 | 0.982 | 16,348 |
| Misalignment (2) | 98.76% | 0.979 | 0.988 | 0.983 | 0.986 | 24,156 |
| Bearing Fault (3) | 97.80% | 0.995 | 0.978 | 0.987 | 0.981 | 52,460 |
| **Macro** | — | **0.960** | **0.983** | **0.971** | **0.978** | — |

### Test confusion matrix (rows=true, cols=predicted)

```
              Normal  Imbalance  Misalignment  BearingFault
Normal          2383          3            52             2
Imbalance         21      16159            40           128
Misalignment     152         42         23856           106
BearingFault      63        670           422         51305
```

---

## Safety Summary (Fault vs. Normal)

| Metric | Value |
|---|---|
| actual_faults | 92,964 windows |
| captured_faults | 92,728 |
| missed_faults | **236** |
| fault_recall | **99.75%** |
| actual_normal | 2,440 windows |
| false_alarms | **57** |
| false_alarm_rate | **2.34%** |

---

## Runtime & Resources

| Metric | Value |
|---|---|
| GPU | Tesla T4 |
| Peak GPU memory | **0.28 GiB (283 MB)** |
| Preprocessing (cold cache) | 1,525 s (~25 min) |
| Per-epoch training time | ~922 s (~15 min) |
| Per-epoch val evaluation | ~27 s |
| Total training time (16 epochs) | 15,189 s (~4 h 13 min) |
| First train step | 1.53 s |
| Estimated RAM for all splits | ~14.5 GiB (materialized tensors) |

---

## Artifact Paths

| Artifact | Local path |
|---|---|
| Best checkpoint (.pt) | `artifacts/runs/20260615_223905_fpga_fault_final_t4/checkpoints/transformer_best.pt` |
| Metrics JSON | `artifacts/runs/20260615_223905_fpga_fault_final_t4/results/metrics.json` |
| Experiment config | `artifacts/runs/20260615_223905_fpga_fault_final_t4/results/transformer_experiment_config.json` |
| History JSON | `artifacts/runs/20260615_223905_fpga_fault_final_t4/results/history.json` |
| Normalization constants | `artifacts/runs/20260615_223905_fpga_fault_final_t4/results/normalization_constants.json` |
| Run summary | `artifacts/runs/20260615_223905_fpga_fault_final_t4/logs/run_summary.json` |
| Model config | `artifacts/runs/20260615_223905_fpga_fault_final_t4/checkpoints/transformer_baseline_model_config.json` |
| Plots directory | `artifacts/runs/20260615_223905_fpga_fault_final_t4/plots/` |
