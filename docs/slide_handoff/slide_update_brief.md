# Slide Update Brief

## Current project goal (2–3 sentences)
This project designs and implements a complete real-time machinery condition monitoring system
that classifies vibration faults using a lightweight Transformer neural network mapped to an FPGA.
The classifier receives 8-channel time-series windows from rotating machinery (MAFAULDA dataset)
and outputs one of four fault classes: normal, imbalance, misalignment, or bearing fault.
The FPGA sends results over UART to a PC display — no cloud, no PC-side inference.

---

## Final software/model pipeline

```
MAFAULDA CSV files (50 kHz, 8 channels)
  → load & quality filter (median fill, missing < 20%)
  → file-level stratified split  [70% train / 10% val / 20% test]
      (all windows from one file stay in the same split — no leakage)
  → fixed windows:  1024 samples, stride = 1024, no overlap  (~20 ms per window)
  → per-window Z-score normalization  (μ=0, σ=1 per window per channel)
  → Transformer encoder classifier  (8 → 32 → 2×encoder → mean pool → 4 logits)
  → weighted cross-entropy loss  (imbalance ratio = 22×)
  → Adam optimizer, lr = 1e-3
  → best checkpoint saved by minimum validation loss
  → test-set evaluation (once, at the end)
```

---

## Agreed final preprocessing decisions

| Decision | Value | Reason |
|---|---|---|
| Window size | 1024 samples | ~20 ms at 50 kHz; compatible with FPGA buffer |
| Stride | 1024 (no overlap) | No cross-window leakage; simpler FPGA streaming |
| Normalization | per-window Z-score | Per-file and global-train tested; per-window best |
| Resampling | OFF | 22 kHz resonance is class-discriminative; aliasing risk |
| Lowpass filter | OFF | Not needed with per-window norm |
| Wiener denoising | OFF | Did not improve val metrics |
| SMOTE / augmentation | OFF | Class-weighted loss handles imbalance |
| Tail windows | OFF | Standard stride-aligned windowing used |
| Channels | all 8 | Vibration-only variant not retrained |

---

## Final model architecture

| Component | Value |
|---|---|
| Input | [batch, 1024 tokens, 8 channels] |
| Input projection | Linear(8 → 32) — projects channels to d_model |
| Positional encoding | Sinusoidal, added to each token |
| Encoder layers | 2 × TransformerEncoderBlock |
| Self-attention | MultiheadAttention (4 heads, d_model = 32) |
| FFN | Linear(32→64) → GELU → Linear(64→32) |
| Residuals | LayerNorm after each sub-layer |
| Pooling | Global mean over 1024 tokens |
| Classifier head | LayerNorm → Linear(32→16) → GELU → Linear(16→4) |
| Output | 4 logits (normal / imbalance / misalignment / bearing) |
| Checkpoint size | 380 KB |
| Peak GPU memory | 0.28 GiB |

Hyperparameters: d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1

---

## Final training setup

| Parameter | Value |
|---|---|
| Dataset | Full MAFAULDA (1,951 files) |
| Split | 1,365 train / 195 val / 391 test files |
| Windows | 333,060 train / 47,580 val / 95,404 test |
| Batch size | 64 |
| Learning rate | 1e-3 (Adam) |
| Max epochs | 30 |
| Early stopping patience | 6 (monitoring val loss) |
| Loss | Weighted cross-entropy (22× imbalance ratio) |
| Class weights | normal×10.04, imbalance×1.46, misalignment×0.98, bearing×0.46 |
| GPU | Tesla T4 (Kaggle) |
| Random seed | 42 |

---

## Final results

### Training outcome
- **Best epoch:** 10
- **Early stopped:** Yes, at epoch 16 (patience = 6)
- **Best val loss:** 0.0353
- **Val accuracy at best:** 98.68%

### Test set results
| Metric | Value |
|---|---|
| Test accuracy | **98.22%** |
| Test loss | 0.0468 |
| Test macro F2 | **0.9780** |
| Test weighted F2 | **0.9822** |

### Per-class test accuracy
| Class | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|
| Normal | 97.66% | 0.910 | 0.977 | 0.942 | 0.963 |
| Imbalance | 98.84% | 0.958 | 0.988 | 0.973 | 0.982 |
| Misalignment | 98.76% | 0.979 | 0.988 | 0.983 | 0.986 |
| Bearing Fault | 97.80% | 0.995 | 0.978 | 0.987 | 0.981 |
| Macro avg | — | 0.960 | 0.983 | 0.971 | 0.978 |

### Safety summary
| Metric | Value |
|---|---|
| Fault recall | **99.75%** (missed 236 / 92,964 fault windows) |
| False alarm rate | **2.34%** (57 / 2,440 normal windows) |

---

## Claims that are now safe to make
- The lightweight Transformer (380 KB, 0.28 GiB GPU) achieves 98.22% test accuracy on full MAFAULDA.
- File-level splitting prevents data leakage; test set was never seen during training.
- Weighted cross-entropy effectively handles the 22× class imbalance.
- Fault recall of 99.75% makes the classifier suitable as a safety-oriented detector.
- The model is small enough (d_model=32, 2 layers) for FPGA mapping.
- Per-window Z-score normalization is the final preprocessing choice.
- Early stopping at epoch 16 confirms the model generalized well (best at epoch 10).

## What must NOT be claimed yet
- FPGA implementation is NOT complete — do not say "deployed on FPGA".
- Real-time hardware inference latency is NOT measured.
- UART / PC display is NOT implemented.
- Quantization has NOT started — Stage 6 is the next step.
- Do not claim any sensor/hardware test results — only MAFAULDA prepared windows were used.

---

## Next stage: Stage 6 — Quantization & Fixed-Point Preparation
- Convert the saved float32 checkpoint to int8 / int16 fixed-point.
- Measure accuracy drop between float and quantized models.
- Export weights, scales, and zero-points for RTL consumption.
- Generate golden input/output vectors for testbench validation.
- Document the quantized model config.

Starting point: `artifacts/runs/20260615_223905_fpga_fault_final_t4/checkpoints/transformer_best.pt`
