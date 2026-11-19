# Next Steps: Stage 6 — Quantization & Fixed-Point Preparation

> **Historical plan, superseded.** This was written before the hardware path was implemented. The active deployment uses INT10 (not the proposed INT8 target), and physical FPGA tests have been completed. Retained solely to document the original plan. See [`project_status.md`](project_status.md) and [`reports/consolidated_results_20261022.md`](reports/consolidated_results_20261022.md).

The software phase (Stages 0–5) is complete. This document describes what Stage 6 involves
and what must be done before hardware implementation can begin.

---

## Starting point

- Checkpoint: `artifacts/runs/20260615_223905_fpga_fault_final_t4/checkpoints/transformer_best.pt`
- Float32 weights, 381 KB
- Model config: `artifacts/runs/20260615_223905_fpga_fault_final_t4/results/transformer_experiment_config.json`
- Test accuracy (float32): **98.22%**

---

## Stage 6 tasks

### 6a — Post-training quantization (PTQ)
- Apply int8 symmetric quantization using PyTorch PTQ utilities.
- Use the training/validation data as calibration set (not the test set).
- Measure accuracy drop vs. the float32 model.
- Target: < 1% accuracy drop at int8.

### 6b — Fixed-point simulation in Python
- Simulate the quantized model in Python using fixed-point arithmetic.
- Compare results with the float32 model on the same test set.
- Document: which layers are sensitive to quantization.

### 6c — Weight export
- Export all weights, biases, scales, and zero-points to a readable format (JSON or binary).
- Ensure the export format is usable by RTL/HLS.
- Document the exact data types and bit widths used.

### 6d — Golden vector generation
- Generate fixed-point input/output pairs from the quantized Python model.
- These serve as ground truth for FPGA testbench validation.
- Format: one input window → class probabilities/decision → testbench-readable file.

### 6e — Accuracy comparison report
- Table: float32 accuracy vs. int8 accuracy vs. int16 accuracy (if applicable).
- Confusion matrix for quantized model.
- Document which quantization scheme and bit width are selected for hardware.

---

## Constraints from the FPGA target

| Constraint | Implication |
|---|---|
| Model must fit in FPGA BRAM/LUT | Keep weight count low — d_model=32 is already very small |
| Fixed-point arithmetic only | All multiply-accumulate must use integer or fixed-point ops |
| No floating-point support | softmax approximation may be needed in hardware |
| Latency budget (TBD) | Determine once resource mapping is done in Stage 8 |
| Input format | Must match the per-window Z-score normalization used in training |

---

## What Stage 6 does NOT do

- Stage 6 does not implement hardware.
- Stage 6 does not change the model architecture.
- Stage 6 does not retrain the model (unless quantization-aware training is needed).
- Stage 6 does not change the preprocessing pipeline.

---

## After Stage 6

- Stage 7: Define the hardware interface (input buffer size, output format, weight memory layout).
- Stage 8: Implement FPGA inference core (RTL or HLS).
- Stages 9–12: Sensor input, UART, PC display, final integration.
