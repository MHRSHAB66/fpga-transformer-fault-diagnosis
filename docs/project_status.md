# Project Status

**Project:** Real-Time Machinery Condition Monitoring Using a Transformer Neural Network on FPGA  
**Student:** Mehrdad Sheikh Abbasi, B.Sc., Amirkabir University of Technology  
**Last updated:** 2026-08-16

---

## Stage Status

| Stage | Name | Status |
|---|---|---|
| 0 | Project setup, environment, config | **COMPLETE** |
| 1 | MAFAULDA data discovery, class mapping | **COMPLETE** |
| 2 | Preprocessing, windowing, DataLoaders | **COMPLETE** |
| 3 | MLP baseline (sanity check) | **COMPLETE** |
| 4 | Lightweight Transformer training | **COMPLETE** |
| 5 | Clean evaluation, metrics export, artifacts | **COMPLETE** |
| 6 | Quantization & fixed-point preparation | **COMPLETE** (int10 selected) |
| 7 | Hardware specification & golden vectors | **COMPLETE** (F@int10 export, verify 16/16) |
| 8 | FPGA inference core | **SOFTWARE DONE** — Vitis synthesis on laptop pending |
| 9 | Sensor / lab input path | TODO (optional) |
| 10 | FPGA-to-PC communication (UART) | **SOFTWARE DONE** (host/ receiver, tested) — on-board link pending |
| 11 | PC-side display UI | **COMPLETE** (host/fault_monitor.py) |
| 12 | Final integration, validation, demo | **IN PROGRESS** (docs/final_report.md; HW numbers pending synth) |

---

## Software Phase (Stages 0-5): COMPLETE

The software/model phase is complete as of 2026-06-15.

**Baseline model (highest accuracy):** 2-layer Transformer,
**98.22%** test accuracy on 95,404 unseen windows.
Checkpoint: `artifacts/runs/20260615_223905_fpga_fault_final_t4/checkpoints/transformer_best.pt`.
See `docs/final_software_result.md`.

## FPGA-deployed model (Stages 6-8): F_1layer @ int10

For the FPGA path a smaller architecture was selected to reduce the eventual
resource and latency cost: **F_1layer** (1 encoder layer, d_model=32, 4 heads,
ffn=64, 9,492 params). The final FPGA board/device is not selected yet, so fit
and resource-budget claims remain pending.

- Float full-test accuracy: **97.55%**
- **int10 full-test accuracy: 97.11%** (drop 0.44%, no class collapse) --
  measured on all 95,404 windows via 4 sharded Kaggle kernels
  (`kaggle/int10_shard/int10_fulltest_combined.json`).
- int10 chosen from a bit-width sweep `[16,14,12,10,8]`; int8 collapses the
  imbalance class (`kaggle/sweep_multibit/`).
- Bit-exact HLS export verified 16/16 (`hls/f1_int10/verify_export.py`).

Consolidated numbers: `docs/final_report.md` (regen:
`python -m src.reporting.generate_final_report`).

---

## What is done vs still pending

**Done (runnable / verified on this machine):**
- Quantization to int10 with full-test accuracy (97.11%).
- Bit-exact HLS/portable-C++ core + golden vectors (`hls/f1_int10/`, verify 16/16).
- Vitis project scaffold (`hls/f1_int10/vitis/`: weights header, ap_int typedefs,
  synthesizable top, run_hls.tcl, SYNTHESIS_GUIDE.md).
- PC-side UART receiver + live UI + frame protocol (`host/`, tested end-to-end).
- PYNQ driver with a verified NumPy backend + hardware-backend template (`pynq/`).

**Still pending (needs the board / Vitis on the laptop):**
- Actual Vitis synthesis + resource/latency report (LUT/FF/BRAM/DSP/latency/II).
- On-board bitstream + real UART link demo.
- Optional live sensor input path (Stage 9).

Do not claim synthesized-hardware resource usage, on-board latency, or a live
sensor demo until those are produced. The int10 accuracy and bit-exact
functional equivalence ARE established and may be reported.

---

## Branch

Current work branch: `day15-arch-sweep-cleanup` (Stages 6-8 quantization/HLS +
Stages 10/11 PC-side). `main` lags at the day11-era clean state; day-per-branch
history is preserved on `origin`.
