# Project Status

**Project:** Real-Time Machinery Condition Monitoring Using a Transformer Neural Network on FPGA  
**Student:** Mehrdad Sheikh Abbasi, B.Sc., Amirkabir University of Technology  
**Last updated:** 2026-09-30 (SEQ=256 INT10 DMA-integrated system; provisional XC7Z020)

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
| 8 | FPGA inference core | **CSim + C-synth + DMA-integrated post-route + bitstream DONE** on provisional/generic XC7Z020 — physical board pending |
| 9 | Sensor / lab input path | TODO (optional) |
| 10 | FPGA-to-PC communication (UART) | **SOFTWARE DONE** (host/ receiver, tested) — on-board link pending |
| 11 | PC-side display UI | **COMPLETE** (host/fault_monitor.py) |
| 12 | Final integration, validation, demo | **IN PROGRESS** — provisional HW numbers available (`docs/reports/hls_seq256_int10_results.md`); physical-board demo pending |

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
ffn=64, 9,492 params). The final FPGA board/device is **not selected yet**; the
XC7Z020 numbers below are a **provisional/generic** configuration for evidence.

**Current hardware path — SEQ=256 INT10 (aggregated ingress):** the acquisition
window stays 1024×8; the hardware IP quantizes to INT10 and performs an exact 4:1
integer aggregation (1024→256 tokens) internally.

- Exact-integer-ingress full-test accuracy: **97.74%** (93,244/95,404),
  macro-F1 **95.63%** (`artifacts/experiments/transformer_f1_seq256_pool4/int10_ingress_local/combined.json`).
- Vitis HLS CSim: **16/16 exact**; DMA-integrated Vivado post-route timing met
  (WNS +0.036 ns), 210/220 DSP, 31,687 LUT, bitstream/HWH/XSA generated. See
  `docs/reports/hls_seq256_int10_results.md` and
  `artifacts/results/seq256_dma_system_dsp_margin_z7020/`.

**Earlier SEQ=1024 INT10 variant (historical):**
- Float full-test accuracy: **97.55%**; **int10 full-test accuracy: 97.11%**
  (drop 0.44%, no class collapse), all 95,404 windows
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

**Done (provisional/generic XC7Z020, in addition to the above):**
- Vitis HLS C-synthesis (est. 14.905 ms interval at 100 MHz) for the SEQ=256
  INT10 ingress core.
- DMA-integrated Vivado 2020.2 implementation (PS7 + AXI-DMA + IP), post-route
  timing met, resources measured (210/220 DSP with a reserved 10-DSP margin),
  bitstream/HWH/XSA generated. Package:
  `artifacts/results/seq256_dma_system_dsp_margin_z7020/`.

**Still pending (physical board only):**
- On-board bitstream loading, real DMA transfer, live UART link demo.
- Board/device selection (XC7Z020 is provisional/generic, not confirmed).
- Optional live sensor input path (Stage 9).

Report the provisional post-route resource/timing numbers ONLY with the
provisional-part caveat. Do not claim physical-board resource usage, on-board
latency, or a live sensor/UART demo until those are produced. The int10 accuracy,
bit-exact functional equivalence, and provisional post-route closure ARE
established and may be reported.

---

## Branch

Current work branch: `day15-arch-sweep-cleanup` (Stages 6-8 quantization/HLS +
Stages 10/11 PC-side). `main` lags at the day11-era clean state; day-per-branch
history is preserved on `origin`.
