# Project Status

**Project:** Real-Time Machinery Condition Monitoring Using a Transformer Neural Network on FPGA  
**Student:** Mehrdad Sheikh Abbasi, B.Sc., Amirkabir University of Technology  
**Last updated:** 2026-10-08 (physically validated on real MYIR Z-Turn XC7Z020; tag `fpga-demo-validated-20261008`). Demo runbook: `docs/demo_runbook.md`.

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
| 8 | FPGA inference core | **COMPLETE — physically validated** on real MYIR Z-Turn XC7Z020 (full-test 95,404 windows + live multiclass) |
| 9 | Sensor / lab input path | TODO (optional) |
| 10 | FPGA-to-PC communication | **COMPLETE** — real USB-UART boot capture (COM6) + live results over SSH/Ethernet on the board |
| 11 | PC-side display UI | **COMPLETE** — Replay + Live FPGA viewer (`host/fpga_result_viewer.py`) |
| 12 | Final integration, validation, demo | **COMPLETE (hardware-validated)** — physical full-test + live multiclass + UART boot; demo flow in `docs/demo_runbook.md` |

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
ffn=64, 9,492 params). The deployment board is the **MYIR Z-Turn XC7Z020** (PYNQ
2.4 / Python 3.6); the system has been **physically validated** on it (below).

**Current hardware path — SEQ=256 INT10 (aggregated ingress):** the acquisition
window stays 1024×8; the hardware IP quantizes to INT10 and performs an exact 4:1
integer aggregation (1024→256 tokens) internally.

- **Physical FPGA full-test (deployed system, demo headline):** accuracy
  **97.7684%** (93,275/95,404), macro-F1 **95.6539%** (macro-P 95.9332%, macro-R
  95.4233%) on the real board, all 95,404 windows
  (`artifacts/results/fpga_fulltest_20261008_141335/fpga_fulltest_metrics.json`).
- Software INT10 reference: **97.7359%** (93,244/95,404), macro-F1 **95.6282%**;
  the FPGA matches it within **+0.03 pp** (board marginally higher)
  (`artifacts/experiments/transformer_f1_seq256_pool4/int10_ingress_local/combined.json`).
- Live multiclass (120-window class-balanced, real board): accuracy **95.0%**,
  macro-F1 **95.0%**, **120/120** match the recorded FPGA predictions, latency mean
  **16.15 ms** (`artifacts/results/fpga_live_multiclass_20261008_230015/`).
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

Consolidated numbers: **`docs/demo_runbook.md` §1** (canonical, deployed SEQ=256
system). Note: `docs/final_report.md` is an older 2026-09-09 auto-snapshot of the
**SEQ=1024** variant (97.11% int10, hardware "TBD") and is superseded for the
deployed-system numbers by the runbook.

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

**Done (physical board, 2026-10-08, tag `fpga-demo-validated-20261008`):**
- On-board bitstream loading + real AXI-DMA transfer: full-test 95,404 windows and
  120-window live multiclass (all four classes), 120/120 match recorded predictions.
- Real USB-UART boot capture (COM6) and live results over SSH/Ethernet.
- Board confirmed: MYIR Z-Turn XC7Z020 (dma16 deployment, 64 KB DMA transfer).

**Still pending / future work:**
- End-to-end real-time: the core is capable (+26.8%), but the SSH/gzip demo path is
  ~5% below the 48.83 windows/s requirement — a direct sensor/DDR→AXI-DMA feed would
  close it (see `docs/demo_runbook.md` §7).
- Hardware confidence/softmax output (currently CONF=N/A; the class output is valid).
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
