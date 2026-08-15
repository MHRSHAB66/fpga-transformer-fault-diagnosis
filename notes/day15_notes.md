# Day 15 Notes

## Main focus
Day 15 took the project from "a trained model" to a **complete, verified software
FPGA path** for the deployed architecture, and prepared everything on the board
side so only Vitis synthesis + the physical board remain.

## What was done
- **Fixed the integer bit-sweep on Kaggle** (two bugs): the data path
  (`DATA_RAW_DIR` pointed at a non-existent Kaggle path) and the model
  state-dict key names (`TuningTransformer` vs `IntegerTransformer`); added a
  1:1 key remap.
- **Bit-width sweep `[16,14,12,10,8]`** on the FPGA-winner `F_1layer`:
  **int10 is the sweet spot** (near-lossless); int8 collapses the imbalance class.
- **Definitive full-test int10 accuracy** measured on all 95,404 windows via
  **4 parallel sharded Kaggle CPU kernels** (the numpy int64 reference is too
  slow for one 12 h session): **int10 = 97.11%** vs float 97.55% (drop 0.44%),
  all classes healthy.
- **Stage 7/8 export for F_1layer @ int10** into a parallel `hls_f_int10/`
  (leaving the day14 A_baseline int16 core untouched): weights/biases/requant
  multipliers/PE/LayerNorm/LUTs + int10 golden vectors.
- **Compiler-free verification**: independent NumPy mirror reproduces the export
  bit-for-bit -- **PASS 16/16 class + 16/16 logits**.
- **C++ core** (`hls_f_int10/`, 1 layer, int10) ready for g++/Vitis.
- **PC side (Stage 10/11)**: `host/frame_protocol.py` + `host/fault_monitor.py`
  (UART receiver + Tkinter UI + mock/file/serial sources), tested end to end.
- **PYNQ side**: `pynq/fault_classifier.py` (verified NumPy backend + hardware
  backend template) and `pynq/run_demo.py` (window -> classify -> UART frame).
- **Vitis HLS scaffold** (`hls_f_int10/vitis/`): flat weights header, ap_int
  typedefs, synthesizable AXIS top, `run_hls.tcl` (Zynq-7010), and
  `SYNTHESIS_GUIDE.md`.
- **Stage 12 report generator** (`src/reporting/generate_final_report.py`) ->
  `docs/final_report.md`.

## Current result
Full software FPGA path is done and verified: **int10 = 97.11%** on the full test
set, bit-exact HLS contract, and a runnable PC + PYNQ demo (NumPy backend). Only
Vitis synthesis (resource/latency report) and the on-board UART link remain --
both need the laptop/board, not this desktop.

## Artifacts
- `kaggle/int10_shard/int10_fulltest_combined.json`
- `kaggle/sweep_multibit/integer_multibit_sweep_F.json`
- `hls_f_int10/` (data, golden, C++ core, `vitis/`), `host/`, `pynq/`
- `docs/final_report.md`

## Final note
Day 15 was about **hardening and hardware-preparing** the chosen model
(F_1layer @ int10), not model accuracy. The accuracy/quantization story is now
closed and defensible; the remaining work is physical-board bring-up.
