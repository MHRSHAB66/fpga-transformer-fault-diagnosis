# Vitis HLS synthesis guide — F_1layer @ int10

Vitis HLS 2020.2 is installed on the Windows machine. The AXI-Stream top with
real `ap_int<10>` activations was C-simulated against the exported golden
vectors: 16/16 exact logits and predictions. This does not establish RTL
equivalence, final board fit, or end-to-end latency.

Target part: **TBD**. The final FPGA board/device has not been selected. Before
running synthesis, set `FPGA_PART` to the exact Vitis part and record that
device's DSP, BRAM, LUT and FF budget in the project report.

## Files here
| file | role |
|------|------|
| `run_hls.tcl` | project script (C-sim now; csynth/cosim/export gated by flags) |
| `run_top_csim.tcl` | part-independent C-simulation of the actual AXI-Stream top |
| `run_top_csynth.tcl` | baseline C-synthesis for an explicit, provisional or final part |
| `testbench_top.cpp` | checks four logits, class, and output `last` against golden vectors |
| `weights_int10.h` | all int10 constants as flat `static const` arrays (regen: `python gen_weights_header.py`) |
| `ap_types.h` | `act_t/acc_t/lut_t` typedefs; real `ap_int` in synthesis and top C-sim |
| `transformer_top.cpp` | synthesizable top: AXI-Stream in/out, LAYERS=1 inlined, params from the header |

## Stage A — C-simulation of the synthesis top (no part or synthesis needed)
From the repository root, use an ignored scratch directory for the HLS project:
```powershell
New-Item -ItemType Directory -Force tmp/hls_top_csim | Out-Null
Push-Location tmp/hls_top_csim
vitis_hls -f ../../hls/f1_int10/vitis/run_top_csim.tcl
Pop-Location
```
If `vitis_hls` is not on `PATH`, invoke its installed `vitis_hls.bat` instead.
Confirm both `AXI top exact vectors: 16/16` and `CSim done with 0 errors`;
do not rely on the process exit code alone. The script takes source paths from
its own location and leaves generated files in `tmp/hls_top_csim/`.

The older `run_hls.tcl` tests the portable reference core, not
`transformer_top.cpp`, and requires `FPGA_PART` even for C-simulation.

## Stage B — baseline C-synthesis (get the first resource/latency report)
Set `FPGA_PART` to an exact device identifier and run `run_top_csynth.tcl` from
an ignored scratch directory. The **provisional** probes used
`xc7z010clg400-1` and `xc7z020clg400-1` with a 10 ns target in Vitis HLS
2020.2. This does not select a final board. For the PYNQ-Z2-sized probe, set
`FPGA_PART=xc7z020clg400-1`, change to a fresh ignored scratch directory, and
run `vitis_hls -f` on `run_top_csynth.tcl`.

The Z-7020 probe uses `xc7z020clg400-1`, the part on PYNQ-Z2. These are HLS
*estimates* for successive versions of the same model and 10 ns target:

| Top version | BRAM_18K | DSP | FF | LUT |
|-------------|---------:|----:|---:|----:|
| original `long` buffers | 1497 | 112 | 29965 | 51067 |
| int10 buffers, int32 accumulators | 286 | 85 | 19293 | 44015 |
| plus in-place residual | 262 | 71 | 18033 | 43381 |
| plus attention/FFN buffer reuse (current) | 242 | 71 | 18023 | 43381 |
| Z-7020 available | 280 | 220 | 106400 | 53200 |

The pre-row-wise **core-only HLS estimate fits** Z-7020, with 38 BRAM_18K and
9819 LUT left before DMA, control, and other system logic. This is not a
board/bitstream fit claim. The original `long` arrays synthesized as 64-bit
memories; the activation arrays are now 10-bit. The current row-wise top retains
full-sequence K/V and computes one query and one FFN token at a time; it passed
16/16 exact golden vectors in `tmp/hls_top_csim_fused_row/`. Its Z-7010
resource estimate is a separate synthesis probe, not the 242-BRAM figure above.
The first Z-7010 probe of this row-wise version was stopped before Vitis HLS
produced a report. A subsequent isolated full-sequence probe completed for
`xc7z010clg400-1` at 10 ns after 4891 seconds (about 81.5 minutes):

| Top version | BRAM_18K | DSP | FF | LUT |
|-------------|---------:|----:|---:|----:|
| row-wise, Q40, `finish_token` isolated, SEQ=1024 | 104 | 40 | 21072 | 24518 |
| Z-7010 available | 120 | 80 | 35200 | 17600 |

The LUT estimate is 139% of Z-7010 capacity, so this version does **not**
fit Z-7010. Top-level latency and interval remain `?`; the 7.297 ns clock
estimate is not a measured per-window inference time. The report is at
`tmp/hls_z7010_finish_token/finish_token_csynth/sol1/syn/report/transformer_top_csynth.rpt`.
The corresponding C simulation passed all 16 golden vectors exactly
(`tmp/hls_top_csim_finish_token/`). A later bounded-rsqrt change also passed
16/16 (`tmp/hls_top_csim_bounded_rsqrt/`), but its resource and timing effects
have not been synthesized yet.
An isolated Q32 coefficient export (`tmp/hls_q32_csim/`) also matched all
16 golden vectors exactly in C simulation. This does not prove equivalence
for every input; keep Q40 as the maintained format until a full evaluation
and resource comparison justify changing it.
For the same isolated SEQ=64 diagnostic topology (before `finish_token` and
bounded-rsqrt changes), Q32 estimated 40 BRAM, 33 DSP, 14613 FF, and 20991
LUT, versus Q40's 43 BRAM, 36 DSP, 18994 FF, and 21943 LUT. Q32 reduces
resources in this diagnostic, but still exceeds the Z-7010 LUT budget
(17600). Do not extrapolate these numbers to SEQ=1024.
The O(SEQ²) attention remains, so memory/latency optimization may still be needed.
Top-level latency and interval are `?` in the HLS report; no end-to-end
real-time claim can be made. The estimated clock period is 7.30 ns versus the
10 ns target, not post-implementation timing. Each probe's raw report is at
`fault_int10_top_csynth/sol1/syn/report/transformer_top_csynth.rpt` under its
own ignored scratch directory.

## Stage C — optimize to fit the board

**2026-09-21 evidence update:** the bounded-rsqrt Q40 snapshot subsequently
completed synthesis for provisional `xc7z020clg400-1`. Preserved inputs,
C-simulation log (16/16 exact), and synthesis report are under
`artifacts/results/hls_z7020_q40_bounded_snapshot/`. This newer report gives
104 BRAM_18K, 40 DSP, 20,610 FF, 24,738 LUT and a bounded top latency of
74,246,781 cycles (0.742 s at the 10 ns target), with interval 74,246,782
cycles. Earlier `?` latency statements above describe the older probes.
This remains a core HLS estimate, not a board or real-time result.

1. **Narrow the datapath (done for activations/accumulators).** Main activation
   arrays use `act_t=ap_int<10>`, MAC and softmax sums use `acc_t=ap_int<32>`.
   Current C-sim remains exact; re-check numeric bounds after any re-export.
2. **Kill floating point.** Convert each `double` requant/affine constant to an
   integer multiplier `M0 · 2^-shift` (or `ap_fixed<18,2>`); regenerate them in
   `gen_weights_header.py`. Removes FP DSPs/LUTs.
3. **Reduce sequence-sized intermediates (partly done).** The current top
   stores full K/V but computes Q, context and the post-attention/FFN path one
   token at a time. Each query still attends to all 1024 keys, preserving the
   model. Input and projected `x` are still full-sequence buffers; there is no
   `dataflow` pipeline or measured throughput improvement yet.
4. **Pipeline the MAC loops.** `#pragma HLS PIPELINE` on the inner `Din`/`HD`
   loops; `#pragma HLS ARRAY_PARTITION` on the small dims (`D=32`, `HD=8`,
   `DFF=64`) for parallel MACs; budget DSP and LUT for the selected board and
   reserve space for DMA/control before declaring system fit.
5. **Re-run csynth**, read II/latency + LUT/FF/BRAM/DSP, iterate until it fits.
   F_1layer was chosen precisely to halve the encoder HW vs the 2-layer baseline;
   if attention still overflows BRAM, the documented fallbacks are a shorter
   window (512) or windowed/strided attention — both require a re-export, not a
   silent change.

## Stage D — export + integrate
1. `DO_EXPORT 1` → `export_design -format ip_catalog` (Verilog RTL IP).
2. Once a compatible Zynq board and its PYNQ image are selected, make a Vivado
   block design with PS + AXI DMA + this IP (AXIS in/out) + AXI-Lite control;
   generate bitstream → `design.bit` + `design.hwh`.
3. Copy both to the board; in `pynq/fault_classifier.py` `HardwareBackend`, set
   the overlay path and confirm the DMA name (`overlay.ip_dict`).
4. Sanity-check on hardware with `hls/f1_int10/golden/` (logits must match), then
   `python -m pynq.run_demo --bitstream design.bit --port /dev/ttyPS0` with the
   PC running `python -m host.fault_monitor --source serial --port <COM>`.

## Report artifacts to capture (for the thesis, Stage 12)
From the Vitis reports: **LUT, FF, BRAM, DSP, latency, II, Fmax**; from the board:
UART throughput/stability. Save alongside the software metrics (float 97.55% /
int10 97.11%).
