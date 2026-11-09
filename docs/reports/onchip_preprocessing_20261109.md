# Preprocessing inside the FPGA (raw-input core) — 2026-11-09

**Status:** validated on the real board (golden, full test, hard real-time).
The deployed INT10-input argmax overlay is kept unchanged as the fallback.

## 1. What changes

Before: the PC (offline, `tools/export_fpga_testset.py`) z-scored every window
per channel and quantized it to INT10; the FPGA received INT10 codes
(DECISION_LOG D016).

Now: the FPGA receives **raw sensor samples** and does the whole preprocessing
itself. The transformer, its weights and scales, the 4:1 aggregation, the
argmax register and the output stream are unchanged; the DMA frame stays
8,192 × 32-bit words (32 KB). A sensor can therefore replace the file feed
without any processing on the PC or the ARM.

| | INT10-input core (deployed) | raw-input core (new) |
|---|---|---|
| Input word | INT10 code of the z-scored sample | signed 24-bit ADC code |
| z-score + INT10 quantization | PC, float32, offline | FPGA, integer, per window |
| 4:1 aggregation, transformer, argmax | FPGA | FPGA (unchanged) |
| HLS top | `transformer_top_seq256_ingress1024_dsp_margin_argmax.cpp` | `transformer_top_seq256_raw1024_dsp_margin_argmax.cpp` + `onchip_preprocess.h` |

## 2. Raw input contract

`artifacts/preprocessing/raw_input_contract_v1.json`: each channel is modelled as
a 24-bit ADC with a fixed full scale FS_c, chosen from the **training split
only** as the next power of two ≥ 1.25 × max|x|:

| Channel | 0 tach | 1–3 acc. b1 | 4 acc. b2 ax | 5 acc. b2 rad | 6 acc. b2 tan | 7 mic |
|---|---|---|---|---|---|---|
| Training max abs | 5.11 | 53.9 | 15.7 | 12.5 | 28.8 | 204.1 |
| FS_c | 8 | 128 | 32 | 16 | 64 | 256 |

code = clip(round(x / FS_c · 2²³), −2²³, 2²³ − 1). The z-score is scale invariant,
so the FPGA never uses FS_c; it only sets the ADC resolution of the simulation.

## 3. Integer algorithm (bit-exact reference: `src/quantize/onchip_preprocess.py`)

Per window (N = 1024) and channel:

1. S1 = Σx, μ = (S1 + 512) >> 10 (integer mean)
2. d = x − μ, Q = Σd² (exact, < 2⁵⁸)
3. Q = 0 → all codes 0; else k = (62 − bitlen Q)/2, r = isqrt(Q·4^k) (31 bits),
   m = (C·2³¹)/r with C = round(32 / s_in · 2²⁰), sh = 51 − k
4. q = sign(d) · min(511, (|d|·m + 2^(sh−1)) >> sh) = round(d / (s_in · σ))

then the existing exact 4:1 aggregation. One isqrt and one 64-bit division per
channel per window (bit-serial, no DSP needed); one multiply per sample.
Hardware (`onchip_preprocess.h`): three 8,192-cycle passes (store + Σx, Σd²,
quantize + aggregate) plus the factors, ≈ 0.26 ms at 100 MHz.

## 4. Verification plan and results

| Step | Result |
|---|---|
| Python reference vs float32 PC path, 252 real windows | 0.049 % of INT10 codes differ, all by ±1 (input quantization / rounding ties) |
| C++ (`onchip_preprocess.h`, g++) vs Python reference, 20 edge windows | 0 token mismatches (constant/zero channels, full-scale square wave, clamped words, ±1 LSB noise) |
| C++ vs Python reference, first 2,000 real raw test windows | 0 token mismatches (8.2 M tokens), both with wide native integers and with the hardware `ap_int` widths |
| Raw test input (`tools/export_fpga_testset_raw.py`) | 95,404 windows, 0 clipped samples on every channel, window order and labels identical to the previous board input |
| Full test, g++ C-model of the new top (`artifacts/results/onchip_preprocess_20261109/csim_raw_argmax/`) | **97.7705 %** (93,277 / 95,404), macro-F1 95.6686 %, macro AUC 0.9991; `pred_class` = argmax on all windows. Previous INT10-input system: 97.7684 % (93,275), macro-F1 95.6539 %. 46 windows change class (23 become right, 21 become wrong); 42,702 windows have identical logits |
| Full test, Python reference on the raw input (`numpy_reference_raw/`) | logits equal to the C-model on **95,404 / 95,404** windows (`cmodel_vs_reference.json`) |
| Vitis HLS C-simulation, golden `golden_raw1024_argmax` (41 vectors: 37 real incl. 8 ties, 4 edge) | **41/41 exact** (logits + class register) |
| HLS C-synthesis (synthesis-only probe, same source) | core interval 15.04 ms at 100 MHz (INT10-input core 14.905 ms, +0.13 ms); preprocessing block: 0 DSP, 12 BRAM18, ~8.3 k LUT (estimate); estimated Fmax 130 MHz |
| HLS C-synthesis of the final source (`seq256_dma_raw_argmax_z7020_20261109/hls_reports/`) | interval 1,502,098 cycles = 15.02 ms at 100 MHz; estimated Fmax 130 MHz; preprocessing block 0 DSP |
| Vivado 2020.2 post-route, dma16 system, IP synthesized with MAX_DSP 210 | **timing met: WNS +0.167 ns**, WHS +0.014 ns, 0 failing endpoints, 0 routing errors; LUT 36,670 (68.9 %), FF 36,163 (34.0 %), BRAM 76.5 tiles (54.6 %), DSP 210/220 — vs the deployed INT10-input build: +4,904 LUT, +5,211 FF, +6 BRAM, +0 DSP |
| Board: golden `golden_raw1024_argmax` on the real Z-Turn | **41/41 exact** (logits + class register, 8 ties, 4 edge windows), class_source = fpga_register (`raw_core_board_golden_*`) |
| Board: all 95,404 raw test windows | **95,404 / 95,404 logits equal to the C-model**, accuracy **97.7705 %** (93,277), per-window DMA-to-result p50 16.03 ms (INT10-input core 15.95 ms), die ≤ 54.3 °C (`raw_core_board_fulltest_*`) |
| Board: hard real-time acceptance (R1 configuration: RAM windows, 30,000 + 200 warm-up, SCHED_FIFO 50 on CPU 1) | **0 deadline misses in 30,000**, response p50 15.96 ms, p99.9 16.56 ms, max 16.64 ms (INT10-input core: p50 15.82, max 16.55 ms); all 30,200 classes and logits equal to the C-model (`raw_core_board_rt_*`) |

### Timing closure notes

Two builds failed timing before the final one: (1) the per-sample multiplies were
placed in LUTs because the transformer IP is synthesized with MAX_DSP 210 (all
DSP48s already used) and one LUT multiplier took 11 ns; (2) splitting them into
small multiplies let Vivado give those DSPs, which pushed a transformer
requantization multiply into LUTs (−0.85 ns). The final source computes both
products without multipliers (two-bit digits selecting 0/a/2a/3a plus an adder
chain, `pp_mul_digits`), which the HLS scheduler pipelines; the transformer keeps
its original DSP mapping. Vitis HLS 2020.2 also needs the IP core revision set to
"1" when packaging (date-based revision overflow), and the Vivado project uses a
short path (`tmp/vraw`) because the HLS module file names exceed the Windows
260-character limit otherwise. Build script: `tools/build_raw_core.ps1`.
