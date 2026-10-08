# Fitting the bit-exact int10 Transformer core on Zynq-7010 (xc7z010clg400-1)

Measured campaign, Vitis HLS 2020.2, 10 ns target, `-DHLS_USE_AP_INT`
(`ap_int<10>` act / `ap_int<32>` acc / `ap_int<20>` lut). All variants preserve the
approved model (SEQ=1024, D=32, 4 heads, 1 layer), weights, int10/Q40 scales, LUT
contents, class IDs, tensor order, AXI interface and golden vectors. Every variant
is **bit-exact** (native g++ A/B byte-identical to baseline; Vitis CSim 16/16).

`xc7z010clg400-1` / `xc7z020clg400-1` are **provisional experimental targets** — the
final board is not selected. All figures are **C-synthesis estimates**, core only
(no DMA/PS/bitstream), not post-implementation.

## Stage 1 — measured 7010 LUT hierarchy of v3 (core = 26,607 LUT, top 152.6%)

| Consumer | LUT | Note |
|---|---:|---|
| finish_token (1 inst) | 8,779 | 2× layernorm + 2× linear + gelu + residual |
| Expression glue | 4,846 | residual/gelu/requant/pe/pool inline |
| **8× context-output requant** | **4,440** | 8×555 — **confirmed** (HD=8 unrolled output loop) |
| classifier layernorm | 2,477 | |
| Multiplexer / Register | 2,140 / 1,568 | |
| softmax sdiv | 1,347 | 32-bit divider |

Shareable (execute sequentially, duplicated in hardware): the 8 context-output
requant units; the 3 layernorm datapaths; the requant/residual 64-bit
multiply+round-to-even logic scattered across linears/residuals/pooling.

## Measured results (all bit-exact, CSim 16/16)

| Variant | Device | Latency cyc | ms@10ns | Interval | Est.clk | Slack | BRAM | DSP | FF | LUT | Fit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| baseline (frozen) | 7020 | 74,246,781 | 742 | 74,246,782 | 7.297 | — | 104/280 | 40/220 | 20,610 | 24,738/53,200 | yes |
| v3 BRAM-fixed | 7020 | 44,930,707 | 449 | 44,930,708 | 7.300 | 0.00 | 108/280 | 46/220 | 23,736 | 26,857/53,200 | yes |
| v3 BRAM-fixed | **7010** | 44,930,707 | 449 | 44,930,708 | 7.300 | 0.00 | 108/120 | 46/80 | 23,736 | **26,857/17,600 (152.6%)** | **no** |
| A share ctx-requant | 7010 | 44,914,033 | 449 | 44,922,236 | 7.299 | −0.00 | 108/120 | 49/80 | 21,523 | **24,576 (139%)** | no |
| B +share layernorm | 7010 | 44,666,235 | 447 | 44,666,236 | 7.300 | −0.00 | 107/120 | 45/80 | 20,116 | **22,197 (126%)** | no |
| C 4 context lanes | 7010 | 48,860,539 | 489 | 48,860,540 | 7.300 | −0.00 | 103/120 | 41/80 | 20,017 | **22,608 (128%)** | no |
| D 2 context lanes | 7010 | 57,347,451 | 573 | 57,347,452 | 7.300 | −0.00 | 103/120 | 39/80 | 20,250 | **22,287 (126.6%)** | no |
| **E +share requant/residual** | **7010** | **49,129,442** | **491** | 49,129,443 | **8.396** | **+1.10** | **107/120 (89%)** | **36/80 (45%)** | **11,905 (34%)** | **15,138/17,600 (86.0%)** | **YES** |

## Findings (measured)

1. **Operator sharing is the effective LUT lever:** v3 152.6% → A 139% → B 126% → **E 86% (fits)**.
2. **Context-lane reduction does NOT help LUT:** C(4 lanes)=128% and D(2 lanes)=126.6% are both ≥ B(8 lanes)=126%, and only add latency (C 489 ms, D 573 ms vs B 447 ms). The LUT is dominated by lane-independent nonlinear/normalization/requant/residual arithmetic — confirmed by the hierarchy (context muxing ≈ 200 LUT).
3. **The decisive change (E):** capping `requant` and `residual` to one shared instance each (`ALLOCATION limit=1`, made non-inlined) removed **~7,059 LUT and ~8,211 FF** vs B by time-sharing the duplicated 64-bit multiply/round datapaths.

## Selected fastest fitting candidate: **Variant E**

`hls/f1_int10/vitis/transformer_top_opt_z7010_e.cpp`
SHA-256 `04F688729EE271F50AC8E16BA2418CC70519AE2BAA0E2D4199E0E732C0F050D3`

- **Fits xc7z010clg400-1:** LUT 86.0%, BRAM 89.2%, DSP 45.0%, FF 33.8% — all ≤ capacity.
- Finite latency 49,129,442 cyc (0.491 s @ 10 ns), interval 49,129,443; est. clock 8.396 ns, **slack +1.10 ns** (timing met with margin).
- Bit-exact: Vitis CSim 16/16, native g++ A/B byte-identical.
- It is the ONLY fitting candidate (A/B/C/D/v3 all overflow LUT), hence the fastest fit.

### Trade-offs
- Latency penalty vs v3-on-7020: **+9.3%** (49.13 M vs 44.93 M cycles) — the cost of time-sharing the arithmetic to fit the smaller device. Speedup vs the original baseline: **1.51×**.
- New tightest resource on 7010 is **BRAM (89.2%)**, then LUT (86.0%). There is ~14% LUT headroom that could later be traded back for latency (e.g. `requant` limit=2) if desired.
- Real-time (48.83 windows/s) still **not** achieved (E ≈ 2.0 wps; gap ≈ 24×).

## Regressions (all pass)
Vitis CSim ap_int 16/16 (every variant); native g++ A/B byte-identical (every
variant); NumPy quantized oracle 16/16; `unittest` 30/30 OK; `git diff --check`
clean except the pre-existing user change in `src/train_transformer_baseline.py`
(untouched). Baseline `transformer_top.cpp` and v3 sources unchanged.

## Board readiness
**Not established.** These are HLS C-synthesis estimates for the core only. Fitting
at C-synth level is necessary but not sufficient: place-and-route, bitstream, DMA/PS
integration and on-board validation on an actually selected device remain required.

## Evidence
`artifacts/results/hls_z7010_opt_{a_shared_requant,b_shared_nonlinear,c_context_lanes4,d_context_lanes2,e_shared_arith}/`,
`artifacts/results/hls_z7010_opt_ctx_v3_bram_fixed/`, machine-readable
`artifacts/results/hls_z7010_fitting_campaign_summary.json`.
