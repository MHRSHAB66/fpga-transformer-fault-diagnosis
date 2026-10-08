# v3 (BRAM-fixed) — measured Zynq-7020 vs Zynq-7010

Same source, model, int10 formats, golden vectors, AXI interface, SEQ=1024, and
10 ns clock. Only the target part differs. All numbers are **Vitis HLS 2020.2
C-synthesis estimates** (not post-implementation, not board results).

Source: `hls/f1_int10/vitis/transformer_top_opt_bram.cpp`
SHA-256 `21AE87EEEA656E499E7CF7B35938B30652AC88FEFF1B1066373EEDE51622887B`
(identical for both runs; matches the preserved 7020 v3 snapshot).

Both are **provisional experimental targets** — the final board is not selected.

## Measured comparison

| Metric | Zynq-7020 (xc7z020clg400-1) | Zynq-7010 (xc7z010clg400-1) |
|---|---:|---:|
| CSim (ap_int, 16 golden) | 16/16 exact | 16/16 exact |
| Latency (cycles) | 44,930,707 | 44,930,707 |
| Latency @ 10 ns | 449.31 ms | 449.31 ms |
| Interval (cycles) | 44,930,708 | 44,930,708 |
| Estimated clock | 7.300 ns | 7.300 ns |
| Reported slack | 0.00 ns | 0.00 ns |
| BRAM_18K used / avail | 108 / 280 (38.6%) | 108 / 120 (**90.0%**) |
| DSP used / avail | 46 / 220 (20.9%) | 46 / 80 (57.5%) |
| FF used / avail | 23,736 / 106,400 (22.3%) | 23,736 / 35,200 (67.4%) |
| LUT used / avail | 26,857 / 53,200 (50.5%) | 26,857 / 17,600 (**152.6%**) |
| **Fit (C-synth estimate)** | **YES** | **NO — LUT overflow** |

The design's absolute resource usage is part-independent; only the device
*capacities* differ. On 7010 the LUT budget (17,600) is less than half of 7020's
(53,200), so the same 26,857 LUT that fit comfortably on 7020 overflow 7010.

## Verdict

**The current v3 core does not fit Zynq-7010.**
- First limiting resource: **LUT = 26,857 / 17,600 = 152.6%** (over by **+9,257**, +52.6%).
- Second tightest: BRAM_18K at 90.0% (108/120 — fits, little margin).
- DSP (57.5%) and FF (67.4%) fit.

## Measured LUT diagnosis (7010 report hierarchy)

- **8 parallel `requant` units = 4,440 LUT (+4,832 FF)** — a side-effect of the
  HD=8-unrolled context *output* loop `for d: ctx[off+d]=requant(cacc[d],…)`.
- `finish_token` module = 6,975 LUT; classifier layernorm = 1,581 LUT; the rest is
  the softmax integer divider, per-datapath requant, `linear_fixed`, and control.

**Smallest technically justified next step (NOT applied here):** roll the context
output requant loop (drop its `#pragma HLS UNROLL`) so a single `requant` unit is
shared → ~3.9k LUT saved at ~8 extra cycles/head/token (negligible; bit-exact).
This alone does not close the 9.3k-LUT gap, so further LUT sharing (reuse `requant`
across score/context/linear datapaths; a bounded/shared softmax reciprocal) would
also be needed. Any such change must be re-synthesized and re-measured — do not
assume it fits.

## Evidence levels (kept distinct)

- **Functional correctness:** verified — Vitis CSim 16/16 (ap_int), native g++ A/B
  byte-identical, NumPy oracle 16/16, unittest 30/30.
- **C-synthesis resource fit:** measured — fits 7020, does NOT fit 7010 (LUT).
- **Post-implementation timing:** not run (no place-and-route).
- **Board readiness:** not established — requires implementation/bitstream and
  on-hardware validation on an actually selected device.

Evidence: `artifacts/results/hls_z7010_opt_ctx_v3_bram_fixed/` (reports, source,
Tcl, SHA-256, JSON) and `artifacts/results/hls_z7020_opt_ctx_v3_bram_fixed/`.
