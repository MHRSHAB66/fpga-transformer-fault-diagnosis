# B04 — single-pipeline checked projection and per-row checksum

Source: `hls/research/b04_integrity_fast/make_b04_top.py` → `transformer_top_seq256_raw1024_b04_argmax.cpp`
(from the B03 top, which is generated from B02, which is generated from the canonical raw-input
argmax top; no canonical file is modified). Builds: `-Variant b04` (no injection hardware, cost and
latency) and `-Variant b04i` (`-DB04_INJECT`, the B03 injection registers, detection measurements).

## Why B02/B03 cost +1.1 ms and B04 does not

Vitis HLS flattens the canonical projection nest `r (256) × o (32) × i (8)` into one pipeline with
II=1: 65,546 cycles. In B02/B03 the comparison `if (a > B || a < -B) err = 1;` after the inner MAC
loop stopped the flattening: the outer loop ran unpipelined at 21 cycles per output, 172,032 cycles
(HLS reports `artifacts/research/B02_20261113/b02_build/hls_reports/`). B04 writes the projection as
one loop over `t = 0 … 65,535` (`i = t % 8`, `o = (t / 8) % 32`, `r = t / 256`, all powers of two)
with `#pragma HLS PIPELINE II=1`; the range comparison, the requantization and the checksum update
happen in the iteration with `i == 7`. HLS: 65,545 cycles, II=1; whole core 1,479,894 cycles vs
1,477,436 canonical (+2,458 cycles ≈ 0.025 ms: the two input-signature passes, 2 × 2,048, and the
repeated classifier tail) vs 1,586,380 for B02.

## Arithmetic (all exact integers; values from `hls/f1_int10_seq256/vitis/weights_int10.h`)

| quantity | definition | proven bound | hardware type |
|---|---|---|---|
| token code `x[r][i]` | on-FPGA z-score code, clamped by `PP_QMAX`, then 4:1 rounded mean | \|x\| ≤ 511 | `ap_int<10>` |
| weight `W[o][i]` | INT10 export | \|W\| ≤ 511 | ROM (`long` constants) |
| accumulator `acc[r][o] = Σ_i x·W` | 8 products | \|acc\| ≤ 511·max_o Σ_i\|W\| = 718,977 < 2^20 | `ap_int<32>` |
| range check `acc + b[o]` | bias \|b\| ≤ 10,208 | \|acc+b\| ≤ 719,316 = `PROJ_ACC_BOUND` (bit2) | 64-bit compare |
| `WSUM[i] = Σ_o W[o][i]` | constants (generator) | [413, −85, 415, −551, −29, −560, −19, −1128] | ROM |
| `x · WSUM[i]` | second product | \|·\| ≤ 512·1128 = 577,536 | fabric multiplier (`BIND_OP impl=fabric`) |
| `chk[r] = Σ_i x·WSUM[i]` | 8 terms, computed while `o == 0` | \|chk\| ≤ 511·Σ\|WSUM\| = 1,635,200 | `ap_int<32>` |
| `rowsum[r] = Σ_o acc[r][o]` | 32 terms, after any site-2 injection | \|rowsum\| ≤ 511·Σ\|W\| = 10,627,778 < 2^24 | `ap_int<32>` |

In clean operation no intermediate exceeds 2^24, so the 32-bit registers never wrap and
`rowsum[r] == chk[r]` holds exactly by distributivity (Σ_o Σ_i x_i W_oi = Σ_i x_i Σ_o W_oi).
Requantization is untouched: `out = requant(acc + b[o], Mq)` with the canonical rounding; the
checksum only reads `acc`. Verified: C-model logits and class equal the canonical C-model on all
95,404 test windows with `integrity_status = 0` (`artifacts/research/B04_20261114/csim_b04_vs_canonical.json`).

## What bit3 (checksum) detects and what it does not

- Any change `e ≠ 0` of ONE accumulator of a row changes `rowsum` by `e`. In 32-bit hardware
  arithmetic the comparison fails unless `e ≡ 0 (mod 2^32)`; a single-bit flip changes the value by
  ±2^k, k ≤ 31, which is never 0 mod 2^32. So every single-bit accumulator flip is flagged; this
  includes the projection-accumulator bit 18–19 flips that stayed inside the proven range and were
  silent in B03.
- Several corrupted accumulators in one row are missed only if their errors cancel exactly.
- A corrupted input code `x` read by the projection changes `acc` and `chk` consistently and is NOT
  detected by bit3; that case is covered by the input-buffer signature (bit0).
- A weight-ROM upset changes `acc` but not `WSUM` and is detected; an upset in the checksum path
  itself gives a flag on a correct result (fail-safe, availability cost).
- The classifier tail is evaluated twice on ONE time-multiplexed unit (bit1): this detects a transient
  upset of one evaluation, not a persistent or common-mode fault of the shared unit or its weights.
- Scope: logical single-bit injection at three sites, transient (one window), on the research
  overlay. No radiation, configuration-memory or field upset-rate claim.
