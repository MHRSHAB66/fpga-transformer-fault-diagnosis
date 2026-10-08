# SEQ=256 INT10 HLS results

## Qualification and evidence layers

This report distinguishes four independent evidence layers. They must not be
conflated:

1. **Measured software accuracy** — full-test INT10 integer-ingress evaluation.
2. **Measured CSim correctness** — Vitis HLS 2020.2 CSim, bit-exact vs golden.
3. **HLS C-synthesis estimates** — operator-count estimates, *not* place-and-route.
4. **Vivado post-route implementation** — the integrated PS + AXI-DMA + IP system.

**Physical-board validation is NOT done.** No board has been programmed; no live
DMA transfer, UART link, or end-to-end hardware run has been measured. The target
part `xc7z020clg400-1` is a **provisional / generic** configuration selected for
resource and timing evidence — it is not a confirmed board.

## Approved acquisition → aggregation → inference pipeline

The software path starts from the complete **1024-sample, 8-channel** acquisition
window (50 kHz, 20.48 ms span). The exact numeric contract is:

```
normalize each [1024,8] window
  → quantize to signed INT10 (code = round(x / input_scale), clamp to ±511)
  → exact integer 4:1 aggregation  → [256,8]
  → 256-token Transformer inference (1 layer, d_model=32, 4 heads, ffn=64)
  → 4 integer logits
  → argmax class
```

- `input_scale = 0.028626115354773118`, INT10 (`ACT_QMAX = 511`).
- Aggregation is signed round-half-away-from-zero mean, factor 4:
  `q = (|sum| + 2) >> 2` with the sign reapplied, then clamped to INT10. This is
  bit-exact between the NumPy reference and the HLS block.
- In the **integrated design the hardware IP performs the 4:1 aggregation
  internally**: the AXI-Stream ingress receives all `1024 × 8 = 8192` INT10 codes
  (row-major `[sample, channel]`, one code per 32-bit word, TLAST on word 8191),
  aggregates on-chip to `[256,8]`, and emits 4 INT32 logits (TLAST on logit 3).
- Class map is unchanged: `0 normal, 1 imbalance, 2 misalignment, 3 bearing_fault`.

## 1. Measured software validation

- INT10 full-test accuracy: **0.9773594399** (93,244 / 95,404 windows)
- INT10 full-test macro-F1: **0.9562824349**
- Per-class F1: 0.8953, 0.9690, 0.9769, 0.9840
- Reference (float aggregate-then-quantize): accuracy 0.9765628276, macro-F1
  0.9559999458 — the exact hardware ordering is +0.0797 pp accuracy / +0.0282 pp
  macro-F1, i.e. no degradation.
- Evidence: `artifacts/experiments/transformer_f1_seq256_pool4/int10_ingress_local/combined.json`

## 2. Measured CSim correctness

- Vitis HLS 2020.2 CSim (GCC) using `ap_int` types: **16/16 golden vectors exact**
  (bit-exact logits *and* class), TLAST behaviour verified.
- Evidence: `.../seq256_dma_system_dsp_margin_z7020/hls_csim/transformer_top_csim.log`

## 3. HLS C-synthesis estimates (not post-route)

Target clock 10 ns (100 MHz), estimated clock 7.299 ns, for `xc7z020-clg400-1`.

| Quantity | HLS estimate |
|---|---:|
| Latency | 1,490,526 cycles |
| Interval | 1,490,527 cycles |
| Time at 100 MHz | 14.90527 ms |
| BRAM_18K | 145 |
| DSP | 156 |
| FF | 46,948 |
| LUT | 52,864 |

These are operator-count estimates only. Vivado place-and-route (Section 4) is
authoritative for real resource usage; note DSP rises from the HLS estimate of
156 to 210 after synthesis because wide multipliers expand into extra DSP48
primitives.

## 4. Vivado post-route — integrated PS + DMA + Transformer system

The generated HLS IP was packaged and integrated into a complete block design:
**Zynq PS7 + AXI-DMA (MM2S + S2MM, 32-bit) + AXI interconnect (HP0) +
`transformer_top` IP**, at a 100 MHz FCLK. The design was synthesized, placed,
routed, and a bitstream written **in context** in Vivado 2020.2 for the
provisional `xc7z020clg400-1`.

### Post-route timing (authoritative)

| Metric | Value |
|---|---:|
| Clock target | 10 ns (100 MHz) |
| WNS (setup) | **+0.036 ns** |
| TNS | 0 ns |
| WHS (hold) | **+0.020 ns** |
| THS | 0 ns |
| WPWS (pulse width) | +3.750 ns |
| Result | **All user-specified timing constraints are met** |

### Post-route resources (authoritative)

| Resource | Used | Available | % |
|---|---:|---:|---:|
| Slice LUTs | 31,687 | 53,200 | 59.56 |
| Slice Registers | 30,889 | 106,400 | 29.03 |
| Block RAM Tile | 70.5 | 140 | 50.36 |
| DSP48E1 | 210 | 220 | 95.45 |

### Routing / DRC

- Route status: **66,275 / 66,275 routable nets fully routed, 0 routing errors.**
- DRC: 196 violations, **0 critical, 0 error** — all Warning/Advisory (DSP
  input/output pipelining performance advisories, one gated-clock check, one
  no-routable-loads, two write-first advisories). Bitstream generated
  successfully.

### Real-time comparison

Compute time **14.90527 ms** (HLS scheduled interval at 100 MHz, with post-route
timing closed) versus the acquisition-window period **20.48 ms**. The
**computational** real-time condition (compute < acquisition period) is met at the
target clock, leaving ≈ 5.57 ms margin (≈ 67.1 windows/s vs the 48.83 windows/s
required). This is a synthesis/timing result, not a live on-board throughput.

## 5. DSP-margin build vs the previous 220-DSP version

The `transformer_top` IP is synthesized out-of-context with
`synth_design ... -max_dsp 210`, deliberately reserving **ten DSP48E1** sites at
the implemented-system level. This makes the margin real (Vivado can expand wide
multipliers into extra DSP48 primitives) rather than an HLS estimate.

| Metric | 220-DSP build | 210-DSP margin build | Δ |
|---|---:|---:|---:|
| DSP48E1 | 220 (100.00%) | 210 (95.45%) | −10 |
| Slice LUTs | 30,892 | 31,687 | +795 |
| Slice Registers | 30,603 | 30,889 | +286 |
| Block RAM Tile | 70.5 | 70.5 | 0 |
| WNS (ns) | +0.280 | +0.036 | −0.244 |
| WHS (ns) | +0.017 | +0.020 | +0.003 |
| Latency (cycles) | 1,489,004 | 1,490,526 | **+1,522** |
| Interval (cycles) | 1,489,005 | 1,490,527 | **+1,522** |
| Fully routed nets | 65,365 | 66,275 | — |
| Routing errors | 0 | 0 | 0 |

**Cost of the margin:** the ten freed DSPs become LUT logic, costing +795 LUT,
+286 FF, and reducing setup slack from +0.280 ns to +0.036 ns (still positive;
timing met). The DSP-margin build also uses a revised HLS core (Q40 arithmetic +
token-buffer reuse), so its scheduled latency is **+1,522 cycles** vs the earlier
220-DSP integrated build — **+0.01522 ms at 100 MHz (~+0.1022%)**. This is
negligible but **measured, not zero**, and still far below the 20.48 ms
acquisition period. The prior build lives at
`artifacts/results/seq256_dma_system_z7020/`.

## Artifacts and hashes

Permanent package: `artifacts/results/seq256_dma_system_dsp_margin_z7020/`
(see its `MANIFEST.sha256` for every file hash; `summary.json` for machine-readable numbers).

| File | SHA-256 (abbrev.) |
|---|---|
| `bitstream/design_1_wrapper.bit` | `21c7ae29…420b` |
| `bitstream/design_1.hwh` (same build as .bit) | `449b09bf…e153` |
| `bitstream/seq256_dma_system.xsa` | `18499b87…f9d` |
| `hls_src/transformer_top_seq256_ingress1024_dsp_margin.cpp` | `25b91d37…2b9e` |
| `tcl/run_seq256_dma_system_dsp_margin.tcl` | `1ec7f73d…ec27` |

Block design (read from the generated HWH): DMA `axi_dma_0` (AXI-Lite
`0x41E00000`), IP `transformer_top_0` (AXI-Lite ctrl `0x40000000`), MM2S/S2MM
tdata 32-bit, 8192 input words, 4 output words per window.

Historical evidence (earlier OOC IP-only and C-synthesis runs) is retained in
`artifacts/results/hls_seq256_int10_summary.json` under the `synthesis` key and in
`artifacts/results/hls_seq256_int10_ingress1024_z7020*/`.

## Limitations

- Target part is a **provisional/generic XC7Z020** configuration, not a selected board.
- Bitstream generation and post-route timing are **complete**; **physical-board
  loading, real DMA transfer, UART communication, and live end-to-end validation
  remain unmeasured.**
- HLS C-synthesis numbers are estimates; only the Vivado post-route numbers are
  place-and-route measurements. C-synthesis estimates are not called "post-route".
- The integer hardware path outputs 4 logits and an argmax class only; a
  confidence/probability value is **not** produced in hardware (reported as NA).
