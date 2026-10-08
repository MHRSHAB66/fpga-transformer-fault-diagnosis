# SEQ=256 INT10 Transformer + AXI-DMA integration for MYIR Z-Turn

Permanent, versioned evidence package for the verified SEQ=256 INT10 Transformer
system integrated with a Zynq PS + AXI-DMA data path. This directory is a
self-contained snapshot copied out of `tmp/` so the result survives Vivado
project cleanup. Nothing here overwrites earlier evidence.

> **Selected target.** This build targets the MYIR Z-Turn `MYS-7Z020-C-S`,
> populated with `XC7Z020-1CLG400C`; the Vivado part is
> `xc7z020clg400-1`. Physical validation of this rebuilt bitstream is pending.

## Contents

| Path | What it is | Tool / stage |
|---|---|---|
| `hls_src/transformer_top_seq256_ingress1024_dsp_margin.cpp` | Bit-exact HLS C++ top (1024→256 ingress + core) | source |
| `hls_src/ap_types.h`, `hls_src/weights_int10.h` | Fixed-point typedefs + exported INT10 weights | source |
| `hls_src/testbench_ingress1024.cpp` | CSim testbench (golden compare) | source |
| `hls_reports/transformer_top_csynth.rpt` | HLS C-synthesis report (**estimates**) | Vitis HLS 2020.2 |
| `hls_csim/transformer_top_csim.log` | CSim log: **16/16 exact** | Vitis HLS 2020.2 CSim |
| `hls_csim/golden_in.txt`, `golden_expected.txt` | The 16 golden vectors compared | Python reference |
| `tcl/run_seq256_ingress1024_dsp_margin_verify_and_synth.tcl` | HLS CSim + C-synthesis + export driver | Vitis HLS |
| `tcl/export_seq256_ingress1024_dsp_margin.tcl` | HLS IP export | Vitis HLS |
| `tcl/run_seq256_dma_system_dsp_margin.tcl` | Vivado block-design build (`-max_dsp 210`) | Vivado 2020.2 |
| `vivado_reports/post_synth_*.rpt` | Post-synthesis util/timing | Vivado 2020.2 |
| `vivado_reports/post_route_timing_summary.rpt` | **Post-route** timing (authoritative) | Vivado 2020.2 |
| `vivado_reports/post_route_utilization.rpt` | **Post-route** resources (authoritative) | Vivado 2020.2 |
| `vivado_reports/post_route_status.rpt` | Route status (fully routed, 0 errors) | Vivado 2020.2 |
| `vivado_reports/post_route_drc.rpt` | DRC (196 warnings/advisories, 0 critical/error) | Vivado 2020.2 |
| `bitstream/design_1_wrapper.bit` | Bitstream | Vivado impl |
| `bitstream/design_1.hwh` | Hardware handoff (**same build** as .bit) | Vivado impl |
| `bitstream/seq256_dma_system.xsa` | Exported hardware platform | Vivado impl |
| `software_evidence/fulltest_int10_ingress_combined.json` | Full-test accuracy/macro-F1 | Python int10 reference |
| `software_evidence/model_manifest.json` | Model + quantization manifest | Stage 6/7 export |
| `software_evidence/integer_config_scales.json` | Calibrated INT10 activation scales | Stage 6 export |
| `summary.json` | Machine-readable summary of all numbers | this package |
| `MANIFEST.sha256` | SHA-256 of every file in this package | this package |

## Evidence layers (do not conflate)

1. **Measured software accuracy** — 95,404 test windows, 93,244 correct,
   accuracy = 0.9773594399, macro-F1 = 0.9562824349. Exact-integer ingress order.
2. **Measured CSim correctness** — Vitis HLS 2020.2 CSim, 16/16 golden vectors
   exact (bit-exact logits + class).
3. **HLS estimates** — C-synthesis operator estimates only (latency 1,490,526
   cycles / interval 1,490,527, 14.90527 ms at 100 MHz; est. DSP 156, LUT 52,864).
   **These are not place-and-route numbers.**
4. **Vivado post-route** — implemented PS+DMA+IP system on the selected part:
   WNS +0.364 ns, TNS 0, WHS +0.018 ns, THS 0 (all met); LUT 31,734/53,200
   (59.65%), FF 30,921/106,400 (29.06%), BRAM 70.5/140 (50.36%), DSP 210/220
   (95.45%); 66,347/66,347 nets fully routed, 0 routing errors. AXI-DMA remains
   in simple mode with `C_SG_LENGTH_WIDTH=16`, so one 32,768-byte input frame
   fits in a single transfer and TLAST remains on word 8191.
5. **Physical-board validation — NOT DONE.** No board programmed, no live DMA,
   no UART, no end-to-end hardware test.

## DSP margin

`transformer_top` OOC synthesis is constrained with `synth_design ... -max_dsp
210`, reserving ten DSP48E1 sites. Versus the earlier 220/220-DSP build
(`artifacts/results/seq256_dma_system_z7020/`):

| Metric | 220-DSP build | 210-DSP margin build | Δ |
|---|---:|---:|---:|
| DSP48E1 | 220 (100.00%) | 210 (95.45%) | −10 |
| Slice LUTs | 30,892 | 31,734 | +842 |
| Slice Registers | 30,603 | 30,921 | +318 |
| Block RAM Tile | 70.5 | 70.5 | 0 |
| WNS (ns) | +0.280 | +0.364 | +0.084 |
| WHS (ns) | +0.017 | +0.020 | +0.003 |
| Latency (cycles) | 1,489,004 | 1,490,526 | **+1,522** |
| Interval (cycles) | 1,489,005 | 1,490,527 | **+1,522** |
| Routing errors | 0 | 0 | 0 |

The ten freed DSPs are re-implemented as LUT logic. The cost is +795 LUT, +286 FF
and reduced setup slack (still positive; timing met). The DSP-margin build uses a
revised HLS core (Q40 arithmetic + token-buffer reuse), so its scheduled latency
is **+1,522 cycles** vs the earlier 220-DSP integrated build (**+0.01522 ms at
100 MHz, ~+0.1022%**) — negligible but **measured, not zero**, still well below
20.48 ms.

## Real-time note

Compute time 14.90527 ms < acquisition-window period 20.48 ms, so the
**computational** real-time condition is met at the 100 MHz target clock. This is
based on the HLS scheduled interval and Vivado timing closure; it is not a live
on-board throughput measurement.

## Reproduce

1. HLS: `vitis_hls -f tcl/run_seq256_ingress1024_dsp_margin_verify_and_synth.tcl`
   (runs CSim + C-synth + IP export from `hls_src/`).
2. Vivado: `vivado -mode batch -source tcl/run_seq256_dma_system_dsp_margin.tcl`
   (builds the PS+DMA+IP system, implements, writes `.bit`/`.xsa`).

The Tcl paths assume the repository layout; the copies here are the exact scripts
used for this build.
