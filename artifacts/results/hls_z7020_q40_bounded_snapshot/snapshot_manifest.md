# Frozen Z-7020 probe inputs

Captured from `hls/f1_int10/vitis/` and `hls/f1_int10/golden/` on
2026-09-19 before the C simulation and synthesis started. Both stages use
`run_verify_and_synth.tcl` in this directory, Vitis HLS 2020.2,
`xc7z020clg400-1`, and a 10 ns target clock.

| File | SHA-256 |
|---|---|
| `transformer_top.cpp` | `F0E76E0379BFF5D1E96D9DD78B76EDB1A9717CD88E1388CD49C834B005043FEA` |
| `ap_types.h` | `FDE052890B3E4511717027746EA2B4AEC744C410A3C372095F106A183B54FC76` |
| `weights_int10.h` | `65714EC3E45742CD2C456A64106F4F96BADD3F9A5DDC954E6A0D6A8C52D25863` |
| `testbench_top.cpp` | `93A562B3A8E2A6EEB1A5DDC16F17523E98F21864C86BCD6610C958862C490BD3` |
| `golden/golden_in.txt` | `5201C1A497976747B68F47B3D18D33CE56F65B260BC36DF600966303F04214EE` |
| `golden/golden_expected.txt` | `935F08F72B873DCDE0EE0E93EBDB1469209AD7D4C25F71E9A42C714A10B3FD28` |
| `run_verify_and_synth.tcl` | `A7FDD025E7AF76C461FFFB79D610174C35F06EDCDDE7F3851FAC4055E56D8350` |

This snapshot uses `SEQ=1024`, `CH=8`, `D=32`, one encoder layer,
four attention heads, and Q40 integer coefficients in the active arithmetic.
The generated header still contains unused `double` reference arrays.

## Completed results (reviewed 2026-09-21)

The preserved `transformer_top_csim.log` reports 16/16 exact golden vectors
and zero C-simulation errors. `transformer_top_csynth.rpt`, generated on
2026-09-19, reports 104 BRAM_18K, 40 DSP, 20,610 FF and 24,738 LUT.
Latency is 74,246,781 cycles (about 0.742 seconds at the 10 ns target);
the initiation interval is 74,246,782 cycles. Estimated clock period is
7.297 ns, not a measured hardware clock or inference latency.

These are provisional Z-7020 core estimates, not final board selection,
RTL equivalence, system fit, or a real-time demonstration. No synthesis
or C-simulation was rerun during the 2026-09-21 Git handoff; the seven
input hashes above were verified against the preserved files.
