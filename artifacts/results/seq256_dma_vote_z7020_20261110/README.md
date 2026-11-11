# Raw-input core + temporal vote, XC7Z020 build (2026-11-11)

Source: commit b31d485 (`temporal_vote.h` with 2-bit classes, 4-bit counts and a
select-chain pick), top `transformer_top_seq256_raw1024_dsp_margin_vote.cpp`,
built with `powershell -File tools/build_raw_core.ps1 -Variant vote`.

| | raw-input core (2026-11-09) | + temporal vote (this build) |
|---|---|---|
| HLS C-simulation, golden_raw1024_argmax | 41/41 | 41/41 logits + class, 41/41 votes = Python (k = 5) |
| Post-route Slice LUTs | 36,670 (68.9 %) | 36,911 (69.4 %) |
| Post-route Slice registers | 36,163 | 36,349 |
| DSP / BRAM tiles | 210 / 76.5 | 210 / 76.5 |
| WNS / WHS at 100 MHz | +0.167 / ≥ 0 ns | +0.135 / +0.007 ns |

The vote costs +241 LUT and +186 FF after routing. Two earlier builds of the
vote were stopped (`first_build_int32_aborted/`, `second_build_flagpick_aborted/`).

AXI-Lite registers added to the raw core (`bitstream/design_1.hwh`): `vote_k`
(input, clamped to 1..15), `vote_class` and `vote_count` (outputs with valid bits).
Deployment bundle: `artifacts/deploy/seq256_vote_z7020_20261110/`. BIT and XSA
are gitignored and recorded in `artifacts/manifests/artifact_manifest.json`.
Board validation (2026-11-11, MYIR Z-Turn): golden 41/41 (class from the FPGA register);
full test 95,404/95,404 logits = C-model, accuracy 97.7705 %, FPGA vote (k = 5)
95,404/95,404 = Python, system accuracy 99.084 %; hard real time 0 misses in 30,000
(max 16.66 ms). Evidence: `../raw_core_board_{golden_20261111_130854,fulltest_20261111_130927,rt_20261111_133749}/`.
