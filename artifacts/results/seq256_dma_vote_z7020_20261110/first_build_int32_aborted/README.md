# First vote build (int32 vote logic), aborted 2026-11-10 23:08

Source: commit 40cdee7 (`temporal_vote.h` with `int` for the class history,
counts and window counter).

| | raw-input core (deployed) | + vote, int32 logic |
|---|---|---|
| HLS estimate, transformer_top LUT / FF | 63,382 / 57,092 | 66,757 / 59,463 |
| Vivado post-synthesis Slice LUTs | 38,159 (71.7 %) | 45,453 (85.4 %) |
| Post-placement WNS | +0.168 ns | +0.129 ns |
| Router, after global iteration 1 | WNS -0.230 ns | WNS -1.442 ns, TNS -97.1 ns ("Congestion is preventing the router from routing all nets") |

The 32-bit vote logic cost about 7.3 k LUT after synthesis and congested the
routing on the 85 %-full device, so the run was stopped during routing (no
bitstream). The vote was rewritten with 2-bit classes and 4-bit counts (same
results: C++ native and ap_int builds equal host/temporal_vote.py on 5,000
random steps; HLS C-simulation on the golden vectors) and rebuilt.

Files: HLS synthesis report, Vivado post-synthesis utilization/timing, and the
implementation log up to the abort (`impl_runme_until_abort.log`, local only:
`*.log` is gitignored). Reference numbers of the raw-input core:
`../../seq256_dma_raw_argmax_z7020_20261109/vivado_reports/`.
