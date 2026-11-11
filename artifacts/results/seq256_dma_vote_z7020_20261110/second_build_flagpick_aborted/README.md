# Second vote build (narrow types, early-exit pick), stopped 2026-11-11 11:04

Source: commit 7ba2fee. The pick loop scanned newest -> oldest with a `found`
flag (`if (!found && valid[i] && is_best[hist[i]]) { winner = ...; found = true; }`).
HLS turned it into control flow: the top-level FSM grew from 10 (raw core) to 26
states with phi multiplexers, and Vivado synthesized the transformer IP to
51,059 LUT (96 % of the device) against 34,987 for the raw core.

Hierarchical utilization of the two IP netlists (out-of-context synthesis):
the transformer core and the preprocessing are identical in both; the whole
difference sits in the top module itself, 587 -> 16,563 LUT
(`ip_hierarchical_utilization.rpt`, `reference_raw_ip_hierarchical_utilization.rpt`).
`top_level_lut_groups.txt` groups the top-level LUT cells by name: ~7.6 k in the
FSM next-state logic and ~8.6 k in the `winner` selection.

Fix (next commit): scan oldest -> newest and overwrite with a select, so the
newest match is the last one written; `seen` and `best` updates also as selects.
Same results (C++ native and ap_int builds = Python on 5,000 random steps; HLS
C-simulation 41/41 logits + class, 41/41 votes). HLS: 15 FSM states, mux 225 LUT.
Out-of-context IP synthesis: 35,216 LUT / 32,719 FF / 210 DSP, i.e. the vote
costs +229 LUT and +186 FF over the raw-input core.
