# HLS experiment log — 2026-09-19

These are provisional Z-7010 (`xc7z010clg400-1`) Vitis HLS 2020.2 probes
at a 10 ns target, not a final-board selection or post-implementation result.
The maintained model is F_1layer @ int10, `SEQ=1024`, eight input channels,
four output classes. All C-simulation checks below compare all four logits of
each of the 16 existing golden vectors exactly.

| Experiment | Result | Evidence |
|---|---|---|
| Full-length Q40 top, explicit 32-bit softmax numerator | Csim 16/16; full synthesis stopped after 20 minutes with no report | `tmp/hls_z7010_div32/` |
| SEQ=64 Q40 diagnostic (same arithmetic, shorter sequence) | 43 BRAM_18K, 36 DSP, 18,994 FF, 21,943 LUT | `artifacts/results/hls_z7010_seq64_q40_csynth.rpt` |
| One full-length attention query in isolation | Completed in about 32 s; 66 BRAM_18K, 40 DSP, 8,634 FF, 10,563 LUT; **not** a full-core estimate | `artifacts/results/hls_z7010_seq1024_attention_probe_csynth.rpt` |
| Full-length top with attention zeroed (diagnostic only) | No report after about 6 minutes; stopped. Slow synthesis is not attributable solely to attention | `tmp/diag_no_attention_1024/` |
| Q40 top with per-token post-attention work isolated in `finish_token` | Csim 16/16; full synthesis completed in 4,891 s: 104 BRAM_18K, 40 DSP, 21,072 FF, 24,518 LUT | `artifacts/results/hls_z7010_seq1024_finish_token_csim.log`, `artifacts/results/hls_z7010_seq1024_finish_token_csynth.rpt` |
| Bounded `rsqrt_fxp` loop (maintained source after the synthesis snapshot) | Csim 16/16; full synthesis/resource effect not yet measured | `artifacts/results/hls_seq1024_bounded_rsqrt_csim.log` |
| Experimental Q32 coefficient export, no architecture change | Csim 16/16; SEQ=64 diagnostic: 40 BRAM_18K, 33 DSP, 14,613 FF, 20,991 LUT. Not yet full-length synthesized or evaluated on the full test set | `artifacts/results/hls_seq1024_q32_csim.log`, `artifacts/results/hls_z7010_seq64_q32_csynth.rpt` |

Z-7010 capacity in the full-length HLS report: 120 BRAM_18K, 80 DSP,
35,200 FF, 17,600 LUT. The completed full-length Q40 estimate is **139% LUT**,
so the core does not currently fit Z-7010. HLS reports top-level latency
and interval as `?`; the 7.297 ns estimated clock period is not an inference
latency measurement. No C/RTL cosimulation, place-and-route, bitstream, or
board test is claimed.

Next steps: reduce LUT in a function-level measured manner (the current
`finish_token` and `attention0_row` subreports are useful starting points),
re-run full-length synthesis after each change, bound/measure latency, and
then validate RTL and board behavior. The Q32 path stays experimental until
more than the 16 golden vectors and a full-length resource report support it.
