# Weekly report 11 - verified evidence notes

This file is an engineering evidence log, not the final Persian report.

## Continuation from weekly report 10

Weekly report 10 ended with four pending items: finish the current Zynq-7020
synthesis, determine per-window latency, investigate further LUT reduction, and
then proceed to C/RTL co-simulation and board implementation.

## Newly completed result: current Zynq-7020 C synthesis

The previously pending synthesis completed for the frozen Q40/int10 source
snapshot, with `SEQ=1024`, 8 input channels, one encoder layer, and target part
`xc7z020-clg400-1` at a 10 ns target clock.

| Metric | Current core estimate | Device capacity | Exact use |
|---|---:|---:|---:|
| BRAM_18K | 104 | 280 | 37.14% |
| LUT | 24,738 | 53,200 | 46.50% |
| FF | 20,610 | 106,400 | 19.37% |
| DSP | 40 | 220 | 18.18% |

The core estimate is below the device capacity for all four reported resources.
This is not yet proof that the complete system fits: DMA, the processing system,
implementation overhead, and timing closure are not included.

The matching C simulation reproduced all four integer logits for all 16 golden
vectors exactly (`16/16`, zero errors).

## Progress relative to the intermediate Zynq-7020 version

The intermediate version reported in weekly report 10 used 242 BRAM_18K,
43,381 LUT, 18,023 FF, and 71 DSP. The current row-wise Q40/int10 version uses
104 BRAM_18K, 24,738 LUT, 20,610 FF, and 40 DSP. This corresponds to reductions
of 57.02% in BRAM, 42.97% in LUT, and 43.66% in DSP, with a 14.35% increase in
FF. Because multiple implementation changes separate the two source versions,
these figures describe overall design progression rather than the effect of one
isolated optimization.

## Newly measured limitation: latency

The synthesis report estimates 74,246,781 cycles per window. At the 10 ns target
clock this is about 0.742 s per window, or 1.35 windows/s. A non-overlapping
1024-sample window is acquired in 20.48 ms at 50 kHz, requiring 48.83 windows/s.
The current core is therefore about 36.25 times slower than the continuous input
rate and cannot yet be described as real-time for consecutive windows.

## Latency bottleneck analysis

An architecture-level operation count was calculated separately from the HLS
latency measurement. For `SEQ=1024`, four heads, and head dimension 8, computing
the attention scores requires 33,554,432 multiply-accumulate operations, and
forming the attention context requires another 33,554,432. Together these two
quadratic attention operations account for 67,108,864 of the 75,760,192 listed
matrix-multiply operations, or 88.58%. This is an analytical workload share,
not a per-function latency reported by HLS, but it identifies full-sequence
attention as the first optimization target.

Meeting the continuous non-overlapping input rate requires at least a 36.25x
end-to-end throughput improvement over the current baseline. A small local
optimization cannot be reported as sufficient until a new full-size synthesis
measures the resulting interval and latency.

## Evidence paths

- Synthesis report: `artifacts/results/hls_z7020_q40_bounded_snapshot/transformer_top_csynth.rpt`
- C-simulation log: `artifacts/results/hls_z7020_q40_bounded_snapshot/transformer_top_csim.log`
- Frozen source hashes: `artifacts/results/hls_z7020_q40_bounded_snapshot/snapshot_manifest.md`
- Machine-readable summary: `artifacts/results/hls_z7020_current_summary.json`

## Next measured work

1. Attribute the 74.25-million-cycle latency to attention and post-attention
   operations using bounded component-level HLS probes.
2. Select one parallelization factor that stays within the remaining Zynq-7020
   DSP/LUT budget, then confirm bit-exact C simulation.
3. Synthesize the selected version and compare resource use and latency against
   this frozen baseline.
4. Attempt C/RTL co-simulation only after the optimized synthesis report is
   available; bitstream and board claims remain pending.
