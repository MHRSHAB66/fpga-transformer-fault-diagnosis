# Hardware versions and evidence index

**Documentation index — reconciled 2026-11-20.** This is a navigation aid, not a
new hardware result. For each number, consult the linked original report and
its machine-readable evidence before citing it. Historical HLS campaigns must
not be presented as physical-board benchmarks.

## Chronological implementation decisions

| Implementation / experiment | Target and evidence level | Finding | Source |
|---|---|---|---|
| SEQ=1024, v3 | XC7Z010, Vitis HLS C-synthesis | 26,857 LUT / 17,600 available (152.6%); 449 ms estimated interval. **Does not fit HLS LUT budget.** | [`hls_z7010_fitting_campaign.md`](hls_z7010_fitting_campaign.md) |
| SEQ=1024, shared arithmetic (E) | XC7Z010, Vitis HLS C-synthesis | 15,138 LUT (86.0%, fits **estimated** resource budget), but 491 ms interval; **not real time** | [`hls_z7010_fitting_campaign.md`](hls_z7010_fitting_campaign.md) |
| SEQ=1024, fused K+V prefix (prefix_opt2) | XC7Z020, Vitis HLS C-synthesis | 49,889 LUT (about 94%), 62.30 ms interval, still above 20.48 ms budget; 16/16 CSim | [`hls_realtime_continuation.md`](hls_realtime_continuation.md) |
| SEQ=256, INT10 integrated PS + AXI-DMA | XC7Z020, Vitis HLS estimate | Interval 1,490,527 cycles = 14.905 ms at 100 MHz, not measured on the board | [`summary.json`](../../artifacts/results/seq256_dma_system_dsp_margin_dma16_z7020_20261008/summary.json) |
| Same integrated build | XC7Z020, Vivado **post-route** | 31,734 LUT / 210 DSP; WNS +0.364 ns, meets routed timing at 100 MHz | [`post_route_utilization.rpt`](../../artifacts/results/seq256_dma_system_dsp_margin_dma16_z7020_20261008/vivado_reports/post_route_utilization.rpt), [`summary.json`](../../artifacts/results/seq256_dma_system_dsp_margin_dma16_z7020_20261008/summary.json) |
| Baseline deployed INT10 on board | XC7Z020, **physical** | 97.7684% (93,275 / 95,404) test accuracy; all four logits match software INT10 reference on every window | [`consolidated_results_20261022.md`](consolidated_results_20261022.md) sections 1–2 |
| Board-only prepared-input throughput (pipelined host) | XC7Z020, **physical**, average throughput | 60.938 windows/s over 95,404; **723** loop iterations exceeded 20.48 ms. Throughput is not a no-miss guarantee | [`realtime_verdict.json`](../../artifacts/results/fpga_realtime_bench_20261022/realtime_verdict.json) |
| Board-boundary hard-real-time acceptance | XC7Z020, **physical**, RAM inputs + SCHED_FIFO | 0 deadline misses over 100,000 windows; maximum 16.56 ms | [`hard_rt_calibration_power_20261024.md`](hard_rt_calibration_power_20261024.md) |
| Raw-input preprocessing inside FPGA | XC7Z020, Vivado + **physical** | Raw 1024×8 ingress, on-FPGA z-score and INT10, 97.7705% test accuracy; post-route 36,670 LUT / 210 DSP | [`onchip_preprocessing_20261109.md`](onchip_preprocessing_20261109.md) |
| Raw-input + temporal vote, deployed-model weights | XC7Z020, Vivado + **physical**, continuous stream | k=5 system accuracy 99.084%, compared with per-window model accuracy 97.7705%; this is **not** a change to core model accuracy | [`paper_readiness_review_20261110.md`](paper_readiness_review_20261110.md) section 6 |
| Noise-augmented weights, **separate** bitstream | XC7Z020, **physical**, 0-dB noisy windows | 90.1304% per-window; 93.835% after temporal vote k=5; 95,404/95,404 logits match C-model | [`paper_readiness_review_20261110.md`](paper_readiness_review_20261110.md) section 5 |
| Vote-overlay endurance | XC7Z020, **physical**, RAM input + SCHED_FIFO | 1,400,000 windows with 0 misses (max 19.25 ms); does **not** prove sensor-to-PC hard-real-time operation | [`paper_readiness_review_20261110.md`](paper_readiness_review_20261110.md) section 7 |
| Ten-class Transformer | PyTorch float **software only** | 96.92% ten-class accuracy; **no ten-class FPGA deployment demonstrated** | [`paper_readiness_review_20261110.md`](paper_readiness_review_20261110.md) section 6b |

## Interpreting and citing numbers

- **C Simulation / golden vectors** checks functional agreement; it does not
  establish resource fit, routed timing, or board execution.
- **HLS C-synthesis** reports *estimates*, including resource counts and
  scheduled latency/interval. An HLS estimate such as 14.905 ms must not be
  reported as a physical-board measurement.
- **Vivado post-route** reports placed/routed resource utilization and timing
  closure for that exact build. Its numbers can differ from an HLS estimate.
- **Physical board** measurements are strongest for actual behavior, but still
  require test conditions: input source, sample count, scheduling, and whether
  the metric is average throughput, a per-window deadline, or system accuracy.
- The 20.48 ms arrival period assumes 50,000 raw samples/s and 1,024-sample
  non-overlapping windows. Aggregating raw samples into 256 tokens does **not**
  change that period.

## Documentation boundaries

- Current overview: [`../../README.md`](../../README.md).
- Current stage status (including what is still pending): [`../project_status.md`](../project_status.md).
- Primary deployed baseline results: [`consolidated_results_20261022.md`](consolidated_results_20261022.md).
- Later paper/robustness/generalization tests: [`paper_readiness_review_20261110.md`](paper_readiness_review_20261110.md).
- Demo instructions: [`../demo_runbook.md`](../demo_runbook.md).

Retain historical HLS experiment reports for reproducibility even when their
"board not tested" statements reflect an earlier point in time. Do not delete
raw evidence or build reports as part of documentation cleanup.
