# Long-run hard real-time criterion (fixed BEFORE run L1, 2026-10-24)

Identical to `hard_rt_acceptance_criterion_20261024.md` except for the length of the
acceptance interval:

- Arrival every P = 20.48 ms from windows preloaded in RAM (dataset indices
  0–4095, cycled in order), one intact 32,768-byte MM2S frame per window.
- Metric: `response_k = result available − release_k` (includes waiting,
  copy, DMA send, FPGA, DMA receive, class read); no skipping.
- Warm-up: first 200 windows excluded.
- **Acceptance interval: the next 100,000 consecutive windows (≈ 34.1 min).**
- Configuration: the validated one — `tools/board/fpga_rt_acceptance.py`
  (sha256 `a473a225dcc00e96…`, unchanged), per-process SCHED_FIFO priority 50,
  CPU affinity {1}, argmax deployment, independent guard (75 °C), stream mode.
- Pass = 0 responses > 20.48 ms in the acceptance interval, every class equal to
  the recorded board prediction, all 4 logits equal to the reference, guard rc 0,
  not aborted. Otherwise: AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME NOT VERIFIED
  (with the exact miss count). A thermal or other abort is reported as such and is
  not retried.
- Scope: board-side RAM-backed workload, this scheduler/affinity configuration,
  this duration, kernel 4.14.0-xilinx `SMP PREEMPT` (not PREEMPT_RT) → empirical
  measured bound, not a formal worst-case proof. The live demo path (gzip/SD, SSH,
  Windows GUI) stays AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME NOT VERIFIED.
