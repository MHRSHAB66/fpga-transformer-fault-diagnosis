# Hard real-time acceptance criterion (fixed BEFORE the 2026-10-24 runs)

Written before any code change or measurement for this test. It must not be
relaxed after a result is seen.

## Arrival model
- One complete window (1024 samples × 8 channels, one intact 32,768-byte MM2S
  frame) **arrives every P = 1024 / 50,000 s = 20.48 ms** (48.828125 windows/s).
- Arrival k happens at `release_k = t0 + k·P` on the board's monotonic clock.
  A window is never processed before its release (no look-ahead).
- Arrivals are emulated from prepared MAFAULDA windows that are **preloaded into
  board RAM** before t0 (stand-in for a sensor DMA writing into DDR). The SD-card
  file is not read during the timed interval. Footprint and loading time of the
  preload are measured and reported. This is a prepared-window emulation, not a
  live sensor.

## Metric (per window, nothing hidden)
- `response_k = t_result_k − release_k`, where `t_result_k` is when the FPGA's
  class (from the `pred_class` register path of the deployed driver) and logits
  are available to the board program. It includes any wait for an earlier window
  (queueing), the copy into the DMA buffer, DMA send, FPGA execution, DMA receive
  and the class read.
- No window is skipped: a late window is still processed (and counted late).
- Also recorded: start lag (`t_start_k − release_k`), DMA+FPGA time, output
  interval, and correctness of every class against the recorded board
  predictions.

## Pass condition
- **Deadline:** `response_k ≤ P = 20.48 ms` (the result of window k is ready
  before window k+1 arrives, so no backlog can build).
- **Warm-up:** the first 200 windows (≈ 4.1 s) are excluded.
- **Acceptance interval:** the next **30,000** consecutive windows (≈ 10.24 min).
- **HARD REAL-TIME VERIFIED** only if, in the acceptance interval, there are
  **0 deadline misses**, every class equals the recorded prediction, the guard
  exits 0, and the run is not aborted.
- Otherwise: **AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME NOT VERIFIED**
  (with the exact miss count), or **BLOCKED** if the run cannot be executed.

## Scope limits declared in advance
- Covers the board path (arrival in RAM → FPGA class available on the board).
  PC arrival and GUI rendering are measured separately and are not part of this
  board-side criterion; the PC path (SSH/Ethernet, Windows) has no real-time
  guarantee and is reported only as measured.
- Kernel is `SMP PREEMPT`, not PREEMPT_RT; a pass is an empirical bound for the
  measured interval, not a formal worst-case proof.
