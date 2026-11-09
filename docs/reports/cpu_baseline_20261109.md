# CPU baseline: the same INT10 model on the board's ARM vs the FPGA (2026-11-09)

**Question:** how much does the FPGA gain over running the identical quantized
model on the processor that is already on the board (Zynq PS, dual ARM
Cortex-A9, 667 MHz), and can that processor alone meet the 20.48 ms real-time
period?

## Method

- Model: the deployed SEQ=256 INT10 top (argmax variant) compiled as ordinary
  C++ — the same source the FPGA core was synthesized from, so the computation is
  identical. Its logits must equal the physical board logits of the same windows.
- Windows: the 256-window class-balanced RAM demo set (64 per class), loaded into
  memory first; the timer covers only the inference call (4:1 aggregation +
  transformer + argmax), not file I/O.
- Builds on the board (g++ 7.3): `-O2`, and `-O3 -mcpu=cortex-a9 -mfpu=neon
  -mfloat-abi=hard`. One process pinned to CPU 1 (latency), then two processes
  on CPU 0 and CPU 1 (throughput). Every run under the temperature guard.
- FPGA reference: board-side response p50 15.82 ms (release → FPGA class ready,
  includes the 32 KB DMA transfer; hard-RT run R1, 2026-10-24).
- Tools: `hls/f1_int10_seq256/gpp_csim/cpu_baseline_bench.cpp`,
  `tools/board/run_cpu_baseline.sh`, `tools/cpu_baseline.py`; evidence
  `artifacts/results/cpu_baseline_20261109/`.

## Results (ARM Cortex-A9, per window)

| Build / run | p50 | mean | p95 | max | Logits = board | FPGA speed-up (p50) |
|---|---|---|---|---|---|---|
| -O2, one core | 218.6 ms | 221.9 ms | 234.2 ms | 279.6 ms | 256/256 | 13.8× |
| -O3 + NEON tuning, one core | 216.5 ms | 218.8 ms | 227.6 ms | 260.2 ms | 256/256 | 13.7× |
| -O2, two cores in parallel | 219.6 ms | 230.1 ms | 274.3 ms | 363.8 ms | 256/256 each | — |

Two cores together: about 8.4–8.7 windows/s (wall time includes each process's
untimed warm-up pass), against 48.83 windows/s required and 60.9 windows/s
measured on the FPGA path.

## Reference point: a laptop CPU (not an embedded option)

Same source and windows on the development PC (Intel Core i7-13620H, g++ 15.2,
one process pinned to a performance core, priority High; unpinned runs were
noisy because Windows moved the process between core types and are not used):

| Build | p50 | mean | p95 | max | Logits = board |
|---|---|---|---|---|---|
| -O2 | 7.98 ms | 7.92 ms | 8.84 ms | 9.54 ms | 256/256 |
| -O3 -march=native | 6.64 ms | 6.64 ms | 7.58 ms | 8.32 ms | 256/256 |

A 45 W-class laptop processor runs one window about 2.0–2.4× faster than the
FPGA core. The FPGA's advantage in this project is over the processor that is
actually on the embedded board, with the hard deadline verified on the board.
Power was not measured (no instrument), so no energy comparison is claimed.

## Conclusion

- The FPGA core is about **14× faster** per window than one ARM core running the
  bit-identical integer model, and about 7× faster than both ARM cores together.
- The ARM alone needs ~219 ms per window, **~10.7× longer than the 20.48 ms
  period**; real-time classification of this model on the board is only possible
  with the FPGA.
- Compiler auto-vectorization (NEON) gains ~1 %: the integer requantization and
  softmax/LayerNorm code is scalar. A hand-vectorized ARM version could be
  faster; that was not attempted, so the speed-up is relative to the same
  portable C++.
