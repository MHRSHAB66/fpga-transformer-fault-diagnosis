# ram_realtime_demo — acceptance criterion (fixed BEFORE the physical runs, 2026-10-24)

Mode: `ram_realtime_demo` (`host/ram_rt_demo.py` + `tools/board/fpga_ram_rt_demo.py`), stock
kernel 4.14.0-xilinx, deployed argmax overlay (BIT ce3c254b…, HWH eabc0163…).

Input set: `artifacts/ram_demo_sets/class_balanced_256_v1/` — 256 REAL test-set windows
(64 per true class, `host.live_sampling` class_balanced plan), INT10 codes copied byte for byte
from `fpga_testset_in.i32.gz`; set.i32 sha256 `06ac41bd…`; staged on the board in
`/home/xilinx/ram_demo_sets/class_balanced_256_v1/` and re-hashed by every preflight.

## Two measurements, two boundaries
1. **Board response latency** = board release (t0 + k·20.48 ms, board clock) → FPGA class ready
   (S2MM complete + class register read). This is the hard-RT boundary.
   Pass (per run): n_total = 30,200 (warm-up 200, 30,000 accepted), 0 accepted windows > 20.48 ms,
   class == recorded 2026-10-08 FPGA prediction and all four logits == INT10 reference for every
   executed window (by real dataset index), guard rc 0, workload rc 0, no runner/guard left.
   Execution policy: SCHED_FIFO 50 + CPU {1} for the runner's timed main thread only; the
   temperature thread and the publisher process keep normal scheduling.
2. **End-to-end latency** = board release → PC receipt of the result line, and → Tk GUI update,
   on the PC clock (offset from ≥1 clock-sync anchor before and after the run over the same SSH
   channel; error bound = RTT/2). **Measured and reported only — not part of any hard-RT claim**,
   because SSH/Ethernet and Windows scheduling are not controlled.

The display hand-off (non-blocking pipe write) occurs after the class-ready timestamp; any line
the PC cannot take is dropped for display and counted (`display_dropped`), never delaying the board.

Verifier: `python -m tools.verify_ram_rt_demo <evidence_dir>`. No automatic retry.
