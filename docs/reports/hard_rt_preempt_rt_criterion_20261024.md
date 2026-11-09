# PREEMPT_RT comparison — acceptance criterion (fixed BEFORE the RT-kernel runs, 2026-10-24)

Identical to `hard_rt_acceptance_criterion_20261024.md` / `hard_rt_longrun_criterion_20261024.md`;
only the kernel differs.

- Kernel under test: `4.14.0-rt1-rt-xilinx+ #1 SMP PREEMPT RT` booted ONCE from U-Boot
  (`fatload mmc 0 0x10000000 image_rt.ub && bootm 0x10000000`, no `saveenv`).
  Proof: `artifacts/results/preempt_rt_boot_20261024/11_rt_kernel_proof.txt`.
- Workload: `tools/board/fpga_rt_acceptance.py` (sha256 a473a225…, unchanged), same
  overlay (BIT ce3c254b…, HWH eabc0163…), RAM-preloaded windows 0–4095 cycled, release
  every 20.48 ms, 200-window warm-up, response = done − release, miss = response > 20.48 ms,
  per-process SCHED_FIFO 50, CPU affinity {1}, independent guard 75 °C / 1 s, no retry.
- Runs: K1 30,200 (30,000 accepted), K2 30,200 (repeat), K3 100,200 (100,000 accepted).
- Pass per run: 0 misses, classes = recorded 2026-10-08 predictions for every executed
  window, all four logits = INT10 reference, guard rc 0, workload rc 0.
- Verifier: `python -m tools.verify_hard_rt_20261024 --base artifacts/results/hard_rt_preempt_rt_20261024 --out-name verification.json`.
- Comparison against R0/R1/R2/L1 (stock kernel) is descriptive; no improvement is claimed
  unless the numbers show it. Note: on PREEMPT_RT the IRQ threads also run at FIFO 50;
  the runner priority is deliberately NOT changed (fair comparison).
