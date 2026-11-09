# PREEMPT_RT physical boot and stock-vs-RT hard-RT comparison — 2026-10-24

Authorised by Mehrdad on 2026-10-24 (additive SD files, reboots, one-time U-Boot boot over
COM6; no `saveenv`, no flashing, no QSPI/network change). Nothing committed or pushed.
Criterion fixed before the RT runs: `hard_rt_preempt_rt_criterion_20261024.md`.
Evidence: `artifacts/results/preempt_rt_boot_20261024/` (boot/state) and
`artifacts/results/hard_rt_preempt_rt_20261024/` (timing runs, `verification.json`,
`comparison.{json,md}`). Preceding feasibility/rollback: `hard_rt_longrun_preempt_rt_20261024.md`.

## 1. Result in one paragraph
The PC-built PREEMPT_RT kernel (`4.14.0-rt1-rt-xilinx+`) **booted once** on the Z-Turn from the
U-Boot prompt, ran the deployed argmax overlay with identical FPGA output (40/40 golden vectors,
all classes and all four logits equal on 160,400 timed windows), and met the 20.48 ms deadline in
every accepted window (0 misses in 30,000 + 30,000 + 100,000). **It gives no meaningful
improvement** over the stock kernel for this workload: with per-process SCHED_FIFO 50 on CPU 1
the stock kernel already had 0 misses (R1, R2, L1); the RT kernel lowers p99–max by only
≈0.03–0.10 ms, with *more* involuntary context switches. The board was returned to the original
`4.14.0-xilinx` kernel by a normal reboot and verified.

## 2. Procedure and safety record
| step | evidence | result |
|---|---|---|
| rollback package re-hashed (6 files) + candidate SHA256SUMS | (command output in session; files unchanged) | all OK |
| pre-change state, BOARD_IDLE, boot partition hashes (ro mount) | `00_pre_state.txt`, `01_pre_bootpart_ro.txt` | `BOOT.BIN 19ef4427…`, `image.ub e04eabda…` = rollback |
| UART COM6 capture check (no reboot) | `02_uart_check.log` | prompt received |
| additive deploy: `image_rt.ub` → FAT p1 (new file), modules → `/lib/modules/4.14.0-rt1-rt-xilinx+` (new dir) | `03_deploy_additive.txt`, `03b_*`, `03c_modules_match.txt` | `image_rt.ub d939fb14…` on board; 160/160 module files identical; `image.ub`/`BOOT.BIN` hashes unchanged |
| unplanned normal reboot (script aborted on ssh stderr before stopping autoboot) | `09_failed_attempt_script_abort_after_trigger.log` | booted stock kernel; shows the additive files do not affect default boot; script fixed |
| one-time boot `fatload mmc 0 0x10000000 image_rt.ub && bootm 0x10000000`, no saveenv | `10_uart_boot_rt.log`, `.meta.json` | FIT hashes OK, `Linux version 4.14.0-rt1-rt-xilinx+ … SMP PREEMPT RT`, login reached |
| kernel proof | `11_rt_kernel_proof.txt` | `uname` `SMP PREEMPT RT`; `/sys/kernel/realtime`=1; `/proc/config.gz`: `PREEMPT_RT_FULL=y`; same cmdline; threaded IRQs (`irq/*` FIFO 50) |
| function: preflight (BIT/HWH identity, DMA contract, XADC), overlay load, DMA, class register, guard | `20_rtk_golden40/` | PASS 40/40, class_source=fpga_register, identical to the 2026-10-22 stock-kernel golden output |
| timing runs K1–K3 | `hard_rt_preempt_rt_20261024/` | see §3 |
| restore: normal reboot, autoboot not interrupted | `50_uart_boot_restore_stock.log` | `reading image.ub`, `Linux version 4.14.0-xilinx … SMP PREEMPT Wed Nov 13 03:44:45 UTC 2019` |
| restored-state proof | `51_stock_restored_state.txt` | uname stock, no `/sys/kernel/realtime`, `BOOT.BIN`/`image.ub` hashes = rollback, BIT `ce3c254b…`/HWH `eabc0163…`, BOARD_IDLE |

`/lib` and `/lib/modules` had their directory mtime set by tar (mode 755/root unchanged).

## 3. Comparison (all from raw npz via the independent verifier)
Same runner (sha256 `a473a225…`), overlay, windows 0–4095 cycled, 200 warm-up, 20.48 ms
deadline, SCHED_FIFO 50 + CPU {1} for the runner process only, guard 75 °C.

| run | kernel | sched | accepted | misses | p50 | p99 | p99.9 | p99.99 | max (ms) | start lag max | rate /s | classes | 4 logits | invol. ctx sw | °C start→max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | stock | default | 30,000 | **73** | 15.83 | 16.92 | 24.18 | 36.35 | 43.86 | 23.50 | 48.83 | all | all | 4,341 | 46.6→47.2 |
| R1 | stock | FIFO50/CPU1 | 30,000 | 0 | 15.82 | 16.26 | 16.40 | 16.47 | 16.55 | 0.044 | 48.83 | all | all | 0 | 47.1→47.2 |
| R2 | stock | FIFO50/CPU1 | 30,000 | 0 | 15.82 | 16.26 | 16.41 | 16.48 | 16.51 | 0.063 | 48.83 | all | all | 0 | 46.8→47.0 |
| L1 | stock | FIFO50/CPU1 | 100,000 | 0 | 15.82 | 16.27 | 16.40 | 16.49 | 16.56 | 0.087 | 48.83 | all | all | 2 | 46.3→47.1 |
| **K1** | PREEMPT_RT | FIFO50/CPU1 | 30,000 | **0** | 15.84 | 16.23 | 16.35 | 16.41 | 16.46 | 0.070 | 48.83 | all | all | 675 | 46.6→58.9 |
| **K2** | PREEMPT_RT | FIFO50/CPU1 | 30,000 | **0** | 15.84 | 16.25 | 16.36 | 16.42 | 16.45 | 0.040 | 48.83 | all | all | 685 | 59.3→59.8 |
| **K3** | PREEMPT_RT | FIFO50/CPU1 | 100,000 | **0** | 15.84 | 16.24 | 16.35 | 16.42 | 16.47 | 0.063 | 48.83 | all | all | 2,240 | 46.6→49.5 |

("all" = 30,200/30,200 or 100,200/100,200 executed windows, incl. warm-up.) No guard emergency
stop in any run. Reproduce: `python -m tools.verify_hard_rt_20261024 --base artifacts/results/hard_rt_preempt_rt_20261024 --out-name verification.json`
and `python -m tools.compare_hard_rt_kernels_20261024`.

Honest reading:
- The deadline outcome is the same on both kernels under FIFO50/CPU1 (0 misses). PREEMPT_RT's
  tail is ≈0.03–0.10 ms lower (p99.9 16.35 vs 16.40; max 16.45–16.47 vs 16.51–16.56) and p50 is
  ≈0.02 ms higher — differences of <1 % of the 20.48 ms period, i.e. **no practical improvement**.
- Margin to the deadline is ≈4.0 ms on both kernels; it is dominated by the fixed DMA+FPGA time.
- PREEMPT_RT did not run the scheduler-free default case (R0) — not part of the brief.
- **K2 transport incident:** the PC's SSH stream dropped at 388 s (`Connection reset`); the board
  logged `macb … eth0: link down` / `link up` 7 s later. The workload was not aborted and not
  retried; it completed under its guard and its raw record was collected afterwards
  (`K2_TRANSPORT_NOTE.md`). Its exit-code lines were lost with the SSH channel, so the verifier's
  formal flag is false for K2 although every timing/class/logit condition holds on the raw data.
  The same link-down occurred on the stock kernel on 2026-10-23, so it is not attributed to RT.
- **Temperature:** after the RT boot the die warmed from 46.6 to ≈59.8 °C and then cooled back to
  ≈47 °C at ≈17.5 min uptime *while K2 was still running*, i.e. a transient heat source unrelated
  to the workload. K3 (later) peaked at 49.5 °C. After the stock reboot, sampling at 1/min showed
  no such warm-up (`52_stock_postboot_temp_cpu.txt`). The source was not identified.

## 4. Boundaries (unchanged wording rules)
- **HARD REAL-TIME VERIFIED** applies only to the board-side RAM-backed workload with
  per-process SCHED_FIFO 50 + CPU 1, for the tested durations: stock kernel 2×30,000 + 100,000,
  PREEMPT_RT kernel 30,000 + 30,000 (K2: raw data, exit codes lost) + 100,000 windows.
- On the stock kernel it is an empirical measured bound, not a formal proof. On PREEMPT_RT it is
  also a measured result for these durations; PREEMPT_RT bounds kernel latencies better but no
  worst-case analysis was done, so it is **not a formal proof either**.
- The deployed default stays the stock kernel. The RT kernel is an optional, unaccepted variant
  (built with a different compiler, gcc 9.4; open risk list in `hard_rt_longrun_preempt_rt_20261024.md` §4).
- Live SD/gzip + SSH path: **AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME NOT VERIFIED**.

## 5. Files left on the board (clearly labelled, inert for normal boot)
- SD FAT p1: `image_rt.ub` (sha256 `d939fb14…`) — only loaded by the manual U-Boot command.
- `/lib/modules/4.14.0-rt1-rt-xilinx+/` — only used when that kernel runs.
- Label file: `/home/xilinx/PREEMPT_RT_EXPERIMENT_20261024_README.txt`.
- Normal boot verified after the experiment (§2). Removal (only if requested): mount p1 rw, delete
  `image_rt.ub`, `rm -r /lib/modules/4.14.0-rt1-rt-xilinx+`, `sync`; then re-hash `BOOT.BIN`/`image.ub`.

## 6. Re-running the RT boot (unchanged default)
`powershell -File tools\uart_rt_oneshot_boot.ps1 -Image image_rt.ub -OutFile <log> -MetaFile <json> -RebootCmd "ssh -i %USERPROFILE%\.ssh\id_zturn -o BatchMode=yes root@192.168.2.99 sync;reboot"`;
return with `-NoInterrupt` (normal reboot) or a power cycle.
