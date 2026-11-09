# Long-run hard real-time validation and PREEMPT_RT feasibility — 2026-10-24

> **Update (later on 2026-10-24, authorised):** the candidate below was then booted once and
> compared physically — see `hard_rt_preempt_rt_comparison_20261024.md`. The text below is the
> state at the time of writing and is kept unchanged.

Nothing was committed or pushed. The board was **not** flashed, rebooted or
reconfigured; no file on the board was changed (the SD boot partition was only
mounted read-only in `/tmp` for copying). Deployment: argmax-in-FPGA overlay
`seq256_argmax_review_d55cbdfe469f` (BIT `ce3c254b…`, HWH `eabc0163…`).

## 1. Verdicts

| scope | verdict |
|---|---|
| Board-side RAM-backed workload (`tools/board/fpga_rt_acceptance.py`, per-process SCHED_FIFO 50, CPU affinity {1}), tested durations | **HARD REAL-TIME VERIFIED** — 0 deadline misses in 30,000 (R1), 30,000 (R2) and **100,000 (L1)** windows |
| Same workload, default scheduler (R0) | 73 misses in 30,000 — not hard real-time (kept as evidence) |
| Live demo path (gzip/SD input, SSH, Windows GUI) | **AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME NOT VERIFIED** (unchanged) |
| Kernel | 4.14.0-xilinx `SMP PREEMPT`, **not PREEMPT_RT** → all results are empirical measured bounds for the tested durations, **not a formal worst-case proof** |
| PREEMPT_RT on the board | **not available**; a candidate kernel was **built on the PC** but has **not been booted or validated** — pending your approval and a controlled maintenance window |

## 2. Long-run test L1 (PHYSICAL)

Criterion fixed before the run: `docs/reports/hard_rt_longrun_criterion_20261024.md`
(same as the 30k criterion, acceptance interval 100,000 windows).
Runner sha256 `a473a225dcc00e96…` (identical to R1/R2). Command:
`powershell -File tools/run_hard_rt_20261024.ps1 -Only L1_fifo50_cpu1_100k`.

| run | sched | accepted windows | misses | p50 | p99 | p99.9 | p99.99 | max (ms) | rate /s | classes = recorded | 4 logits = ref | guard | max °C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | default | 30,000 | **73** | 15.83 | 16.92 | 24.18 | 36.35 | 43.86 | 48.8285 | 30,200/30,200 | 30,200/30,200 | 0 | 47.4 |
| R1 | FIFO 50, CPU1 | 30,000 | **0** | 15.82 | 16.26 | 16.40 | 16.47 | 16.55 | 48.8285 | 30,200/30,200 | 30,200/30,200 | 0 | 47.4 |
| R2 | FIFO 50, CPU1 | 30,000 | **0** | 15.82 | 16.26 | 16.41 | 16.48 | 16.51 | 48.8285 | 30,200/30,200 | 30,200/30,200 | 0 | 47.1 |
| **L1** | FIFO 50, CPU1 | **100,000** | **0** | 15.82 | 16.27 | 16.40 | 16.49 | **16.56** | 48.8282 | 100,200/100,200 | 100,200/100,200 | 0 | 47.0 |

L1 details: 34.2 min (2,052 s), max start lag after release 0.087 ms, 2 involuntary
context switches in the whole run, preload 7.97 s / 149 MB, one SSH login during the
run (its own stream session), no emergency stop, post-run `BOARD_IDLE`. Calibrated
CONF recomputed from the physical logits equals the reference (1,033 sampled windows,
max |Δ| = 0). Every latency is in `artifacts/results/hard_rt_20261024/L1_fifo50_cpu1_100k/remote/out/rt_raw.npz`;
independent check: `python -m tools.verify_hard_rt_20261024` →
`artifacts/results/hard_rt_20261024/verification_v2.json` (the earlier
`verification.json` is kept unchanged).

## 3. PREEMPT_RT inspection (read-only)

Evidence: `artifacts/rollback/board_boot_backup_20261024/` (`board_inventory_readonly.txt`,
`proc_config_4.14.0-xilinx.txt`, hashes).

- Board: MYIR Z-Turn (Zynq-7020, `xlnx,zynq-7000`), PynqLinux **PYNQ 2.4**, revision
  `2019_11_12 d9c7521` (= PYNQ tag `v2.4`).
- Kernel: `4.14.0-xilinx #1 SMP PREEMPT Wed Nov 13 03:44:45 UTC 2019`, gcc 7.3.0;
  config: `CONFIG_PREEMPT=y`, no `PREEMPT_RT*`, `CONFIG_HZ=100`, `HIGH_RES_TIMERS=y`.
- Boot: SD card. `mmcblk0p1` (FAT, 100 MB, not mounted) holds `BOOT.BIN` (FSBL +
  U-Boot 2018.01 "for zturn") and `image.ub` (FIT: kernel@0 zImage 4.16 MB at 0x80000 +
  fdt@0 14.9 KB, no initramfs). Root fs `mmcblk0p2` ext4 29 GB (20 GB free).
  U-Boot: `bootdelay=4`, `bootcmd → uenvboot (optional uEnv.txt, absent) →
  fatload image.ub → bootm`. QSPI holds an unused factory U-Boot (2015).
- No other kernel, no kernel packages, no sources, no `/boot`; one modules dir.
- Conclusion: **no PREEMPT_RT kernel is installed; none can be selected without
  adding files and rebooting.**

## 4. Feasibility on the PC (no board involvement)

In WSL Ubuntu 20.04, without root (toolchain from Ubuntu `.deb`s unpacked in the user
home): `linux-xlnx` tag `xilinx-v2018.3` (commit `51445f91…`, the PetaLinux 2018.3
kernel PYNQ 2.4 uses) + `patch-4.14-rt1` (kernel.org, sha256 `1025b5be…`):
- patch dry-run: 430 files, **0 failed hunks**, 2 offset hunks;
- config = board `/proc/config.gz` + `PREEMPT_RT_FULL=y`, `LOCALVERSION=-rt-xilinx`;
- build `zImage modules` with arm-linux-gnueabihf-gcc **9.4** (board kernel: 7.3): rc 0,
  712 compiler warnings; version string
  `4.14.0-rt1-rt-xilinx+ … SMP PREEMPT RT`;
- candidate FIT `image_rt.ub` in the same layout as `image.ub` (zImage 4.19 MB at
  0x80000 + the board's **unchanged** DTB, sha256 `b72d3b5a…`) and 148 modules for
  `4.14.0-rt1-rt-xilinx+`.
Artifacts: `artifacts/preempt_rt_candidate_20261024/` (`image_rt.ub`, `image_rt.its`,
`modules_rt.tar.gz`, `config_4.14.0-rt1-rt-xilinx.txt`, `SHA256SUMS`, inputs, build log tail).

Open compatibility risks (why it is **not** called compatible yet): PYNQ 2.4 may carry
kernel patches beyond `xilinx-v2018.3`; different compiler version; RT-safety of the
Xilinx DMA/xlnk/fpga-manager drivers is untested; PYNQ user space might expect the
exact module directory name. Only a boot test can settle these.

## 5. Rollback package (verified)

`artifacts/rollback/board_boot_backup_20261024/`: `BOOT.BIN`, `image.ub`, QSPI
`mtd0–3`, kernel config, inventory; local hashes = board-side hashes (all 6 OK);
`ROLLBACK_PROCEDURE.md` gives the restore steps. **Missing for a fully safe
experiment:** a full SD-card image taken with a card reader (only that protects
against an unexpected write to the root filesystem).

## 6. Proposed migration plan (NOT executed — needs your approval)

Least-persistent route: the default boot is never changed; the RT kernel is booted
**once** from the U-Boot prompt, and any power cycle returns to the known-good kernel.

1. Prerequisites: you are physically at the board (power cycle, SD card) for the
   whole window; full SD image taken; dashboard closed.
2. Add (do not replace) files: `image_rt.ub` → SD boot partition (FAT) as a new file;
   `modules_rt.tar.gz` → unpack to `/lib/modules/4.14.0-rt1-rt-xilinx+` (new dir).
   Both are additive; `image.ub`, `BOOT.BIN`, `/lib/modules/4.14.0-xilinx` untouched.
3. `sync; reboot` over SSH while `tools/uart_boot_capture.ps1`-style capture on COM6
   interrupts autoboot (bootdelay 4 s) and types:
   `fatload mmc 0 0x10000000 image_rt.ub && bootm 0x10000000` (no `saveenv`).
4. Acceptance before any timing test: `uname -a` contains `PREEMPT RT`,
   `/sys/kernel/realtime` = 1; SSH works; XADC temperature valid;
   `python -m host.live_preflight --n 1` PASS; argmax golden test 40/40
   (`verify_argmax_golden.py`); short guarded live run with class_source=fpga_register.
   Any failure → power cycle (back to `image.ub`) and stop.
5. Apples-to-apples runs (same runner, windows, SCHED_FIFO 50/CPU1, guard, verifier):
   2 × 30,000 + 1 × 100,000 windows; then power cycle back to the original kernel and
   verify `uname`, preflight and BIT/HWH hashes; remove the added files only if you ask.
6. Report both kernels in one table; no claim of improvement unless measured.

Approvals needed: (a) writing two additive items to the board's SD card,
(b) rebooting the board (twice), (c) your physical presence for recovery.

## 7. Reproduction
```powershell
powershell -File tools/run_hard_rt_20261024.ps1 -Only L1_fifo50_cpu1_100k
python -m tools.verify_hard_rt_20261024
# PC-only feasibility (WSL): scratch scripts copied to artifacts/preempt_rt_candidate_20261024/scripts/
```
