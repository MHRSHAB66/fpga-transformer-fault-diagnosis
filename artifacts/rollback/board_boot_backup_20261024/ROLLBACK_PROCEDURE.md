# Rollback package — MYIR Z-Turn XC7Z020, PYNQ 2.4 image (taken 2026-10-24, read-only)

Nothing on the board was modified to create this package. The SD boot partition was
mounted **read-only** in `/tmp` for copying and unmounted immediately (it is not
mounted in normal operation). QSPI partitions were read with `cat /dev/mtdN`.

## Contents (sha256 in `local_sha256.txt`, board-side hashes in `board_side_hashes.txt` — identical)
| file | origin | role |
|---|---|---|
| `BOOT.BIN` (`19ef4427…`) | SD `mmcblk0p1` | FSBL + U-Boot 2018.01 ("U-BOOT for zturn", built 2019-11-13) — the U-Boot actually used |
| `image.ub` (`e04eabda…`) | SD `mmcblk0p1` | FIT image: kernel 4.14.0-xilinx + device tree (booted by `cp_kernel2ram && bootm`) |
| `qspi_mtd0.bin` … `qspi_mtd3.bin` | QSPI `boot`, `bootenv`, `kernel`, `spare` | factory QSPI content (U-Boot 2013.10 from 2015); not used by the current SD boot |
| `proc_config_4.14.0-xilinx.txt` | `/proc/config.gz` | running kernel configuration (`CONFIG_PREEMPT=y`, `HZ=100`, no PREEMPT_RT) |
| `board_inventory_readonly.txt` | SSH | uname, cmdline, OS/PYNQ revision, modules, mtd, lsblk, memory |

Root filesystem: `mmcblk0p2` (ext4, 29 GB, 7.2 GB used). Kernel cmdline:
`root=/dev/mmcblk0p2 rw earlyprintk rootfstype=ext4 rootwait devtmpfs.mount=1 uio_pdrv_genirq.of_id="generic-uio" clk_ignore_unused`.
U-Boot default: `bootcmd=run default_bootcmd` → `run uenvboot` (imports `uEnv.txt`
from the SD boot partition if present — currently absent) → load `image.ub` → `bootm`;
`bootdelay=4` (interruptible on the UART, COM6, 115200 8N1).

## Restoring the current known-good boot (if an experiment ever changed it)
Any planned RT experiment must only ADD files (e.g. `image_rt.ub`) and boot them once
from the U-Boot prompt, so normally nothing needs restoring: a power cycle boots
`image.ub` again. If the boot partition was changed anyway:
1. Power off the board, remove the SD card, insert it into the PC.
2. On the FAT boot partition: delete any added `uEnv.txt` and added images; copy
   `BOOT.BIN` and `image.ub` from this folder back; verify with
   `sha256sum BOOT.BIN image.ub` against `local_sha256.txt`.
3. Re-insert, power on, check over SSH: `uname -a` must print
   `4.14.0-xilinx #1 SMP PREEMPT Wed Nov 13 03:44:45 UTC 2019`; run
   `python -m host.live_preflight --n 1` (must PASS, BIT/HWH hashes unchanged).
4. If the board does not boot even with the original files, the SD root filesystem is
   suspect — restore from a full SD image (not part of this package; take one with a
   card reader before any experiment that writes to the card).
QSPI is never written by any planned step; `qspi_mtd*.bin` exist only for completeness.

The FPGA deployment directory `/home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f`
lives on the root filesystem and is not touched by any boot change.
