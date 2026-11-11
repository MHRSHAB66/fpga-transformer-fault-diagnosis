# Project Status

**Project:** Real-Time Machinery Condition Monitoring Using a Transformer Neural Network on FPGA  
**Student:** Mehrdad Sheikh Abbasi, B.Sc., Amirkabir University of Technology  
**Last updated:** 2026-11-11 (paper study; temporal vote in FPGA logic validated on the board). Earlier: 2026-11-09 evening (UART path, preprocessing in the FPGA, CPU baseline, SD image; earlier the same day the final review: all 2026-10-22..24 work committed, off-repo backup, ROC/AUC + final figures, artifact manifest; remaining board work planned). Reports: `docs/reports/consolidated_results_20261022.md` (canonical numbers, §8 ROC/AUC), `docs/reports/hard_rt_calibration_power_20261024.md`, `docs/reports/gui_demo_timing_20261023.md`. Figures: `docs/figures/final/`. Demo runbook: `docs/demo_runbook.md`.

---

## Stage Status

| Stage | Name | Status |
|---|---|---|
| 0 | Project setup, environment, config | **COMPLETE** |
| 1 | MAFAULDA data discovery, class mapping | **COMPLETE** |
| 2 | Preprocessing, windowing, DataLoaders | **COMPLETE** |
| 3 | MLP baseline (sanity check) | **COMPLETE** |
| 4 | Lightweight Transformer training | **COMPLETE** |
| 5 | Clean evaluation, metrics export, artifacts | **COMPLETE** |
| 6 | Quantization & fixed-point preparation | **COMPLETE** (int10 selected) |
| 7 | Hardware specification & golden vectors | **COMPLETE** (F@int10 export, verify 16/16) |
| 8 | FPGA inference core | **COMPLETE — physically validated** on real MYIR Z-Turn XC7Z020 (full-test 95,404 windows + live multiclass) |
| 9 | Sensor / lab input path | Preprocessing now inside the FPGA (raw 24-bit samples in, validated on the board 2026-11-09); a physical sensor stays optional |
| 10 | FPGA-to-PC communication | **COMPLETE** — results over the board's USB-UART (10,000/10,000 frames, 2026-11-09) and over SSH/TCP Ethernet |
| 11 | PC-side display UI | **COMPLETE** — Replay + Live FPGA viewer (`host/fpga_result_viewer.py`) |
| 12 | Final integration, validation, demo | **COMPLETE (hardware-validated)** — physical full-test + live multiclass + UART boot; demo flow in `docs/demo_runbook.md` |

---

## Software Phase (Stages 0-5): COMPLETE

The software/model phase is complete as of 2026-06-15.

**Baseline model (highest accuracy):** 2-layer Transformer,
**98.22%** test accuracy on 95,404 unseen windows.
Checkpoint: `artifacts/runs/2layer_baseline/checkpoints/transformer_best.pt` (local copy of
Kaggle run `20260615_223905_fpga_fault_final_t4`; metrics `artifacts/runs/2layer_baseline/results/metrics.json`).
See `docs/final_software_result.md`.

## FPGA-deployed model (Stages 6-8): F_1layer @ int10

For the FPGA path a smaller architecture was selected to reduce the eventual
resource and latency cost: **F_1layer** (1 encoder layer, d_model=32, 4 heads,
ffn=64, 9,492 params). The deployment board is the **MYIR Z-Turn XC7Z020** (PYNQ
2.4 / Python 3.6); the system has been **physically validated** on it (below).

**Current hardware path — SEQ=256 INT10 (aggregated ingress):** the acquisition
window stays 1024×8; the hardware IP quantizes to INT10 and performs an exact 4:1
integer aggregation (1024→256 tokens) internally.

- **Physical FPGA full-test (deployed system, demo headline):** accuracy
  **97.7684%** (93,275/95,404), macro-F1 **95.6539%** (macro-P 95.9332%, macro-R
  95.4233%) on the real board, all 95,404 windows
  (`artifacts/results/fpga_fulltest_20261008_141335/fpga_fulltest_metrics.json`).
- Python INT10 reference **with the deployed (export) scales**: 97.7684% (93,275) —
  identical to the board on all 95,404 windows, including all 4 logits (physical
  logits recorded 2026-10-22). The previously quoted 97.7359% (93,244) is a
  *differently calibrated* INT10 model (12 vs 20 calibration files) that differs
  from the board on 225 windows; it is not the hardware reference
  (`artifacts/results/board_vs_reference_20261022/comparison.json`).
- Live multiclass (120-window class-balanced, real board): accuracy **95.0%**,
  macro-F1 **95.0%**, **120/120** match the recorded FPGA predictions, latency mean
  **16.15 ms** (`artifacts/results/fpga_live_multiclass_20261008_230015/`).
- Vitis HLS CSim: **16/16 exact**. Deployed dma16 build, Vivado post-route: timing
  met (WNS +0.364 ns), 210/220 DSP, 31,734 LUT, 30,921 FF, 70.5 BRAM tiles
  (`artifacts/results/seq256_dma_system_dsp_margin_dma16_z7020_20261008/`). The
  often-quoted WNS +0.036 ns / 31,687 LUT belong to the earlier 16 KB-DMA build
  (`seq256_dma_system_dsp_margin_z7020/`), which is not deployed.

**Earlier SEQ=1024 INT10 variant (historical):**
- Float full-test accuracy: **97.55%**; **int10 full-test accuracy: 97.11%**
  (drop 0.44%, no class collapse), all 95,404 windows
  (`kaggle/int10_shard/int10_fulltest_combined.json`).
- int10 chosen from a bit-width sweep `[16,14,12,10,8]`; int8 collapses the
  imbalance class (`kaggle/sweep_multibit/`).
- Bit-exact HLS export verified 16/16 (`hls/f1_int10/verify_export.py`).

Consolidated numbers: **`docs/reports/consolidated_results_20261022.md`** (all
results with evidence paths); demo figures: **`docs/demo_runbook.md` §1** (deployed SEQ=256
system). Note: `docs/final_report.md` is an older 2026-09-09 auto-snapshot of the
**SEQ=1024** variant (97.11% int10, hardware "TBD") and is superseded for the
deployed-system numbers by the runbook.

---

## What is done vs still pending

**Done (runnable / verified on this machine):**
- Quantization to int10 with full-test accuracy (97.11%).
- Bit-exact HLS/portable-C++ core + golden vectors (`hls/f1_int10/`, verify 16/16).
- Vitis project scaffold (`hls/f1_int10/vitis/`: weights header, ap_int typedefs,
  synthesizable top, run_hls.tcl, SYNTHESIS_GUIDE.md).
- PC-side UART receiver + live UI + frame protocol (`host/`, tested end-to-end).
- PYNQ driver with a verified NumPy backend + hardware-backend template (`pynq/`).

**Done (provisional/generic XC7Z020, in addition to the above):**
- Vitis HLS C-synthesis (est. 14.905 ms interval at 100 MHz) for the SEQ=256
  INT10 ingress core.
- DMA-integrated Vivado 2020.2 implementation (PS7 + AXI-DMA + IP), post-route
  timing met, resources measured (210/220 DSP with a reserved 10-DSP margin),
  bitstream/HWH/XSA generated. Package:
  `artifacts/results/seq256_dma_system_dsp_margin_z7020/`.

**Done (physical board, 2026-10-08, tag `fpga-demo-validated-20261008`):**
- On-board bitstream loading + real AXI-DMA transfer: full-test 95,404 windows and
  120-window live multiclass (all four classes), 120/120 match recorded predictions.
- Real USB-UART boot capture (COM6) and live results over SSH/Ethernet.
- Board confirmed: MYIR Z-Turn XC7Z020 (dma16 deployment, 64 KB DMA transfer).

**Done 2026-10-22 (see `docs/reports/consolidated_results_20261022.md`):**
- Board-vs-reference: same-input, same-model comparison bit-exact on 95,404 windows.
- Real-time on prepared windows: board-only 60.94 inf/s (95,404) and end-to-end
  60.30 / 58.72 inf/s (10,000) vs 48.83 required — sustained average met; hard
  per-window deadline not demonstrated.
- Live CONF from the FPGA logits (uncalibrated softmax, ECE measured).
- On-FPGA argmax variant: synthesized, routed and tested on the board: 40/40
  golden vectors including 16 ties; a separate balanced live sample achieved
  39/40 against ground truth and 40/40 against recorded prior-board predictions.
  The new overlay's full 95,404-window board run completed on 2026-10-23:
  accuracy 97.768437%, 95,404/95,404 classes matching the earlier overlay.
  See `docs/reports/live_guard_argmax_progress_20261022.md`.

**Done 2026-10-23 (see `docs/reports/gui_demo_timing_20261023.md`):**
- Physical GUI validation with the argmax overlay (real window, real board; buttons
  triggered with Tk `invoke()`, OS mouse input was unavailable): preflight, a
  40-window class-balanced run with all four classes, Start/Stop twice, Stop during
  start-up, Close mid-run, and three safety faults (leftover process, missing guard,
  guard dying mid-run) — board clean after every step.
- Timed defense rehearsal, all steps PASS (`artifacts/results/demo_rehearsal_20261023/`).
- Live logits of the argmax full test equal the Python INT10 reference on all
  95,404 windows; CONF recomputed identically; the GUI labels CONF as uncalibrated.
- Tail-latency attribution measured: SD-card input I/O stalls (up to ~0.8 s) and OS
  preemption of the busy-polling DMA loop (up to ~42 ms); extra SSH logins add
  CPU load. Queue-based prefetch (thread and process) measured WORSE on this board
  and is off by default.

**Done 2026-10-24 (see `docs/reports/hard_rt_calibration_power_20261024.md`):**
- Hard real-time acceptance test (criterion fixed beforehand,
  `docs/reports/hard_rt_acceptance_criterion_20261024.md`): board-side path with
  RAM-backed prepared windows released every 20.48 ms — **0 deadline misses in
  30,000 windows, twice**, with per-process SCHED_FIFO + CPU affinity (max response
  16.55 ms); default scheduling: 73 misses. All classes and logits = reference.
  Scope: not the live SD/SSH/GUI demo path; SMP PREEMPT kernel (empirical bound).
- Calibrated confidence: temperature T = 1.4248 fitted on validation only; test ECE
  0.79 % → 0.09 %, NLL 0.0628 → 0.0584; shown in Live mode, labelled; physical
  logits = reference.
- Long run L1 (criterion `docs/reports/hard_rt_longrun_criterion_20261024.md`):
  same RAM-backed workload and SCHED_FIFO/CPU1 configuration, **0 deadline misses
  in 100,000 windows** (34.2 min), p99.99 16.49 ms, max 16.56 ms; all classes and
  logits = reference (`docs/reports/hard_rt_longrun_preempt_rt_20261024.md`).
- PREEMPT_RT: not available on the board (stock 4.14.0-xilinx `SMP PREEMPT`);
  read-only rollback package taken (`artifacts/rollback/board_boot_backup_20261024/`);
  a candidate PREEMPT_RT kernel (linux-xlnx xilinx-v2018.3 + patch-4.14-rt1) was
  built on the PC (`artifacts/preempt_rt_candidate_20261024/`).
- PREEMPT_RT physical comparison (authorised, same day): candidate copied additively
  (`image_rt.ub` on SD p1, own module dir), booted ONCE from U-Boot over COM6 (no
  `saveenv`); proven `SMP PREEMPT RT`, `/sys/kernel/realtime`=1; golden 40/40 identical.
  Same workload: **0 misses in 30,000 + 30,000 + 100,000** (max 16.47 ms) — tail only
  ≈0.05–0.1 ms lower than stock, **no practical improvement**; K2's SSH stream was lost to
  a board Ethernet link-down (workload completed, raw data verified, exit codes lost).
  Board returned to the stock kernel by normal reboot and verified (boot-file hashes,
  BIT/HWH, BOARD_IDLE). RT files left inert + labelled
  (`docs/reports/hard_rt_preempt_rt_comparison_20261024.md`).
- New separate GUI/CLI mode `ram_realtime_demo` (legacy live mode unchanged and default):
  documented 256-window RAM set, FIFO 50/CPU 1 timed thread, non-blocking display
  hand-off. Board response **0 misses in 30,000 accepted windows** (max 16.50 ms) =
  HARD REAL-TIME VERIFIED for that boundary/duration; SSH receipt and GUI update measured
  separately (not hard RT: 2.8 % of receipts > 20.48 ms; GUI p50 ≈ 73 ms). GUI Start/Stop/
  Close validated on the real window (`docs/reports/ram_realtime_demo_20261024.md`).
- End-to-end investigation (criterion frozen first: `docs/reports/e2e_hard_rt_criterion_20261024.md`):
  existing `ram_realtime_demo` GUI path 30,000/30,000 windows late (e2e max 292 ms, 80 ms
  GUI poll); new mode `ram_tcp_e2e_demo` (TCP data plane, no SSH, per-frame 1 ms render)
  e2e p50 19.2–20.2 ms but 14,489 and 8,866 of 30,000 late (max 112/135 ms) ⇒
  **END-TO-END HARD REAL-TIME NOT VERIFIED** for both; board stage still 0 misses; all
  frames/classes/logits correct, drops accounted (`docs/reports/e2e_hard_rt_verdict_20261024.md`).
- Manual mouse pass of the GUI by Mehrdad (2026-10-24), board idle afterwards.
- Power/energy: BLOCKED — no current/power telemetry on the board and no external
  instrument (`docs/reports/power_measurement_blocker_20261024.md`).

**Done 2026-11-09 (final review):**
- Every deliverable of the signed proposal (v5) checked against code and evidence;
  `python -m unittest discover -s tests`: 203 tests OK (4 skipped).
- The 2026-10-22..24 work committed in 16 topical commits; bulky raw dumps and
  board binaries gitignored and recorded by SHA-256 in
  `artifacts/manifests/artifact_manifest.json` (73 items, tracked files hashed
  from their committed bytes).
- Off-repo backup `D:\fpga_fault_diagnosis_backup_20261109\`: git bundle of all
  refs + 2,941 untracked files (2.16 GB), every copy re-hashed.
- ROC/AUC of the deployed model from the physical logits (macro AUC 0.9991,
  micro 0.9993) and seven final figures (`python -m tools.plot_final_figures`).

**Done 2026-11-09 with the board (evening):**
- UART result path (`docs/reports/uart_result_path_20261109.md`): FPGA class ->
  PS UART -> CP2102 -> COM6, 10,000/10,000 frames byte-identical, 0 malformed/gaps/
  duplicates, classes/logits/CONF correct, 48.80 frames/s.
- Preprocessing inside the FPGA (`docs/reports/onchip_preprocessing_20261109.md`):
  raw 24-bit samples in; integer z-score + INT10 in the IP; post-route timing met
  (WNS +0.167 ns, +4.9 k LUT, +0 DSP); on the board 41/41 golden, 95,404/95,404
  logits = C-model, accuracy 97.7705 %, hard RT 0 misses in 30,000 (max 16.64 ms).
  Deployment `seq256_raw_argmax_z7020_20261109`; the INT10-input argmax overlay
  stays as fallback.
- CPU baseline (`docs/reports/cpu_baseline_20261109.md`): same INT10 model on the
  board's ARM Cortex-A9 needs ~219 ms/window -> FPGA ~14x faster; the ARM alone
  cannot meet 20.48 ms. Laptop i7 reference: ~6.6-8 ms.
- Full SD-card image of the board (31.17 GB, verified) in the off-repo backup.
- Board Ethernet pinned to 100 Mb/s (link flapped at 1 Gb/s); original setting in
  `/root/ethtool_eth0_before_20261109.txt` on the board.

**Done 2026-11-10/11 (paper study, `docs/reports/paper_readiness_review_20261110.md`):**
- Literature search and evaluation review: the project split tests new recordings
  and speeds of known conditions, not unseen severities. File-level bootstrap CIs
  (accuracy 97.77 % [97.29, 98.26]); noise and dead-sensor sweeps of the deployed
  model on the bit-exact C-model; Vivado power ESTIMATE; latency breakdown.
- Stricter splits, same architecture and recipe retrained (float): unseen speed band
  97.9 %, unseen middle severity 95.7 %, unseen highest severity 74.1 %.
- Noise-augmented training; its INT10 export (C-model = Python reference): clean
  97.40 % (normal recall 96.6 %), 0 dB 90.1 % vs 82.8 % for the deployed model.
  A second bitstream with these weights is a paper study; the deployed model is
  unchanged.
- Temporal majority vote in FPGA logic (AXI-Lite `vote_k`/`vote_class`/`vote_count`):
  +241 LUT / +186 FF, timing met (WNS +0.135 ns). On the board (deployment
  `seq256_vote_z7020_20261110`): golden 41/41, full test logits = C-model on all
  95,404 windows (97.7705 %), votes = Python on all windows, system accuracy
  99.084 % at k = 5, hard real time 0 misses in 30,000 (max 16.66 ms).
  New figures in `docs/figures/paper/` (`python -m tools.plot_paper_figures`).

**Remaining with the board (until ≈ 2026-12-09):** board run of the noise-augmented
second bitstream; daytime endurance runs; UI refresh (Mehrdad will describe it),
then a recorded demo video; power/energy if an inline DC meter is found (the board
runs from a 5 V / 3 A adapter).

**Still pending / future work:**
- Legacy live demo path (gzip from SD, SSH, GUI) remains AVERAGE-THROUGHPUT VERIFIED,
  HARD REAL-TIME NOT VERIFIED; the RAM mode covers the board boundary only — a sensor
  input path is future work.
- Power/energy measurement once an inline meter or logging supply is available.
- Optional: longer `ram_realtime_demo` run (100,000 windows); unexplained post-boot
  warm-up (≈59 °C) seen only after the RT boot.
- Optional live sensor input path (Stage 9; on-board preprocessing done 2026-11-09).

Claims: physical-board results (bit-exactness, latency, hard real time at the
board boundary, UART delivery) are established on the MYIR Z-Turn XC7Z020. Not
established: a live sensor input, end-to-end hard real time through the PC GUI,
measured power/energy (only a Vivado estimate), generalization to unseen severities
beyond the training range.

---

## Branch

All work is on `main`. The 2026-10-22..24 work was committed on 2026-11-09 as
topical commits after `e42ec4f`; tag `fpga-demo-validated-20261008` still points
to the earlier merge `40548f0`; day-per-branch history is preserved on `origin`.
