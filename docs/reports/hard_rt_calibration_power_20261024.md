# Hard real-time, calibrated confidence, power — 2026-10-24

Deployment: argmax-in-FPGA overlay `/home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f`
(BIT `ce3c254b…`, HWH `eabc0163…`, verified by every preflight). Older results
(dma16 overlay, PS argmax) are not mixed in. Labels: **PHYSICAL** = measured on the
board, **SOFTWARE** = computed on the PC, **BLOCKED** = not measurable here.
Nothing committed/pushed; no flash, reboot, network or SSH configuration change.
The temperature dashboard was closed (board `auth.log`: 0–1 logins per run).

## Summary of verdicts

| item | verdict |
|---|---|
| 1. Hard real-time (declared criterion, board-side, RAM-backed arrivals) | **HARD REAL-TIME VERIFIED** — 0 misses in 30,000 windows, twice (R1, R2) with per-process `SCHED_FIFO`; baseline (R0) had 73 misses |
| … current live demo path (SD-card gzip input, SSH, PC GUI) | **AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME NOT VERIFIED** (unchanged; see 2026-10-23 report) |
| 2. Calibrated confidence | **Calibration improves validation and test** (test ECE 0.79 % → 0.09 %); integrated in the live display; physical logits = reference |
| 3. Power / energy | **BLOCKED** — no current/power telemetry on the board, no instrument (`power_measurement_blocker_20261024.md`) |

---

## 1. Hard real-time (PHYSICAL)

**Criterion** (written before the runs, not changed afterwards):
`docs/reports/hard_rt_acceptance_criterion_20261024.md`. One window arrives every
20.48 ms (release_k = t0 + k·P); `response_k = result available − release_k`
(includes waiting for earlier windows, copy, DMA send, FPGA, DMA receive, class
read); no skipping; warm-up 200; acceptance interval 30,000 windows; pass = 0
responses > 20.48 ms + all classes = recorded predictions + guard rc 0.

**Program:** `tools/board/fpga_rt_acceptance.py` (deployed driver, one intact
32,768-byte MM2S frame per window, class from the FPGA `pred_class` register).
Arrivals: the first 4,096 test windows (dataset indices 0–4095) preloaded into RAM
(load 8.0 s, process max RSS 17 MB → 149 MB of the board's 1 GB), released in
order 0…4095, 0…, for 30,200 releases. Each run under `tools/run_board_guarded.py
--stream` (preflight, new remote dir, independent guard 75 °C, one SSH session).

| run | scheduling | acceptance windows | misses | p50 | p99 | p99.9 | max | involuntary ctx sw. | classes = recorded | logits = reference | guard | max °C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | SCHED_OTHER (default) | 30,000 | **73** | 15.83 | 16.92 | 24.18 | 43.86 ms | 4,341 | 30,200/30,200 | 30,200/30,200 | 0 | 47.4 |
| R1 | SCHED_FIFO 50, CPU 1 | 30,000 | **0** | 15.82 | 16.26 | 16.40 | 16.55 ms | 0 | 30,200/30,200 | 30,200/30,200 | 0 | 47.4 |
| R2 (repeat of R1) | SCHED_FIFO 50, CPU 1 | 30,000 | **0** | 15.82 | 16.26 | 16.41 | 16.51 ms | — | 30,200/30,200 | 30,200/30,200 | 0 | 47.1 |

Independent recomputation from raw timestamps: `artifacts/results/hard_rt_20261024/verification.json`
(R1/R2 `hard_rt_pass: true`, R0 `false`). Margin in R1/R2: worst response 16.55 ms
vs 20.48 ms (3.9 ms spare).

**What was incompatible with hard real time (measured):**
1. SD-card I/O on the gzip input — blocking waits up to ~0.8 s (2026-10-23 read
   diagnostic). Removed from the timed path by RAM-backed arrivals.
2. OS preemption of the busy-polling DMA loop — R0 shows it alone causes 73 misses
   (max 43.9 ms) even without SD I/O and without external SSH logins.
3. External SSH logins (dashboard polling) — extra CPU load (2026-10-23); absent here.

**Improvement implemented (smallest justified):** per-process `SCHED_FIFO` (prio 50)
and CPU affinity {1} for the inference loop only (`os.sched_setscheduler`,
`os.sched_setaffinity`; ends with the process; no global setting changed; kernel
RT throttling left at 95 %; the temperature thread keeps normal priority; spin
before each release limited to 0.5 ms → busy share ≈ 80 %). With it, involuntary
context switches dropped from 4,341 to 0 and the tail collapsed to ≤ 16.6 ms.
Options inspected and not used: queue-based prefetch (measured worse on
2026-10-23), interrupt-driven DMA (not exposed by the PYNQ 2.4 driver; would need
driver/overlay work), hardware FIFO (would change the IP).

**Scope / limits (declared in advance):** board-side path with prepared windows in
RAM, not a live sensor; `SMP PREEMPT` kernel (not PREEMPT_RT) → empirical bound over
2 × 10.3 min, not a formal worst-case proof; 4,096 distinct windows cycled. The
live GUI/demo path (gzip from SD, SSH to the PC, Windows GUI) is **not** covered and
keeps its 2026-10-23 verdict. PC arrival and GUI rendering were measured on
2026-10-23 (PC extra delay median ~0.6 ms, GUI render < 1 ms/event) but have no
real-time guarantee.

Tests: `tests/test_rt_acceptance.py` (order, frame integrity, no skipping,
late-window counting, DMA timeout, SIGTERM, SCHED_FIFO path) — PC 4 OK + 1 skipped,
board 5/5 OK (`artifacts/results/hard_rt_20261024/board_unittest_rt_acceptance.txt`).
Guard failure, Stop/Close cleanup and orphan checks of the wrapper were covered
physically on 2026-10-23.

Commands:
```powershell
powershell -File tools/run_hard_rt_20261024.ps1                  # R0, R1, R2
powershell -File tools/run_hard_rt_20261024.ps1 -Only R1_fifo50_cpu1
python -m tools.verify_hard_rt_20261024                           # independent check
```

---

## 2. Calibrated confidence (SOFTWARE fit, PHYSICAL logit check)

- Method: temperature scaling, `conf = max softmax(s_logits · logits_int / T)`,
  **T = 1.424821**, fitted on the **validation split only** (195 files, 47,580
  windows; Python INT10 reference logits with the deployed export scales) by
  minimising validation NLL. T is a positive scalar → the FPGA's class never changes.
- Frozen artifact: `artifacts/calibration/conf_temperature_v1/temperature.json`
  (T, s_logits = 0.10779884107882078, input hashes, command); metrics:
  `evaluation.json`, `evaluation_test_blocks.json`.

| split | | ECE (15 bins) | NLL | Brier | MCE | mean CONF | accuracy |
|---|---|---|---|---|---|---|---|
| validation | uncalibrated | 0.71 % | 0.0563 | 0.0291 | 0.345 | 98.81 % | 98.10 % |
| validation | calibrated | 0.11 % | 0.0518 | 0.0284 | 0.178 | 98.12 % | 98.10 % |
| **test (untouched)** | uncalibrated | 0.79 % | 0.0628 | 0.0336 | 0.130 | 98.55 % | 97.77 % |
| **test (untouched)** | calibrated | **0.09 %** | **0.0584** | **0.0329** | 0.123 | 97.76 % | 97.77 % |

Test reliability, 0.53–0.93 bins: uncalibrated over-confident by 2–9 points,
calibrated within ~1 point. ECE lower in 9 of 10 contiguous test blocks.

- Display: `host/confidence.display_confidence` uses T only if the artifact exists
  and its `s_logits` equals the deployed scale; otherwise uncalibrated, labelled.
  Live frame `CONF` = calibrated value; events/evidence also carry
  `conf_uncalibrated` and `conf_kind`; GUI note and log tag `[CONF calibrated]`.
  Replay without logits stays `N/A`; no value is ever derived from a class alone.
- Physical check: in R0/R1/R2 (90,600 inferences of windows 0–4095) every physical
  logit equals the reference (all 4 values), and calibrated CONF recomputed
  independently from the physical logits equals the reference value (max abs
  difference 0.0 on the sampled windows).
- Limits: calibrated on MAFAULDA validation windows; window-level bins are not
  independent (windows of one file are correlated); a calibrated score is an
  average-frequency statement, not a guarantee for one window.

Commands: `python -m tools.fit_confidence_temperature --out artifacts/calibration/conf_temperature_v1`
(refuses to overwrite); tests `tests/test_confidence.py`.

---

## 3. Power / energy — BLOCKED

See `docs/reports/power_measurement_blocker_20261024.md` and
`artifacts/results/power_inspection_20261024/board_inspection.txt`: XADC gives die
temperature and six rail voltages only; no current, power, energy, hwmon, PMBus or
INA-type sensor; only I²C device is the HDMI transmitter; no USB meter; no external
instrument available. No power or energy number is reported.

---

## Remaining work
Required for the thesis claims as stated: none beyond honest scoping above.
Optional:
- bring the RT configuration into the live demo runner (needs RAM/sensor input
  instead of gzip from SD, and per-process SCHED_FIFO) and re-run the criterion on
  that path;
- longer or PREEMPT_RT-kernel runs for stronger worst-case evidence;
- power/energy with an inline DC meter or logging bench supply (procedure in the
  blocker report).
