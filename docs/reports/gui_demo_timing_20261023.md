# GUI validation, defense rehearsal, tail latency and confidence — 2026-10-23

Deployment under test: `/home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f`
(argmax in FPGA logic; BIT sha256 `ce3c254b86797c58…`, HWH `eabc0163be6ae333…`,
local copy `artifacts/deploy/seq256_argmax_z7020_20261022/`). Board: MYIR Z-Turn
XC7Z020 at 192.168.2.99. The board clock is not synchronized (shows 2019);
only intervals in board logs are meaningful. Nothing was committed or pushed;
the board was not flashed or rebooted by this work and its network/SSH settings
were not changed.

Evidence kinds are labelled: **PHYSICAL** (real board run), **GUI-PHYSICAL**
(real application window driving the real board), **SOFTWARE** (Python/mocks,
no board), **STATIC** (inspection only).

---

## 1. What was changed and why

| file | change | why |
|---|---|---|
| `host/live_preflight.py` | new checks: BIT/HWH sha256 vs local bundle `MANIFEST.sha256`; input byte size; temperature computed and < 75 °C; no runner/bench/guard process left (pattern bracketed so it cannot match itself) | the preflight only checked file presence and a raw XADC value; requested identity, complete input and valid temperature |
| `host/live_fpga.py` | `stream()` no longer resets a pending stop and re-checks it right before launch; `stop()` works before/during/after launch; one idempotent `_end_remote()` (kill this run's runner, let the SSH wrapper finish, terminate only if needed), also used on timeout; kill pattern scoped to the run dir and anchored on `^python3[.]6 -u`; META `class_source` parsed (optional key) and stored in evidence | **bug:** Stop/Close during push or preflight was lost and inference started anyway; the old pattern also killed the remote wrapper (SSH rc 255, no `GUARD_RC`) |
| `host/fpga_result_viewer.py` | cancel flag checked before launch; Stop runs off the Tk thread; CONF notes in Live and Replay tabs; connection line shows `class decided by: …`; `main()` split into `build_app()` (no behaviour change) | race fix; Task 4 wording; harness access to the real window |
| `host/confidence.py` | scale file → currently deployed argmax bundle (content identical to the dma16 bundle and the HLS export; test enforces it) | scale must match the deployed model |
| `tools/board/fpga_live_runner.py` | META reports `class_source`; pre-existing bug fixed (`TempMonitor._stop` shadowed `Thread._stop` → TypeError at thread end on Python 3.6); optional, **default-off** measurement/experiment paths: bounded prefetch (thread or process), kernel readahead, paced release, timing fields | evidence; bug; Task 3 experiments. Defaults keep the original overlap loop because the measurements below favour it |
| `tools/board/fpga_tail_diag.py`, `tools/run_board_guarded.py`, `tools/analyze_runner_timing.py`, `tools/summarize_timing_20261023.py`, `tools/check_live_logits_vs_reference.py`, `tools/gui_live_validation.py`, `tools/demo_rehearsal_20261023.py`, `tools/run_*_20261023.ps1` | new measurement / validation tools | Tasks 1–4 |
| tests: `test_live_stop_race.py`, `test_runner_prefetch.py` (new), `test_live_preflight.py`, `test_confidence.py` (extended) | stop at every lifecycle point, buffer order/full/empty/stop/timeout/error, readahead ranges, preflight checks, scale consistency | — |
| docs: `demo_runbook.md` (rewritten for the argmax deployment), `project_status.md`, `host/README.md`, this report, `thesis_wording_20261023.md` | — | Tasks 2, 4, 5 |

Pre-existing uncommitted work by another session (argmax build/deployment, guard
script, `fpga_argmax_fulltest_20261023/`, `live_guard_argmax_progress_20261022.md`,
etc.) was left as found.

Tests (**SOFTWARE**): `python -m unittest discover -s tests` with the system Python
3.12: **163 OK, 3 skipped** (they need Linux `fork`). The same runner test file
run **on the board** (Python 3.6, `fork`, fake driver, no FPGA): **15/15 OK**
(`artifacts/results/fpga_timing_20261023/board_unittest_runner_final.txt`). With the
project `.venv`, 3 training tests error because `torch` is not installed there
(pre-existing, unrelated).

---

## 2. Task 1 — GUI with the argmax deployment (GUI-PHYSICAL)

Harness: `tools/gui_live_validation.py` builds the same window as
`python -m host.fpga_result_viewer` and triggers the real buttons with Tk
`Button.invoke()` (same command as a click). **OS mouse input was not available**
in this session (`SetCursorPos` had no effect while another application held the
foreground), so no physical mouse click was performed. Screenshots via
`PrintWindow`. After every step the board's process table was read over SSH.

| scenario | result | observation | evidence |
|---|---|---|---|
| startup | PASS | both tabs; Live and Replay CONF notes visible | `…20261023d/` |
| Preflight button | PASS | `preflight RESULT: PASS -- safe to Start` | `…20261023c/` |
| 40 class-balanced to completion | PASS | 40 results, all four classes, labels correct, CONF = recomputed from logits for every frame, 40/40 = recorded predictions, guard END, board clean | `…20261023c/screenshots/04_run_balanced_end.png` |
| Stop mid-run ×2 | PASS | 6 results each; Stop→end 4.7 / 4.6 s; guard END; board clean; wrapper rc 1 (runner stopped), no rc 255 | `…20261023g/` |
| Stop during start-up | PASS | "stop requested before launch; nothing started"; no guard log on the board | `…20261023d/` |
| Preflight fault (harmless `sleep` decoy named `temp_guard.sh`) | PASS | Start refused: `preflight failed: no_other_workload_or_guard`; no runner launched | `…20261023d/` |
| Missing guard | PASS | `ABORT guard missing`, rc 1, no inference | `…20261023d/` |
| Guard dies mid-run (real guard, exits after 10 readings) | PASS | 401 results, then `ABORT independent guard exited`; board clean | `…20261023f/` |
| Close window mid-run | PASS | close handler 2.9 s; guard END; board clean | `…20261023g/` |

Run `…20261023c` also contains four FAILs: harness bugs (decoy cleanup missed,
event read before rendering). The GUI behaved correctly (screenshot `07`); the
scenarios were rerun in `d/f/g`. Aborted harness starts `…20261023`/`…b` never
touched the board (`NOTE_ABORTED.txt`). GUI render cost per event: mean 0.5–0.9 ms,
max 4.1 ms. The UI never classifies; the board reports `class_source=fpga_register`.

**Manual mouse pass — done 2026-10-24 (GUI-PHYSICAL, by Mehrdad).** Window opened
with `python -m tools.gui_manual_recorder` (same window as
`host.fpga_result_viewer --mode live`; observe-only, it clicks nothing and logs
real Tk `<ButtonPress-1>` events, which `invoke()` never produces). Read-only
checks before: board idle, CLI preflight PASS (45.9 °C). Settings 300 windows,
class_balanced. Mehrdad clicked with his own mouse (confirmed by him):
Preflight → PASS; Start → `class decided by: fpga_register`, all four classes,
157 results before Stop (157/157 = recorded board predictions, 153/157 correct),
latency 15.9–18.0 ms, temperature 46.4 → max 47.2 °C; **Stop while the run was
active** → `stream ended` 2.9 s later; window closed with X (no run active).
Post-close SSH check: BOARD_IDLE, guard log `guard END (workload exited)`.
Evidence: `artifacts/results/gui_manual_validation_20261024_180431/`. An earlier
pass (`…_180108`) used the 150-window default; its run finished 0.1 s before the
Stop click, so it is superseded.

---

## 3. Task 2 — timed rehearsal (GUI-PHYSICAL + PHYSICAL)

`artifacts/results/demo_rehearsal_20261023/rehearsal.json`, all 10 steps PASS,
515.7 s total:

| step | s | includes |
|---|---|---|
| ping | 1.03 | 2 echoes |
| CLI preflight | 3.76 | one read-only SSH probe |
| GUI window up | 0.56 | interpreter already running; viewer + 95,404 replay records |
| Replay fast-eval | 0.06 | re-score of the recorded 95,404 predictions: accuracy 0.977684, macro-F1 0.956539, 0 discrepancies vs the metrics file |
| Live Preflight button | 5.06 | read-only probe |
| Start → first result | 21.13 | push runner+guard, preflight, SSH, overlay load, gzip skip to the first index |
| Start → 40 results | 219.55 | complete class-balanced run (incl. the 21 s); 39/40 correct, 40/40 = recorded, latency mean 16.4 ms, max 50.1 °C |
| Start → Stop | 34.35 | start-up + 5 results + Stop (4.2 s) |
| Close window | 1.71 | close handler + board check |
| headless 120 | 249.66 | `python -m host.live_fpga --n 120 --sample-mode class_balanced`: 95.0 %, 120/120 = recorded, `class_source=fpga_register`, max 50.1 °C |

Instructions: `docs/demo_runbook.md`.

---

## 4. Task 3 — timing

Requirement: 50,000 Hz / 1,024 stride = **48.83 windows/s**, period 20.48 ms.

### 4.1 What the existing latency metric means (STATIC)
In the default (overlap) loop `lat_ms` = submit of the MM2S frame → logits
collected, and it **includes the read/decompression of the next window when that
read outlasts the core**. It is board-internal; it is neither sensor-to-result
latency nor PC arrival time. The 143 values > 20.48 ms (max 664.9 ms) of the
2026-10-23 full test are values of this metric.

### 4.2 Attribution (PHYSICAL, diagnostics without inference or without file I/O)
| run | finding |
|---|---|
| `diag_read_full` (read only, 95,404) | 191 reads > 10 ms, max 815 ms; CPU time never > 3.6 ms; slow reads: wall − CPU mean 83 ms, 157/191 with voluntary waits → **blocking I/O on the SD card**, not decompression |
| `diag_dma_ram_b` (DMA+FPGA only, 20,000 from RAM) | 36 (0.18 %) > 20.48 ms, max 42.3 ms, **all 36 with involuntary preemption**; CPU ≤ 16.9 ms → **OS preemption of the busy-polling DMA wait**; 20,000/20,000 classes = recorded |
| board `auth.log` | SSH logins from this PC up to ~36/min during parts of the day (a temperature-dashboard poll); every login runs Ubuntu login scripts (`lsb_release` at 20–28 % CPU observed) → **external CPU load**; logged per run in `timing_runs_summary.json` |
| network/PC (S1–S4) | PC receive − board emit, median 0.56–0.76 ms, p99 11–114 ms |
| GUI render | 0.5–0.9 ms mean per event |

### 4.3 Before/after (PHYSICAL; full test unless noted; all predictions = recorded)
| run | loop | ext. logins/min | rate /s | intervals > 20.48 ms | > 100 ms | max ms |
|---|---|---|---|---|---|---|
| A | original overlap | 5.2 | 55.07 | 871 | 309 | 1,502.6 |
| B | + reader **thread** + queue 128 | 5.1 | 52.09 | 4,228 | 264 | 4,168.7 |
| C | thread, paced 48.828 Hz | 22.2 | 47.32 (backlog, response up to 63.6 s) | 26,682 | 134 | 3,180.7 |
| B2 | + reader **process** + queue 128 | ≥36 (partly covered) | 43.30 | 35,063 | 148 | 3,205.3 |

A–C/B2 wrote output and predictions to the board's SD card (same disk as the
input) and were polled over SSH every 10 s, so their absolute numbers are worse
than the live path; **only A vs B is a clean comparison** (same setup, same load).
Result: **both queue designs are regressions on this dual-core board** (thread:
GIL starvation by PYNQ's busy-wait, mean read 18.7 ms instead of ~2 ms; process:
the DMA wait itself slowed: `lat_ms` p90 25.0 / p95 29.9 ms vs p95 16.2 ms in A; B2 also ran under heavier external SSH load, so this is indicative, not a clean comparison). They remain off by default.

Live-path runs (output streamed over one SSH session, no SD writes), alternating,
30,000 contiguous windows each, ~25 external logins/min:

| run | variant | rate /s | > 20.48 ms | > 100 ms | max ms |
|---|---|---|---|---|---|
| S1 | original | 58.41 | 531 | 23 | 713.8 |
| S2 | + 32 MB readahead | 59.42 | 509 | 0 | 64.3 |
| S3 | original | 59.36 | 520 | 0 | 56.7 |
| S4 | + 32 MB readahead | 59.38 | 467 | 0 | 51.6 |

The long stalls appear only in S1, the first run; later runs read the same
~350 MB that was then in the page cache (S3 without readahead also has none). So
**no benefit of readahead is demonstrated**; it stays off by default. All runs:
30,000/30,000 = recorded, guard rc 0, board clean, max 51.3 °C over the day, no
emergency stop.

### 4.4 Conclusion
- Sustained average on prepared windows is above the requirement (live path
  58–59 /s here; 60.85 /s in the clean full test).
- **Hard per-window deadline: not demonstrated.** ~1.6–1.8 % of intervals exceed
  20.48 ms even with a warm cache (20–65 ms, consistent with preemption/CPU load);
  cold-cache SD stalls reach ~0.7–1.5 s.
- Not proven: behaviour with a real sensor stream; behaviour without external SSH
  load in the live path (the clean 2026-10-23 full test suggests fewer misses).
- Plausible next steps (not done; need board/system changes): real-time scheduling
  priority or core isolation for the DMA loop; interrupt-driven (non-polling) DMA
  wait; input from RAM/sensor instead of the SD card; stopping the SSH polling
  during demos (already in the runbook).

### 4.5 Incidents
- `diag_dma_ram_c_extload`: Ethernet link loss at 00:48:19 board time, then the
  board restarted (no kernel panic logged; `last` shows "crash" = power loss);
  temperature 48.9 °C before. Not retried. `…/diag_dma_ram_c_extload/OUTAGE_EVIDENCE.txt`.
- Launch with `setsid -f` failed (not in util-linux 2.31); nothing started.

---

## 5. Task 4 — confidence
- Live logits of the argmax full test equal the Python INT10 reference (export
  scales) on **95,404/95,404** windows; class = argmax for all; recorded CONF =
  recomputed for all (`fpga_timing_20261023/argmax_fulltest_logits_check.json`).
- Scale: `scales.logits` = 0.10779884107882078 in the argmax bundle, the dma16
  bundle and the HLS export (test-enforced).
- UI: Live note "uncalibrated: NOT a guaranteed probability of being correct …
  CONF=NA whenever no logits were received"; Replay note "CONF=N/A: the recorded
  2026-10-08 run stored classes only". No confidence is inferred from a class.
- Calibration numbers are the earlier measured ones (ECE 0.71 % val / 0.79 %
  test); no calibration was fitted. Thesis wording: `thesis_wording_20261023.md`.

---

## 6. Commands
```powershell
python -m host.live_preflight --n 40
python -m tools.gui_live_validation --out artifacts/results/gui_live_validation_<id>      # system Python with Pillow
python -m tools.demo_rehearsal_20261023 --out artifacts/results/demo_rehearsal_<id> --headless
python -m tools.run_board_guarded --tag <tag> --out <dir> --files tools/board/fpga_tail_diag.py -- python3.6 -u "{DIR}/fpga_tail_diag.py" read "{IN}" 95404 "{DIR}/out"
powershell -File tools/run_stream_ab_20261023.ps1
python -m tools.summarize_timing_20261023 --logins artifacts/results/fpga_timing_20261023/board_ssh_logins_authlog.txt --out <json>
python -m tools.check_live_logits_vs_reference artifacts/results/fpga_argmax_fulltest_20261023/live_frames.jsonl --out <json>
```

## 7. Open items
1. Hard real-time not demonstrated (§4.4).
2. Thesis source (outside this repository) not edited; wording prepared.
(The manual mouse pass of §2 was completed on 2026-10-24.)
