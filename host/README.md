# PC-side monitor & result viewer (Stage 10/11)

Live Start and the CLI now deploy the runner and independent temperature guard
into a fresh `/home/xilinx/live_runs/<unique-id>/` directory on each run. The
client automatically runs the read-only preflight before inference. The guard
stops the runner at 75 °C or immediately on an invalid XADC reading or invalid
calibration; its log stays in that run directory. `--no-push` is unavailable
because the legacy files do not provide this independent guard. The existing
runner also checks its own temperature, and no aborted run is retried.

Displays FPGA fault-classification results on the Windows PC. **Display only** —
the PC never re-classifies; the FPGA is the source of truth. Class map is fixed:
`0 normal · 1 imbalance · 2 misalignment · 3 bearing_fault`.

## Files
- `frame_protocol.py` — single source of truth for the wire frame
  `FRAME=<id>,CLASS=<id>,LABEL=<label>,CONF=<conf|NA>`, with parsing, class/label
  validation, and wrap-aware dropped-frame detection.
- `replay_data.py` — loads `fpga_pred.txt` / `labels.txt` / `meta.json`, pairs
  prediction↔ground-truth, and computes accuracy / macro·weighted P-R-F1 /
  confusion matrix. Pure stdlib; checked against `fpga_fulltest_metrics.json`.
- `fpga_result_viewer.py` — the Tkinter UI with **two independent modes** (tabs):
  **Replay Mode** and **Live FPGA Mode**.
- `live_fpga.py` — PC client that drives REAL on-board AXI-DMA inference over
  SSH/Ethernet (launches `tools/board/fpga_live_runner.py`, parses its stream).
- `fault_monitor.py` — the earlier minimal live/mock/file monitor (kept for the
  `mock`/`serial`/`file` FRAME-line sources; superseded by the viewer for demos).

## Run

### The two-mode viewer (recommended)
```bash
# Both tabs (Replay + Live FPGA):
python -m host.fpga_result_viewer

# A single mode:
python -m host.fpga_result_viewer --mode replay
python -m host.fpga_result_viewer --mode live
```

**Replay Mode** — replays all 95404 recorded FPGA results. Shows the latest class
(big, colour-coded), FRAME / CLASS / LABEL / TRUE_CLASS / TRUE_LABEL /
CORRECT·WRONG / CONF (`N/A` — the hardware emits no confidence), a recent-results
table, counters (current/total, correct, incorrect, running accuracy, per
predicted class), and controls: Start / Pause / Resume / Stop / Reset / Prev /
Next / speed select / go-to-frame. **Fast eval** scores all 95404 without
rendering each frame and reports accuracy, macro P/R/F1 and the confusion matrix,
verified against `fpga_fulltest_metrics.json`.

**Live FPGA Mode** — runs real inference **on the board** and displays what the
FPGA returns. The on-board runner feeds each test window through the AXI-DMA
`transformer_top` core and streams back one result per window over SSH/Ethernet.
Shows latest class, per-inference latency, throughput, board die temperature
(start/last/max), connection status and error count, with Start/Pause/Resume/Stop.
Classification is 100% on the FPGA — the PC only sends/triggers and displays.

### Live FPGA — headless (produces versioned evidence)
```bash
python -m host.live_fpga --n 150                         # first 150 windows (sequential)
python -m host.live_fpga --n 120 --sample-mode class_balanced   # multiclass demo
python -m host.live_fpga --n 200 --sample-mode strided          # spread across the set
```
Runs ≥100 real windows, then cross-checks the live FPGA predictions against
ground truth AND against the independently-recorded `fpga_pred.txt` (same core,
must match), by REAL dataset index. Reports latency, throughput and
start/max/end temperature. Output: `artifacts/results/fpga_live_test_<ts>/`.

### Multiclass Live demo (software-prepared; NOT yet hardware-run)
The first ~150 test windows are nearly all `bearing_fault`. `host/live_sampling.py`
builds a deterministic, class-balanced (or strided/explicit) list of REAL dataset
indices so a future demo shows all four classes. Classification still happens on
the FPGA; the PC only chooses which windows to send and shows the true dataset
frame id. The on-board runner accepts an indices file; the GUI/`live_fpga`
`--sample-mode` wire it through. **This selection path has NOT been executed on
hardware yet** — it is unit-tested in software (`tests/test_live_sampling.py`).
Future hardware command (when a board run is authorised):
```bash
python -m host.live_fpga --n 120 --sample-mode class_balanced
```

### Deterministic demo mode `ram_realtime_demo` (separate; legacy mode unchanged)
Select **live mode = ram_realtime_demo** in the Live tab (the banner always states the active
mode and its scope), or run headless:
```bash
python -m host.ram_rt_demo --n 1200      # 1,000 windows after the 200-window warm-up
python -m host.ram_rt_demo --n 30200     # the 30,000-window hard-RT criterion run
python -m tools.verify_ram_rt_demo artifacts/results/ram_rt_demo_<ts>
```
The board preloads a documented finite set (`artifacts/ram_demo_sets/class_balanced_256_v1/`,
256 real windows, 64 per class, sha256-checked by preflight) into RAM, releases one window every
20.48 ms and runs only its timed thread with SCHED_FIFO 50 on CPU 1; results reach SSH through a
non-blocking pipe + separate publisher process (a stalled PC drops display lines, counted, never
the board timing). Same guard, preflight, run-dir and Stop/Close behaviour as the live mode;
`sampling` is ignored. Two measurements: **board response** (release → FPGA class ready, the
hard-RT boundary) and **end-to-end** (release → PC receipt / GUI update, clock-synced over SSH,
measured only). Evidence and numbers: `../docs/reports/ram_realtime_demo_20261024.md`.

### End-to-end TCP mode `ram_tcp_e2e_demo` (separate; other modes unchanged)
Live tab → **live mode = ram_tcp_e2e_demo**. Same RAM set, runner policy (FIFO 50/CPU 1), guard,
preflight (+ port 50555 free) and Stop/Close; results travel over ONE persistent TCP connection
(PC → board:50555, TCP_NODELAY, 57-byte frames with sequence id and board timestamps), not SSH;
every frame is drawn by a 1 ms Tk tick and its render time stamped. Bounded queues with overflow
recorded by sequence id (board pipe, PC display queue); counters shown in the tab. Verifier:
`python -m tools.verify_e2e_20261024 new artifacts/results/e2e_demo_gui_<ts>`.
**Result: END-TO-END HARD REAL-TIME NOT VERIFIED** (30–48 % of windows later than 20.48 ms
after release; the board stage alone uses ≈ 15.9 ms) — `../docs/reports/e2e_hard_rt_verdict_20261024.md`.

### Preflight (non-destructive, read-only)
Before Start, run the preflight — it checks SSH, the deployment in
`host/live_fpga.py` (argmax overlay since 2026-10-22), BIT/HWH present **and
sha256-identical to the local bundle `MANIFEST.sha256`**, the staged input present
**and complete (byte size)**, Python 3.6, driver import, the DMA contract
(65536 B), input-frame size (32768 B), requested N, XADC readable, **temperature
below 75 °C**, and that **no runner, bench or guard process** is left over. It
NEVER loads the overlay or runs inference. `LiveFpgaClient.stream()` repeats it
before every launch and refuses to start on any failure.

Stop/Close: `LiveFpgaClient.stop()` works before launch (prevents it), during
the run (kills only this run's runner, matched by its unique run directory and
anchored on the python command line; the guard then exits), and after. A Stop
during push/preflight never launches inference (`tests/test_live_stop_race.py`;
physically checked 2026-10-23, `artifacts/results/gui_live_validation_20261023*/`).
```bash
python -m host.live_preflight --n 150         # or the "Preflight" button in the Live tab
```

## Live path — board architecture
```
PC (Windows)                         MYIR Z-Turn XC7Z020 (PYNQ 2.4, 192.168.2.99)
  host.live_fpga  --ssh/Ethernet-->  fpga_live_runner.py
    (sends N, reads results)           ├─ TempMonitor thread (XADC, independent of DMA)
                                       └─ FaultClassifier -> AXI-DMA -> transformer_top
  UI shows class/latency/temp  <--stdout(FRAME lines)--  (real FPGA inference)
```
- Current live deployment: `/home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f`
  (DMA-fixed bitstream, `C_SG_LENGTH_WIDTH=16` → one 32768-byte MM2S per window).
- Test-set input is staged on the board (`fpga_testset_in.i32.gz`, transferred over
  Ethernet); only compact result lines travel back.
- **Temperature safety:** a separate guard process stops the runner at `≥75 °C`
  or on an invalid XADC read. The runner also has its own temperature thread and
  stops after 30 seconds without a valid reading. Start/max/end temperatures are
  reported; the independent guard log is retained in the versioned run directory.

## USB-UART boot test (Part 1 tooling)
`../tools/uart_boot_capture.ps1` captures the serial console (COM6, 115200 8N1)
and can trigger a single `sync; reboot` over SSH, saving the full boot log +
timing. Evidence: `../artifacts/results/uart_boot_test_<ts>/`.

## Accuracy & throughput numbers (what each one means)
Deployed FPGA full-test (95,404 windows): **accuracy 97.7684 %, macro-F1 95.6539 %**;
live multiclass (120-window class-balanced): **95.0 % / 95.0 %**, 120/120 match the
recorded FPGA predictions. Canonical table + demo flow: `../docs/demo_runbook.md`.
The throughput/latency figures below are measured from different points in the stack
— do not conflate them:

| number | value | what it measures |
|--------|-------|------------------|
| per-window DMA+core latency | 16.15 ms mean (p95 16.42) | Python-timed send→receive on the board (1/latency = 61.9/s is a derived bound, not a throughput run) |
| board-only, all 95,404 windows (2026-10-22) | 60.94 inf/s overlapping loop · 53.83 inf/s serial loop | on-board loop, no per-window SSH; predictions identical to the full test |
| end-to-end board → SSH → PC parser, 10,000 windows (2026-10-22) | 60.30 / 58.72 inf/s current runner · 51.07 / 52.57 inf/s previous runner | two runs each; Tk GUI rendering not included |
| 2026-10-08 full-test average | 46.4 inf/s | historical; ~53 inf/s until ~41k windows then ~40 inf/s; did not reproduce on 2026-10-22 |
| 2026-10-08 "live SSH/UI 38.9 inf/s" | — | 150-window run, 3.86 s wall including startup; not a sustained rate (superseded) |

Note: a *wall-clock* throughput from a `class_balanced`/`strided` live run is **not**
an inference rate — scattered indices force the on-board runner to gzip-skip tens
of thousands of records to reach far windows. Use the per-window latency instead.

### Real-time verdict (source-of-truth: `src/config.py`)
Continuous-stream requirement = `sample_rate / stride = 50000 / 1024 =` **48.83
windows/s** (window period 20.48 ms; non-overlapping, no final downsampling; the
1024→256 aggregation is on-chip and does not change the window arrival rate).

- **Sustained average on prepared windows — met** (2026-10-22): board-only 60.94
  inf/s over 95,404 windows; end-to-end 60.30 / 58.72 inf/s over 10,000 windows.
- **Live path (this host tool: gzip/SD input, SSH, Windows GUI): AVERAGE-THROUGHPUT
  VERIFIED, HARD REAL-TIME NOT VERIFIED.** ~0.8 % of windows exceeded 20.48 ms
  (input-read stalls up to ~0.85 s, OS preemption).
- **Board-side RAM-backed workload (`tools/board/fpga_rt_acceptance.py`, per-process
  SCHED_FIFO 50 + CPU 1): HARD REAL-TIME VERIFIED for the tested durations only** —
  0 misses in 2 × 30,000 and 1 × 100,000 windows (2026-10-24, worst 16.56 ms); the
  default scheduler had 73 misses in 30,000. Kernel 4.14.0-xilinx is not PREEMPT_RT,
  so this is a measured bound, not a formal worst-case proof
  (`../docs/reports/hard_rt_longrun_preempt_rt_20261024.md`).
- **`ram_realtime_demo` mode (this host tool, 2026-10-24): board response HARD REAL-TIME
  VERIFIED for the tested duration** — 0 misses in 30,000 accepted windows (max 16.50 ms),
  FIFO 50/CPU 1. Its SSH transport and GUI update are **measured, NOT hard real-time**
  (PC receipt p50 16.7 ms, 2.8 % > 20.48 ms, max 85 ms; GUI update p50 ≈ 73 ms incl. the 80 ms
  poll) (`../docs/reports/ram_realtime_demo_20261024.md`).
- **PREEMPT_RT (one-shot boot, 2026-10-24):** same workload 0 misses in 30,000 + 30,000 +
  100,000 windows, tail ≈ 0.05–0.1 ms lower — no practical improvement; the board stays on the
  stock kernel (`../docs/reports/hard_rt_preempt_rt_comparison_20261024.md`).
- **End to end (board release → Windows GUI render), 2026-10-24:** existing RAM mode and the
  new `ram_tcp_e2e_demo` mode both **END-TO-END HARD REAL-TIME NOT VERIFIED** against the frozen
  criterion (`../docs/reports/e2e_hard_rt_verdict_20261024.md`).
- **Scope:** inputs are INT10 codes prepared offline; no live 50 kHz sensor,
  no on-board normalization. Full detail:
  `../artifacts/results/fpga_realtime_bench_20261022/realtime_verdict.json` and
  `../docs/reports/consolidated_results_20261022.md` §3.

### CONF field
Since 2026-10-24 the Live CONF is **temperature-calibrated** when
`artifacts/calibration/conf_temperature_v1/temperature.json` exists and its
`s_logits` equals the deployed scale (T = 1.4248, fitted on the validation split
only; test ECE 0.79 % → 0.09 %). Events carry `conf` (displayed),
`conf_uncalibrated` and `conf_kind`; the GUI labels which one is shown. Without the
artifact the uncalibrated value is shown and labelled. Report:
`../docs/reports/hard_rt_calibration_power_20261024.md`.

Live result lines from the current runner carry `logits=a,b,c,d`; the PC sets
`CONF` = softmax probability of the FPGA's class from those logits
(`host/confidence.py`; calibrated as above when the artifact applies, otherwise the
raw softmax with measured ECE 0.71 % val / 0.79 % test). Lines
without logits (older runner) and Replay of the 2026-10-08 run keep `CONF=NA`.

## Requirements
`replay`/Fast-eval and the Tkinter UI need only the standard library. Live mode
needs the OpenSSH client (`ssh`/`scp`, key `~/.ssh/id_zturn`) and a reachable
board. The legacy `fault_monitor --source serial` needs `pyserial`.
```
FRAME=<id>,CLASS=<class_id>,LABEL=<label>,CONF=NA
```
