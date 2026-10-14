# PC-side monitor & result viewer (Stage 10/11)

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

### Preflight (non-destructive, read-only)
Before Start, run the preflight — it checks SSH, the dma16 deployment, BIT/HWH,
the staged input, Python 3.6, driver import, the DMA contract (65536 B),
input-frame size (32768 B), requested N, XADC temperature and that no runner is
already active. It NEVER loads the overlay or runs inference.
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
- Deployment: `/home/xilinx/deployments/seq256_z7020_pynq_dma16_py36_20261008_131446`
  (DMA-fixed bitstream, `C_SG_LENGTH_WIDTH=16` → one 32768-byte MM2S per window).
- Test-set input is staged on the board (`fpga_testset_in.i32.gz`, transferred over
  Ethernet); only compact result lines travel back.
- **Temperature safety** (watchdog runs in its own thread, never in the DMA path):
  `≥75 °C` → immediate stop; a single invalid reading → continue; no valid reading
  for `≥30 s` → stop. Start/max/end temperature are reported.

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

| number | ~value | what it measures |
|--------|--------|------------------|
| core/DMA probe | ~61.9 inf/s (≈16.2 ms/inf) | bounded on-board probe of the DMA + core round-trip only |
| multiclass live (this validation) | mean 16.15 ms/inf (p95 16.42, max 17.6) | 120-window class_balanced run; latency is the real inference cost |
| Live end-to-end (SSH/UI) | ~38.9 inf/s | full Live path incl. gzip read + per-line SSH/Ethernet streaming |
| sustained full-test | ~46.4 inf/s | the 95404-window physical run, averaged over ~34 min |

Note: a *wall-clock* throughput from a `class_balanced`/`strided` live run is **not**
an inference rate — scattered indices force the on-board runner to gzip-skip tens
of thousands of records to reach far windows. Use the per-window latency instead.

### Real-time verdict (source-of-truth: `src/config.py`)
Continuous-stream requirement = `sample_rate / stride = 50000 / 1024 =` **48.83
windows/s** (window period 20.48 ms; non-overlapping, no final downsampling; the
1024→256 aggregation is on-chip and does not change the window arrival rate).

- **FPGA core/DMA — REAL-TIME CAPABLE.** 61.9 inf/s (16.15 ms) vs 48.83 required:
  **+26.8 %** throughput margin, **+21 %** latency headroom (4.3 ms spare/window).
- **End-to-end current demo path — NOT real-time sustained.** Full-test 46.4 inf/s
  is **−5.0 %** below required (shortfall 2.4/s); the SSH/UI live path 38.9 inf/s
  is **−20.3 %** below. Bottleneck = host/SSH per-window streaming + on-board
  Python loop + gzip input read, **not** the inference core. A direct
  sensor→DDR→AXI-DMA feed (no gzip, no per-window SSH) is expected to close the
  ~5 % gap since the core has margin.
- **Scope:** validated on prepared dataset windows, not a live 50 kHz sensor —
  this is a capacity statement under the configured stream, not sensor
  generalization. Full detail: `artifacts/results/fpga_live_multiclass_<ts>/real_time_verdict.json`.
  (Technical note, not a university report.)

## Requirements
`replay`/Fast-eval and the Tkinter UI need only the standard library. Live mode
needs the OpenSSH client (`ssh`/`scp`, key `~/.ssh/id_zturn`) and a reachable
board. The legacy `fault_monitor --source serial` needs `pyserial`.
```
FRAME=<id>,CLASS=<class_id>,LABEL=<label>,CONF=NA
```
