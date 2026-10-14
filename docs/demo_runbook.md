# FPGA Fault-Diagnosis — Final Demo Runbook

Operational runbook for the live defense demo. Follow top to bottom. Every command
is copy-paste ready. Canonical numbers to cite are in §1; the demo flow is §2–§5;
recovery for network/board hiccups is §6; known limitations to state honestly are §7.

- **System:** Transformer fault classifier (SEQ=256 INT10) on **MYIR Z-Turn XC7Z020**, PYNQ 2.4 / Python 3.6.
- **Board IP:** `192.168.2.99`  ·  **SSH key:** `C:\Users\mine_pc\.ssh\id_zturn`  ·  **UART:** COM6 (115200 8N1).
- **Deployment (the only correct one):** `/home/xilinx/deployments/seq256_z7020_pynq_dma16_py36_20261008_131446` (DMA-fixed, 64 KB max transfer). The older `…_114947` build is **wrong** (16 KB DMA cap) — do not use it.
- **Repo state:** `main` @ tag `fpga-demo-validated-20261008`. Tests: 116/116 pass.
- **Classes:** `0 normal · 1 imbalance · 2 misalignment · 3 bearing_fault`.

---

## 1. Canonical numbers (single source of truth — cite these exactly)

> These are the numbers to put on slides and say out loud. Each row is a **different
> measurement** — do not merge them. The headline FPGA result is the **physical
> full-test** row.

| metric | value | what it is | source |
|---|---|---|---|
| **FPGA full-test accuracy** | **97.7684 %** (93,275 / 95,404) | the deployed SEQ=256 INT10 core on the real board, all 95,404 windows | `artifacts/results/fpga_fulltest_20261008_141335/fpga_fulltest_metrics.json` |
| **FPGA full-test macro-F1** | **95.6539 %** | same run (macro-P 95.9332 %, macro-R 95.4233 %) | same |
| Live multiclass accuracy | **95.0 %** (114 / 120) | 120-window class-balanced live run (30 per class) | `…/fpga_live_multiclass_20261008_230015/live_evidence.json` |
| Live multiclass macro-F1 | **95.0 %** (0.9499) | same live run | same |
| Live↔FPGA match | **120 / 120** (100 %) | live predictions vs the recorded FPGA predictions → deterministic core | same |
| FPGA core / DMA throughput | **≈ 61.9 inf/s** (16.15 ms / inf) | pure DMA + core round-trip | real_time_verdict.json |
| Sustained end-to-end throughput | **≈ 46.4 inf/s** | the 95,404-window physical run, averaged | fpga_fulltest_metrics.json |
| Live SSH/UI throughput | **≈ 38.9 inf/s** | full live path incl. gzip read + per-window SSH stream | live evidence |
| Real-time requirement | **48.83 windows/s** | `50000 Hz / 1024 stride` (non-overlapping) | `src/config.py` |

**Do not confuse with (different models — say which if asked):**
- Software int10 **reference** = 97.74 % (93,244) / macro-F1 95.63 % — the Python
  reference the FPGA matches within **+0.03 pp** (the FPGA is actually slightly
  *higher*, 97.7684 %). Both are correct; the board figure is the one demoed.
- Software **baseline** (2-layer float Transformer, *not* deployed) = 98.22 % test
  accuracy. This is a different, larger model from the software phase.
- Earlier **SEQ=1024** INT10 variant = 97.11 % (historical; superseded by SEQ=256).

**Temperature (safety):** full-test 49.7 → 51.7 °C max; live multiclass 48.2 → 49.2 °C
max. Watchdog stops at ≥ 75 °C (never approached).

---

## 2. Pre-demo checklist (do this ~10 min before)

1. **Power on the board**, connect Ethernet + the USB-UART (COM6) cable. Give it ~45 s to boot.
2. From the laptop, in the repo root `G:\aut\project\fpga_fault_diagnosis`:

```powershell
# a) network is up
ping 192.168.2.99

# b) SSH works + board is idle (no leftover runner / monitor)
ssh -i $HOME\.ssh\id_zturn -o StrictHostKeyChecking=no root@192.168.2.99 "uptime; pgrep -af '[f]pga_live_runner.py' || echo no-runner"

# c) software tests still green (optional, fast)
python -m unittest discover -s tests
```

3. Open a terminal you will keep for the demo. Confirm Python + Tkinter:
```powershell
python -c "import tkinter, host.fpga_result_viewer; print('UI ready')"
```

---

## 3. Preflight (always run first — read-only, no inference)

```powershell
python -m host.live_preflight --n 120
```
Expect **`RESULT: PASS -- safe to Start`** with every row `[OK ]`: SSH, the dma16
deployment, BIT/HWH present, input present, Python 3.6, driver import, DMA contract
= 65536 B, input frame = 32768 B, N valid, XADC temperature readable, no prior runner.

If any row is `[XX ]`, **do not start inference** — go to §6.

---

## 4. Demo flow A — Replay all 95,404 windows (no board needed)

Shows the full recorded FPGA result set and the metrics that match the committed evidence.

```powershell
python -m host.fpga_result_viewer --mode replay
```
On screen, demonstrate:
- **Fast eval** → accuracy **97.7684 %**, macro-F1 **95.6539 %**, confusion matrix —
  the popup states *"Matches fpga_fulltest_metrics.json within 1e-6."*
- Counters: 95,404 total, per-class predicted counts, running accuracy.
- Navigation: **Go to frame** (e.g. 0, 50000, 95403), **Prev/Next**, **Start/Pause/Resume/Reset**.
- Each row shows FRAME · CLASS · LABEL · TRUE_CLASS · TRUE_LABEL · CORRECT/WRONG · **CONF = N/A**.

Talking point: *"Replay is display-only — the PC never re-classifies; these are the
FPGA's own recorded decisions."*

---

## 5. Demo flow B — Live multiclass on the real FPGA

Real on-board AXI-DMA inference, streamed over Ethernet. Two ways — use the GUI for
the live demo, the headless CLI if you want the full evidence file.

### 5a. GUI (recommended for the defense)
```powershell
python -m host.fpga_result_viewer --mode live
```
Then in the **Live FPGA Mode** tab:
1. Set **windows = 40**, **sampling = class_balanced** (shows all four classes).
2. Click **Preflight** → wait for `PASS`.
3. Click **Start**. Watch the big banner change class, and per-inference **latency
   (~16 ms)**, **throughput**, **board temperature (~48–49 °C)**, connection status,
   and per-class counts update live.
4. Show **Pause Display** → banner text reads *"FPGA still running on board"* (honest:
   it only pauses the view). **Resume Display** → catches up.
5. **Stop** ends the on-board runner. Closing the window also cleans up (nothing is
   left running on the board).

> Note: with `class_balanced`, the first "normal" window is deep in the file, so the
> runner gzip-skips to it — expect a few seconds between the first results. The
> per-inference **latency (~16 ms)** is the real number; the wall-clock rate of a
> scattered run is **not** an inference rate.

### 5b. Headless (produces the full evidence JSON — optional)
```powershell
python -m host.live_fpga --n 120 --sample-mode class_balanced
```
Prints and writes `artifacts/results/fpga_live_test_<ts>/` with per-frame CSV/JSONL,
confusion matrix, latency percentiles, temperature and 120/120 match. (The committed
validated run is `fpga_live_multiclass_20261008_230015/`.)

**Live numbers to cite:** 120/120 executed, accuracy **95.0 %**, macro-F1 **95.0 %**,
**120/120** match the recorded FPGA predictions, latency mean **16.15 ms**
(p95 16.42), temp 48.2 → 49.2 °C.

---

## 6. Recovery — if the network/board misbehaves mid-demo

**The board state never corrupts** — the overlay and data are on the SD card; a
dropped SSH connection just ends the current stream. Recovery is always: reconnect,
confirm clean, re-run.

| symptom | fix |
|---|---|
| `ping` fails | check Ethernet cable / board power; wait for boot (~45 s). Board IP is fixed `192.168.2.99`. |
| SSH hangs or drops mid-run | just re-run the command — the board is fine. The runner is killed when SSH closes. |
| Preflight `[XX] no_prior_runner` | a leftover runner. Kill it: see command below, then re-preflight. |
| GUI froze / closed uncleanly | close it; run the "kill leftover" command; the next Start is clean. |
| "Transferred array … exceeds 16384" | wrong deployment selected. The client defaults to the correct `dma16` build — do not pass the `_114947` path. |

**Kill any leftover board processes (safe, read-only otherwise):**
```powershell
ssh -i $HOME\.ssh\id_zturn -o StrictHostKeyChecking=no root@192.168.2.99 "pkill -f '[f]pga_live_runner.py'; pkill -f '[t]emp_monitor_clean'; sleep 1; pgrep -af '[f]pga_live_runner.py' || echo CLEAN"
```

**Re-check temperature only (read-only, no overlay load):**
```powershell
ssh -i $HOME\.ssh\id_zturn -o StrictHostKeyChecking=no root@192.168.2.99 "D=/sys/bus/iio/devices/iio:device0; awk -v r=$(cat $D/in_temp0_raw) -v o=$(cat $D/in_temp0_offset) -v s=$(cat $D/in_temp0_scale) 'BEGIN{printf \"%.2f C\n\",(r+o)*s/1000}'"
```

**Fallback if the board is unavailable entirely:** run **Replay (§4)** only — it needs
no board and still shows the full 97.7684 % / 95.6539 % result set. Say clearly that
the live portion is unavailable; do not improvise numbers.

**Recoverable backup** (if the laptop repo is lost): everything needed to reproduce
is hash-verified at `D:\fpga_fault_diagnosis_release_20261008\` (deployment bundle,
BIT/HWH/XSA, full-test + live evidence, screenshots, 1.05 GB input,
`BACKUP_MANIFEST.json`, `README_DEMO.md`). See that folder's README to restore.

---

## 7. Known limitations (state these honestly in the defense)

1. **Confidence is N/A.** The INT10 hardware core emits the class decision plus 4 raw
   logits, not a calibrated probability, so the UI shows **CONF = N/A** everywhere —
   it never fabricates a number. The **class output is fully valid**; a hardware
   softmax/confidence would be a straightforward add-on, not a correctness gap.
2. **End-to-end is ~5 % below the real-time requirement.** The FPGA **core/DMA is
   real-time capable** (61.9 inf/s vs 48.83 required, +26.8 % margin; 16.15 ms vs the
   20.48 ms window period). But the **current end-to-end demo path** sustains
   ~46.4 inf/s (−5.0 %) because of the host/SSH/gzip/Python orchestration around the
   core — not the core itself. To close the gap, the data path must become a direct
   **sensor / DDR → AXI-DMA → FPGA** feed (no gzip, no per-window SSH). The core
   already has the headroom.
3. **Scope.** Validated on prepared MAFAULDA dataset windows, not a live 50 kHz
   sensor. The real-time statement is a capacity result under the configured stream,
   not a sensor-generalization claim (Stage 9 sensor path is optional / future work).

Full real-time breakdown: `artifacts/results/fpga_live_multiclass_20261008_230015/real_time_verdict.json`.

---

## 8. One-screen command summary

```powershell
cd G:\aut\project\fpga_fault_diagnosis
ping 192.168.2.99                                             # network
python -m host.live_preflight --n 120                        # must say PASS
python -m host.fpga_result_viewer --mode replay              # Flow A: all 95,404
python -m host.fpga_result_viewer --mode live                # Flow B: live multiclass (GUI)
python -m host.live_fpga --n 120 --sample-mode class_balanced  # Flow B: headless + evidence
# recovery: kill leftover runner
ssh -i $HOME\.ssh\id_zturn -o StrictHostKeyChecking=no root@192.168.2.99 "pkill -f '[f]pga_live_runner.py'; echo done"
```
