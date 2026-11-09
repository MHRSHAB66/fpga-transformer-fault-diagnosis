# FPGA Fault-Diagnosis — Defense Demo Runbook

Follow top to bottom. Every command is copy-paste ready for **PowerShell in the repo
root** `G:\aut\project\fpga_fault_diagnosis`. This version was rehearsed end to end
on the real board on 2026-10-23 (evidence: `artifacts/results/demo_rehearsal_20261023/`).
Previous version of this file: see git history.

- **Board:** MYIR Z-Turn XC7Z020, PYNQ 2.4 / Python 3.6 · IP `192.168.2.99` · key `C:\Users\mine_pc\.ssh\id_zturn` · UART COM6 (115200 8N1)
- **Deployment used by the demo (argmax in FPGA logic):** `/home/xilinx/deployments/seq256_argmax_review_d55cbdfe469f`
  - BIT sha256 `ce3c254b8679…` · HWH sha256 `eabc0163be6a…` (local copy: `artifacts/deploy/seq256_argmax_z7020_20261022/`)
  - The preflight checks both hashes against the local `MANIFEST.sha256`.
- **Input on the board:** `/home/xilinx/fpga_fulltest_20261008/fpga_testset_in.i32.gz` (95,404 prepared windows, 1,128,562,912 bytes)
- **Classes:** `0 normal · 1 imbalance · 2 misalignment · 3 bearing_fault`
- **Where the class is decided:** in FPGA logic (`pred_class` register). The board
  reports `class_source=fpga_register`; the PC never classifies.

---

## 1. Numbers to cite (each row is a different experiment — never mix them)

| what | value | experiment | evidence |
|---|---|---|---|
| Full test, argmax-in-FPGA overlay | **97.7684 %** (93,275 / 95,404), macro-F1 95.6539 % | all 95,404 test windows, physical board, 2026-10-23 | `artifacts/results/fpga_argmax_fulltest_20261023/live_evidence.json` |
| … same run vs earlier overlay | 95,404 / 95,404 identical classes | per window | same |
| … logits vs Python INT10 reference | 95,404 / 95,404 identical (all 4 logits) | per window | `artifacts/results/fpga_timing_20261023/argmax_fulltest_logits_check.json` |
| Class-balanced live, 120 windows | **95.0 %** (114 / 120), 120/120 = recorded predictions | 30 windows per class, physical board | rehearsal 2026-10-23: `artifacts/results/demo_rehearsal_20261023/headless_120/` (same result as 2026-10-08) |
| Per-window DMA + FPGA latency | ≈ 16.1–16.4 ms mean | Python-timed submit → result on the board | rehearsal `rehearsal.json` (D3, H1) |
| Sustained rate, full test | 60.85 windows/s | 95,404 prepared windows, live path | `fpga_argmax_fulltest_20261023/live_evidence.json` |
| Requirement | 48.83 windows/s (one window per 20.48 ms) | 50,000 Hz / 1,024-sample stride | `src/config.py` |

The 120-window balanced result and the full-test accuracy have **different class
distributions**; do not compare them as if one were "better".

**Do not say:** "the live demo is hard real-time", "sensor-validated", "measured power".
**You may say (scoped):** "hard real-time verified on the board-side path with
prepared windows in RAM and per-process real-time scheduling: 0 deadline misses in
2 × 30,000 windows and in one 100,000-window run (34 min)" and "CONF is
temperature-calibrated on validation data". Add: "on a non-PREEMPT_RT kernel this is a
measured bound for the tested duration, not a formal worst-case proof". For the
`ram_realtime_demo` GUI mode: "the board response (window release → FPGA class) met every
20.48 ms deadline in 30,000 windows; the SSH link and the screen update are measured
separately and are not real-time guaranteed". PREEMPT_RT was tried (one-shot boot): same
0 misses, no practical improvement; the demo board runs the stock kernel.
See §7.

---

## 2. Before the defense (~10 min)

1. Power the board, connect **Ethernet** and USB-UART; wait ~60 s.
2. Check the PC's Ethernet adapter is *connected* (a link loss looks exactly like a
   dead board; on 2026-10-23 an unplugged/power-cycled board caused this):
   ```powershell
   Get-NetAdapter -Name Ethernet | Select-Object Name, Status, LinkSpeed
   ping 192.168.2.99
   ```
3. **Close the temperature dashboard browser tab** (`tools/zturn_temp_dashboard.py`)
   during inference. Every poll is an SSH login, and each login runs Ubuntu's login
   scripts on the board (measured: up to ~36 logins/min, visible CPU load). The
   runner and an independent guard already watch the temperature.
4. Read-only preflight (no overlay load, no inference) — **must** say PASS:
   ```powershell
   python -m host.live_preflight --n 40
   ```
   Rows: SSH, Python 3.6, deployment dir, BIT/HWH present **and hash-identical**,
   input present **and complete**, driver import, DMA 65,536 B, frame 32,768 B, n
   valid, XADC readable, **temperature below 75 °C**, **no runner/guard/bench left over**.
   Measured: 3.8 s.
5. Optional: `python -m unittest discover -s tests` (~20 s, no board). With the
   system Python 3.12: 163 tests OK (3 skipped, they need Linux `fork`). The
   project `.venv` has no `torch`, so 3 training-related tests error there —
   unrelated to the demo.

---

## 3. Demo A — Replay (no board needed)

```powershell
python -m host.fpga_result_viewer
```
Window opens in < 1 s (measured 0.56 s). Tab **Replay Mode**: click **Fast eval** →
accuracy 97.7684 %, macro-F1 95.6539 %, "matches fpga_fulltest_metrics.json".
Rows show `CONF = N/A` — the note under the table explains why: the recorded
2026-10-08 run stored classes only, no logits.

Talking point: *"Replay only displays the FPGA's own recorded decisions."*

---

## 4. Demo B — Live on the real FPGA (GUI)

In the same window, tab **Live FPGA Mode (real on-board inference)**:

1. Set **windows = 40**, **sampling = class_balanced**.
2. Click **Preflight** → log ends with `preflight RESULT: PASS -- safe to Start` (5 s).
3. Click **Start**. What happens (measured in the rehearsal):
   - runner + guard copied to a new `/home/xilinx/live_runs/<id>/`, preflight
     repeated, SSH launch, FPGA overlay load → **first result after ≈ 21 s**;
   - the connection line shows `class decided by: fpga_register`;
   - the big banner shows the FPGA's class (green normal, yellow imbalance, orange
     misalignment, red bearing fault); the line below shows
     `FRAME=<id>,CLASS=<c>,LABEL=<name>,CONF=<x.xxx>`;
   - per-window latency ≈ 16 ms, board temperature ≈ 49–50 °C, per-class counts;
   - the whole 40-window run takes **≈ 3.7 min** (most of it is the board skipping
     through the compressed file between far-apart windows — not inference time).
     For a shorter demo use 20 windows.
4. Explain the purple note: since 2026-10-24 CONF is the **temperature-calibrated**
   softmax of the FPGA's own 4 logits (T = 1.425, fitted on validation only; test
   ECE 0.79 % → 0.09 %); log lines carry `[CONF calibrated]`. It estimates how often
   such predictions are right on MAFAULDA test data — not a guarantee for one window.
5. **Stop** ends the board run in ≈ 4 s; then nothing is left running.
6. Closing the window (X) also stops the board run first (measured 1.7–4.5 s).

Pause/Resume Display only pause the PC view (the label says so).

**Manually verified with the mouse (2026-10-24, by Mehrdad):** with windows = 300
and class_balanced — Preflight → PASS, Start → `class decided by: fpga_register`,
all four classes, ~16 ms latency, 46–47 °C; Stop while running → `stream ended`
after 2.9 s; X closed the window; board idle afterwards
(`artifacts/results/gui_manual_validation_20261024_180431/`). Use ≥ 300 windows
with class_balanced if you want to show Stop: with the 150-window sequential
default the whole run finishes in ~3 s.

### Demo B2 — deterministic RAM real-time mode (optional, same tab)
1. Set **live mode = ram_realtime_demo**; the banner turns green and reads
   `MODE: RAM REAL-TIME DEMO (SCHED_FIFO 50, CPU 1) -- hard-RT boundary = board response
   only; SSH + GUI measured, NOT hard real-time`.
2. **windows = 1200** (= 1,000 after the 200-window warm-up, ≈ 25 s of timed run);
   `sampling` is ignored (fixed preloaded set of 256 real windows, 64 per class).
3. **Preflight** → PASS incl. `ram_set_identity`. **Start** → first result after the overlay
   load; four classes cycle at 48.8 windows/s, latency ≈ 15.9 ms per window.
4. At the end the log shows `[board RT] accepted=1000 deadline_misses=0 …` and
   `evidence saved: artifacts/results/ram_rt_demo_gui_<ts>`.
5. **Stop** and closing the window behave as in Demo B (validated
   `artifacts/results/gui_ram_demo_validation_20261024c/`).
Say: the hard-real-time statement covers the board response only; the screen update is
≈ 73 ms after release (includes the GUI's 80 ms refresh) and is not real-time guaranteed.
Switch back to `live_sd_gzip (legacy)` for the normal live demo.

A third mode, `ram_tcp_e2e_demo` (blue banner), sends results over a direct TCP connection
and draws every frame (≈ 19–20 ms median from board release to screen). It was tested
against an end-to-end 20.48 ms deadline and **failed** (30–48 % of windows late, tails up to
135 ms): **END-TO-END HARD REAL-TIME NOT VERIFIED** (`docs/reports/e2e_hard_rt_verdict_20261024.md`).
Never say the screen update is real-time.

### Optional evidence run (headless, ≈ 4 min)
```powershell
python -m host.live_fpga --n 120 --sample-mode class_balanced
```
Writes `artifacts/results/fpga_live_test_<timestamp>/` (`live_evidence.json`,
`live_frames.jsonl/.csv`) incl. `class_source_reported_by_board`.

### Measured rehearsal timeline (2026-10-23, `demo_rehearsal_20261023/rehearsal.json`)

| step | seconds | includes |
|---|---|---|
| ping | 1.0 | 2 echoes |
| CLI preflight | 3.8 | one read-only SSH probe |
| GUI window up | 0.6 | python start + 95,404 replay records |
| Live Preflight button | 5.1 | read-only probe |
| Start → first result | 21.1 | push runner+guard, preflight, SSH, overlay load, skip to first window |
| Start → 40 results | 219.6 | whole class-balanced run (incl. the 21 s above) |
| Start → Stop (5 results) | 34.4 | start-up + 5 results + Stop (Stop itself 4.2 s) |
| Close window | 1.7 | close handler + board check |
| headless 120 | 249.7 | optional evidence run |

---

## 5. If something goes wrong

| symptom | what to do |
|---|---|
| `ping` fails | Check `Get-NetAdapter -Name Ethernet` → `Disconnected` means cable/board power. Re-seat cable, check power LED, wait ~60 s. |
| Preflight `[XX] no_other_workload_or_guard` | a previous run is still alive. Wait 10 s and re-run preflight; if it persists use the cleanup command below. |
| Preflight `[XX] temperature_below_stop` | let the board cool; do not start. |
| Preflight `[XX] bit_identity/hwh_identity` | wrong files on the board — do not run; use Replay only. |
| GUI shows `ABORT guard missing` / `independent guard exited` | the safety guard was not running → the run was refused/stopped on purpose. Start again; if repeated, use Replay only. |
| SSH drops mid-run | the board stops that run itself; preflight, then Start again. |
| board unreachable | use **Replay** only and say the live part is unavailable — never improvise numbers. |

Cleanup (only if preflight keeps reporting a leftover process):
```powershell
ssh -i $HOME\.ssh\id_zturn root@192.168.2.99 "pkill -f '^python3[.]6 -u /home/xilinx/live_runs/'; sleep 2; ps -eo pid,args | grep -E '[f]pga_live_runner|[t]emp_guard' || echo CLEAN"
```
Temperature only (read-only; the `temperature_below_stop` row shows °C):
```powershell
python -m host.live_preflight --n 1
```
(The older inline `awk` one-liner is mangled by Windows PowerShell 5.1 quoting —
checked 2026-10-23 — so it is no longer listed.)

---

## 6. Safety behaviour you can mention

- Every live run uses a separate guard process (`temp_guard.sh`): stop at ≥ 75 °C,
  on an invalid reading or missing calibration; if the guard itself dies, the run
  is stopped. Nothing is retried automatically.
- Physically tested on 2026-10-23 (`artifacts/results/gui_live_validation_20261023*/`):
  preflight refusal with a leftover process, missing guard (no inference started),
  guard dying mid-run (run stopped after 401 windows), Stop during start-up (nothing
  launched), Stop and Close mid-run (board clean afterwards).

---

## 7. Limitations to state honestly

1. **Confidence** is temperature-calibrated (validation-fitted; test ECE 0.09 %,
   uncalibrated was 0.79 %). It remains a statistical estimate on MAFAULDA data.
   Replay of the 2026-10-08 run has no logits → `N/A`.
2. **Real time:** *Hard real-time verified* only for the board-side path with
   prepared windows preloaded in RAM, released every 20.48 ms, inference loop at
   per-process SCHED_FIFO: 0 misses in 2 × 30,000 windows and in 100,000 windows
   (worst 16.56 ms); default scheduling: 73 misses in 30,000. Kernel is not
   PREEMPT_RT → measured bound for the tested duration, not a formal proof. The **live demo path you show** (gzip input from the SD
   card, SSH, Windows GUI) is **average-throughput only** (60.85/s full test; some
   windows > 20.48 ms). Evidence: `docs/reports/hard_rt_calibration_power_20261024.md`.
   The `ram_realtime_demo` mode: board response 0 misses in 30,000 accepted windows
   (max 16.50 ms); its SSH/GUI path is not hard real-time (2.8 % of PC receipts > 20.48 ms)
   — `docs/reports/ram_realtime_demo_20261024.md`. PREEMPT_RT (one-shot boot): 0 misses in
   30,000 + 30,000 + 100,000, no practical improvement; not the deployed kernel —
   `docs/reports/hard_rt_preempt_rt_comparison_20261024.md`.
3. **Power/energy not measured:** the board has no current sensor and no external
   meter was available (`docs/reports/power_measurement_blocker_20261024.md`).
4. **Scope:** prepared MAFAULDA windows; no live sensor (Stage 9 not built).

---

## 8. One-screen summary

```powershell
cd G:\aut\project\fpga_fault_diagnosis
Get-NetAdapter -Name Ethernet | Select-Object Status   # must be Up
ping 192.168.2.99
python -m host.live_preflight --n 40                   # must say PASS
python -m host.fpga_result_viewer                       # Replay tab + Live tab (Preflight, Start, Stop)
python -m host.live_fpga --n 120 --sample-mode class_balanced   # optional evidence run
```
