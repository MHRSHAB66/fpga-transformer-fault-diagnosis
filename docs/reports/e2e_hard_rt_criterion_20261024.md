# End-to-end hard-real-time acceptance criterion — FROZEN 2026-10-24 (before implementation and before any end-to-end test)

Scope: decide whether a complete demo path, RAM-preloaded window on the board → class on the
Windows GUI, meets a hard deadline. Applies to (a) the existing `ram_realtime_demo` mode
(measured as-is, no behaviour change) and (b) the new end-to-end mode to be built. Stock kernel
4.14.0-xilinx, deployed argmax overlay (BIT ce3c254b…, HWH eabc0163…). A negative result is valid.

## 1. Timing boundary
- **Start (board clock, CLOCK_MONOTONIC via `time.perf_counter`):** the scheduled release of window k,
  `t_rel(k) = T0 + k·P`, P = 1024/50000 s = 20.48 ms, from the RAM-preloaded set (same definition as the
  validated board-side criterion).
- **End (Windows clock, QueryPerformanceCounter via `time.perf_counter`):** `t_render(k)` = the instant
  `update_idletasks()` returns in the Tk main thread after the GUI widget showing window k's frame ID
  and class was updated, i.e. Tk has drawn that frame ID into the application window.
  Not covered (cannot be measured without an optical sensor): Windows DWM composition and monitor
  scan-out. Any claim is therefore "to Tk render completion"; photon-level visibility is NOT verified.
- **Deadline D = 20.48 ms** (one window period: a result must be on screen before the next window is
  released). No external requirement for a different deadline exists in the project documents.

## 2. Per-stage timestamps (must be recorded per window)
1. release → FPGA class-ready (`t_ready`, board) — board stage, already hard-RT verified;
2. class-ready → board publisher send (`t_pub`, board);
3. publisher send → Windows receipt (`t_rx`, PC);
4. Windows receipt → GUI render completion (`t_render`, PC).
Stages 1–2 on the board clock, 4 on the PC clock, 3 and end-to-end across clocks (§4). Where an
existing path does not expose a stage separately, that is reported as a limitation, not inferred.

## 3. Run structure
- Warm-up: 200 windows (excluded). Accepted: **30,000 windows per run**, **two runs** for the new mode;
  ≥ 30,000 accepted for the measurement of the existing mode.
- Input: `artifacts/ram_demo_sets/class_balanced_256_v1` (256 real windows, 64 per class, sha256
  06ac41bd…), so all four true classes are exercised.
- Independent temperature guard 75 °C, preflight, no automatic retry, BOARD_IDLE after every run.
  A failed run is kept and reported; the second run is still executed to characterise the path.

## 4. Clock calibration and uncertainty
- Board↔PC offset measured by request/response rounds over the run's own data connection (PC send
  time ts, board reply time tb, PC receive time tr): offset sample = tb − (ts+tr)/2, error ≤ (tr−ts)/2.
- Rounds before the run, at least every 0.5 s during the run, and after the run; raw rounds saved.
- Mapping: piecewise-linear between the minimum-RTT round of consecutive ≤ 2 s buckets; per-window
  error bound e(k) = max(RTT/2 of the two anchors) (drift over ≤ 2 s at the measured ppm is added
  explicitly). Drift and its stability are reported.
- A window counts as **on time only if L(k) + e(k) ≤ D**, L(k) = t_render − map(t_rel).
- **The claim is unsupported if max e(k) > 1.0 ms** in a run (→ NOT VERIFIED).
- For the existing mode (start/end sync only) the same bound rule applies to its anchors.

## 5. Pass conditions (all, in every required run)
1. 0 accepted windows with L(k) + e(k) > D;
2. every accepted window rendered exactly once, in sequence order: 0 missing, 0 duplicate,
   0 out-of-order, 0 coalesced/skipped GUI updates;
3. 0 silent drops: every board-side overflow, transport loss or PC-side overflow is accounted with
   its sequence ID; any accounted drop of an accepted window is still a failure of condition 2;
4. class and all four logits shown/recorded on the PC for every accepted window equal the board's
   raw record and the INT10 reference (by real dataset index);
5. board stage 1 has 0 misses (validated criterion), guard rc 0, workload rc 0, no leftover processes;
6. max clock-error bound ≤ 1.0 ms.
Any failure ⇒ **END-TO-END HARD REAL-TIME NOT VERIFIED** for that mode.

## 6. Statements fixed in advance
- SSH and TCP/IP over ordinary Ethernet, and Windows/Tkinter scheduling, are **not presumed hard
  real-time**. Even a pass would be an empirical measured bound for the tested durations on this PC
  and network, not a guarantee and not a formal proof.
- A low mean or low p99 is not evidence of determinism; the verdict uses the maximum (with error bound).
- The legacy SD/gzip + SSH + Windows GUI mode remains **AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME
  NOT VERIFIED** unless it is itself tested against this criterion and passes (it is not tested here).
- The criterion is not changed after measurements are observed. Batched GUI updates are not allowed
  under this criterion (condition 2).
