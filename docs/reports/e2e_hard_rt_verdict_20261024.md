# End-to-end hard real-time — measurement, new transport mode, verdict (2026-10-24)

Criterion frozen BEFORE implementation and tests: `e2e_hard_rt_criterion_20261024.md`
(sha256 `4ae0e1c7…`, `.sha256` file next to it; unchanged). Stock kernel 4.14.0-xilinx, argmax
overlay BIT ce3c254b…/HWH eabc0163… (unchanged). Nothing committed or pushed; no flash, reboot,
QSPI or network-configuration change. Independent verifier:
`python -m tools.verify_e2e_20261024 existing|new <dir>` (raw npz + PC traces + raw sync rounds).

## 1. Verdicts
| what | verdict |
|---|---|
| Board stage, release → FPGA class-ready (all runs below, FIFO 50/CPU 1) | 0 misses in every run — the validated board-side **HARD REAL-TIME VERIFIED** result is unchanged |
| Existing `ram_realtime_demo` GUI path, end to end (30,000 accepted) | **END-TO-END HARD REAL-TIME NOT VERIFIED** — 30,000/30,000 windows late |
| New `ram_tcp_e2e_demo` path, end to end (2 × 30,000 accepted) | **END-TO-END HARD REAL-TIME NOT VERIFIED** — 14,489 and 8,866 windows late |
| Legacy SD/gzip + SSH + Windows GUI mode | not re-tested; remains **AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME NOT VERIFIED** |

The negative result is structural: the board stage alone takes 15.8–16.6 ms of the 20.48 ms
deadline, leaving ≈ 3.9–4.6 ms for publisher, Ethernet/TCP, Windows scheduling and Tk drawing,
none of which is time-bounded. The new mode moves the median below the deadline but not the tail.

## 2. Timing boundary (from the criterion)
Start = scheduled board release t_rel(k) (board CLOCK_MONOTONIC). End = `update_idletasks()`
returned after the widget showing window k's frame ID/class was updated (Windows QPC). DWM
composition and monitor scan-out are not included (no optical sensor) — even a pass would only
have been "to Tk render completion". On time only if L + e ≤ 20.48 ms, e = clock-error bound;
claim unsupported if max e > 1.0 ms. Every accepted window must be rendered once, in order.

## 3. Existing `ram_realtime_demo` measured as-is
Run: `artifacts/results/e2e_existing_ram_mode_20261024/` (harness) +
`artifacts/results/ram_rt_demo_gui_20261024_230459/` (mode evidence). The product code was not
changed; render completion was taken by a Tk `after_idle` callback queued after each update
(FIFO idle queue ⇒ after that update's redraw); the harness stepped Tk with `update()` + 1 ms sleeps.

| stage (ms) | p50 | p95 | p99 | p99.9 | p99.99 | max |
|---|---|---|---|---|---|---|
| 1 release → class-ready (board) | 15.87 | 16.11 | 16.28 | 16.42 | 16.48 | 16.53 |
| 2+3 class-ready → PC receipt (not separable in this mode) | 0.88 | 5.72 | 12.80 | 24.26 | 40.63 | 59.30 |
| 4 receipt → render (80 ms GUI poll) | 59.30 | 102.33 | 120.62 | 158.59 | 233.05 | 274.11 |
| **end to end** | 76.77 | 120.14 | 139.94 | 178.23 | 250.80 | **291.72** (min 21.89) |

30,000/30,000 late; clock bound max 1.99 ms (> 1.0 ms: only start/end anchors, 622 s apart);
0 missing/duplicate/out-of-order; classes and all 4 logits 30,000/30,000; guard/workload rc 0.

## 4. New mode `ram_tcp_e2e_demo` — architecture (why)
- **Data plane without SSH:** board publisher listens on TCP 50555 (board has no firewall; PC
  connects outbound — no network configuration change), one persistent connection, TCP_NODELAY,
  fixed 57-byte binary frames: immutable sequence id, dataset index, class, 4 logits, board
  t_rel/t_ready/t_pubrx/t_pub, cumulative board overflow. SSH = control plane only (launch,
  guard, Stop) — it was the measured cause of stage 2+3 tails in the existing mode.
- **Never-blocking board:** timed thread (FIFO 50, CPU 1) → non-blocking 4 KiB pipe (102 records)
  → forked publisher (FIFO 40, CPU 0) → socket (SO_SNDBUF 8 KiB). Full pipe ⇒ record not queued,
  seq recorded as board overflow; publisher counts undelivered frames after a disconnect.
- **Clock sync on the data connection:** 50 rounds before, 1 every 0.5 s during, 30 after;
  piecewise-linear map between min-RTT anchors per 2 s, bound = max(RTT/2) + segment drift ×
  distance (`host/e2e_clock.py`); a causality check (transport latency < −bound) found 0 violations.
- **GUI:** third Live-tab mode; a 1 ms Tk tick renders EVERY frame individually (no batching) and
  stamps t_render after `update_idletasks()`; 1 ms Windows timer resolution and HIGH priority class
  only while this mode runs; receiver thread TIME_CRITICAL; bounded PC queue (1,024) with
  overflow recorded by seq; banner names the mode; counters show board/PC overflow,
  missing/duplicate/out-of-order and data errors.

## 5. New mode — criterion runs (GUI, `tools/gui_e2e_validation_20261024.py`, real Tk mainloop)
| stage (ms) | run 1 p50 / p95 / p99 / p99.9 / p99.99 / max | run 2 p50 / p95 / p99 / p99.9 / p99.99 / max |
|---|---|---|
| 1 release → class-ready | 15.85 / 16.10 / 16.27 / 16.41 / 16.51 / 16.59 | 15.86 / 16.13 / 16.28 / 16.41 / 16.49 / 16.51 |
| 2 class-ready → publisher send | 0.20 / 0.33 / 0.38 / 0.43 / 0.49 / 0.58 | 0.20 / 0.34 / 0.39 / 0.44 / 0.53 / 0.61 |
| 3 send → PC receipt | 0.37 / 3.85 / 11.97 / 15.39 / 39.39 / 59.79 | 0.31 / 0.48 / 1.23 / 3.20 / 11.10 / 30.41 |
| 4 receipt → Tk render | 3.40 / 7.41 / 21.80 / 59.00 / 76.14 / 95.81 | 2.78 / 7.20 / 12.21 / 31.27 / 99.15 / 118.50 |
| **end to end** | 20.17 / 26.74 / 39.62 / 76.52 / 95.38 / **112.10** | 19.25 / 23.72 / 28.71 / 48.50 / 115.58 / **135.07** |
| late (L + e > 20.48 ms) | **14,489 / 30,000** (48.3 %) | **8,866 / 30,000** (29.6 %) |
| clock bound max | **1.31 ms (> 1.0 ms limit)** | 0.37 ms |

Both runs: 30,000/30,000 rendered once and in order (0 missing, 0 duplicate, 0 out-of-order,
0 PC overflow, 0 board overflow), classes and all 4 logits 30,000/30,000 = board record =
INT10 reference, 4 true classes, board stage 0 misses, guard/workload rc 0, BOARD_IDLE.
Board CPU (runner's own /proc/stat): CPU1 81–82 %, CPU0 9–14 %; 50–51 °C. PC CPU mean ≈ 11 %,
single cores briefly 94–95 %. Evidence: `artifacts/results/e2e_demo_gui_20261024_231859/`,
`…_233000/`, each with `e2e_verification.json`.

## 6. New mode — fault and control scenarios (same window; all mechanically PASS)
| scenario | result | evidence |
|---|---|---|
| Stop after ≈ 300 rendered | runner SIGTERM at window 497, all 497 sent and received, board clean, worker ended 11.7 s after Stop | `e2e_demo_gui_20261024_233044` |
| transport disconnect (PC closes data socket at 8 s, 1,200 windows) | board ran to completion (0 misses, rc 0); publisher counted 828 undelivered from seq 372 = the 828 seqs missing on the PC; PC data error recorded | `…_233145` |
| receiver stall 10 s (TCP backpressure) | no overflow occurred: Windows buffered the ≈ 490 frames despite SO_RCVBUF 8 KiB; publisher send ≤ 0.2 ms; clock bound grew to seconds during the stall (correctly) | `…_233244` |
| display paused 30 s (3,200 windows) | PC queue overflow 498 frames, each recorded by seq (498 not rendered); board unaffected | `…_233424` |
| window closed mid-run | close handler 4.1 s, guard ended, board clean | `gui_validation.json` |
Board-side contract check (`tools/board/check_ram_e2e_publisher.py`): ordered frames, sync, start
handshake, non-blocking writer (worst 136 µs) with overflow after 434 queued records and no gap on
resume, disconnect accounting, publisher exit — PASS
(`artifacts/results/e2e_board_checks_20261024/`; attempt 1 failed on a test bug, kept).
Smoke run (1,000 accepted): 251 late (`gui_e2e_validation_20261024_smoke/`).

## 7. Limitations
- The deadline cannot be met with margin by construction: board stage ≈ 15.9 ms leaves ≈ 4.6 ms
  for a non-real-time network stack, OS and GUI. A pass would need the board stage shortened or a
  real-time receiver/display (e.g. on-board output, or an RT controller), not a Windows/Tk GUI.
- Render completion is a software proxy; DWM/monitor latency (typically ≥ one 60 Hz frame,
  16.7 ms) is not measured and would add to every value above.
- Results hold for this PC, cable and load; Windows was not isolated (other processes ran).
- Run 1 also failed the clock-bound condition (1.31 ms); run 2 met it (0.37 ms).
- The existing-mode harness drove Tk with `update()` + 1 ms sleeps (not `mainloop`); its stage 4
  is dominated by the 80 ms poll either way.
