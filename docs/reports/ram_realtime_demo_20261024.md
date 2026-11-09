# Deterministic demo mode `ram_realtime_demo` — implementation and physical validation (2026-10-24)

Criterion (fixed before the runs): `ram_realtime_demo_criterion_20261024.md`. Stock kernel
4.14.0-xilinx, deployed argmax overlay (BIT ce3c254b…, HWH eabc0163…). Nothing committed or pushed.

## 1. What it is
A separate Live-tab mode next to the unchanged legacy SD/gzip mode (which stays the default):
- **Input:** documented finite set `artifacts/ram_demo_sets/class_balanced_256_v1/` — 256 real
  test-set windows, 64 per true class, INT10 codes copied byte for byte from the staged test-set
  input (`tools/make_ram_demo_set.py`; set sha256 `06ac41bd…`, `provenance.json` lists indices,
  recorded FPGA class and reference logits). Staged once in `/home/xilinx/ram_demo_sets/…`,
  re-hashed by every preflight, preloaded into board RAM before the timed run, cycled.
- **Board runner** `tools/board/fpga_ram_rt_demo.py` (sha256 `71f3778c…`): the validated timing
  loop of `fpga_rt_acceptance.py` (release every 20.48 ms, late windows counted never skipped,
  SCHED_FIFO 50 + CPU {1} for the timed main thread only). New: each result line goes through a
  **non-blocking pipe** to a separately forked **publisher process** (normal scheduling) that writes
  to SSH; if the PC falls behind, the line is dropped for display and counted — the timed thread
  never blocks on SSH. Clock-sync rounds with the PC run before and after (never inside) the loop.
- **PC side** `host/ram_rt_demo.py` (CLI + client), GUI integration in
  `host/fpga_result_viewer.py` (mode menu "live mode", always-visible banner stating the mode and
  its verification scope). Preflight = legacy checks + RAM-set identity + no other demo running;
  the legacy preflight now also refuses to start while a RAM demo or acceptance runner is active.
  Independent temperature guard (75 °C), unique run directory, Stop/Close cleanup reuse the live
  client's mechanism. The FPGA decides every class (`class_source=fpga_register`).

## 2. Two measurements
| measurement | from → to | clock | status |
|---|---|---|---|
| board response latency | board release → FPGA class ready | board | **hard-RT boundary** |
| PC receipt latency | board release → result line read on the PC | PC (offset from SSH clock sync, ±RTT/2 ≈ ±0.4–0.7 ms) | measured only |
| GUI display latency | board release → Tk widgets updated | PC (same) | measured only |

## 3. Physical results (independent verifier `python -m tools.verify_ram_rt_demo <dir>`)
| run | accepted | misses | board p50 / p99 / p99.9 / p99.99 / max (ms) | classes | 4 logits | display dropped | guard/workload rc | PC receipt p50 / p99 / max, >20.48 ms | GUI display p50 / p99 / max, >20.48 ms |
|---|---|---|---|---|---|---|---|---|---|
| criterion, CLI (`ram_rt_demo_20261024_criterion30k`) | **30,000** | **0** | 15.86 / 16.27 / 16.40 / 16.48 / **16.50** | 30,200/30,200 | 30,200/30,200 | 0 | 0 / 0 | 16.70 / 25.65 / 85.47 ms, **851** | — (CLI) |
| GUI run_complete (`ram_rt_demo_gui_20261024_223628`) | 3,000 | 0 | 15.86 / 16.26 / 16.43 / 16.51 / 16.52 | 3,200/3,200 | 3,200/3,200 | 0 | 0 / 0 | 16.77 / 24.21 / 37.29 ms, 88 | 72.6 / 123.3 / 132.6 ms, 2,994 |
| smoke, CLI (`ram_rt_demo_20261024_smoke600`) | 400 | 0 | max 16.43 | 600/600 | 600/600 | 0 | 0 / 0 | 17.35 / 25.79 / 44.22 ms, 43 | — |

Criterion run: 618.5 s, 48.83 windows/s, all four true classes, die 47.4–48.1 °C (guard log),
no emergency stop, `BOARD_IDLE` afterwards; clock map 2 anchors, error bound ±0.43 ms, drift −3.3 ppm.
The GUI display latency includes the viewer's 80 ms queue-poll period (by design of the existing
GUI); it is a display figure, not an inference figure.

**Verdict:** the board-side response of `ram_realtime_demo` is **HARD REAL-TIME VERIFIED for the
tested duration (30,000 accepted windows, FIFO 50/CPU 1)**. SSH transport and the GUI update are
**not** hard real-time (2.8 % of PC receipts and essentially all GUI updates later than 20.48 ms
after release) and are reported as measurements only.

## 4. GUI validation (real window, real board; `tools/gui_ram_demo_validation.py`)
Final code: `artifacts/results/gui_ram_demo_validation_20261024c/` — ALL PASS:
default_mode (legacy banner), select_ram_mode (RAM banner), preflight_ram (incl.
`ram_set_identity`), run_complete (3,200 results, 4 true classes, classes/logits = reference,
evidence saved), stop_midrun (Stop after ≈140 results; runner ended by SIGTERM after 215 windows,
guard ended, board clean, evidence saved), close_midrun (window closed while running; guard ended,
board clean). Method: Tk `Button.invoke()` = the command bound to the button (OS mouse input not
available in this session), screenshots via PrintWindow.
Earlier attempts kept as evidence: `…_20261024/` (preflight bug: client built without runner path
→ fixed + unit test), `…_20261024b/` (all pass, but the banner was cut off and the Stop button was
pushed off-screen by the new mode menu → mode menu moved to its own row, banner wraps;
`layout_fix_screenshots/`), then `…_20261024c/` on the final code.

Board hand-off checks (`tools/board/check_ram_rt_demo_handoff.py`, Python 3.6 on the board):
stalled publisher never blocks the writer (≈18,270 lines dropped and counted), relay verbatim
(500/500), sync without PC skips (no hang) — `artifacts/results/ram_rt_demo_board_checks_20261024/`
(attempts 1–3 failed on harness bugs, kept).

## 5. Limitations
- Prepared MAFAULDA windows from RAM stand in for a sensor; no live 50 kHz acquisition.
- Tested duration: 30,000 accepted windows (+3,000 GUI, +400 smoke) on the stock kernel —
  an empirical bound, not a formal worst-case proof.
- Display drops are possible by design if SSH/PC stall; they are counted and never affect timing.
- The end-to-end numbers depend on the Windows PC, Ethernet and SSH; they are not guaranteed.
