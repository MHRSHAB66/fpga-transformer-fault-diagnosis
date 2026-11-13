# Research program — claims, evidence and equipment queue (2026-11-13)

Ledger with commands and statuses: `docs/research/experiment_ledger.json`.
Frozen baseline and pre-registered gates: `artifacts/research/F00_20261113/`.
Evidence levels: proposed < implemented < software-measured < simulated (C-model/HLS
csim) < synthesized < post-route < physically measured < independently confirmed.
The canonical deployment is unchanged; every variant has its own directory, bundle and
reference. The test split was explored before; it is never called an independent
confirmation set.

## Claim → evidence

| # | Claim | Level | Evidence |
|---|---|---|---|
| 1 | In the INT10 attention, 8-bit signed probabilities lose accuracy through probability-mass truncation, not through all-zero rows (0 zero rows on validation) | software-measured (validation, 195 files) | `artifacts/research/A02_20261113/summary.json` |
| 2 | Weight precision is not the bottleneck (W8: −0.01 pp macro-F1); 8-bit unsigned probabilities cost nothing (+0.49 pp, CI [−0.18, 1.21]) | software-measured | same |
| 3 | The softmax normalisation quotient is ≤ 511, so a 9-step divider is exact | proven + exhaustive (367,657,004 reachable pairs, native and ap_int) | `hls/research/a04_softmax_div/`, `artifacts/research/A04_20261113/exhaustive_*.txt` |
| 4 | The exact divider saves 15.5 % LUT and 18.7 % FF of the whole raw-input core at equal DSP/BRAM/latency, with bit-identical outputs | post-route + physically measured (95,404/95,404 logits = canonical, 0 misses / 30,000) | `artifacts/research/A05_20261113/`, `artifacts/results/raw_core_board_*_20261113_14*` |
| 4b | Unsigned 8-bit probabilities + an exact 8-bit divider (a different numerical model, own reference and goldens) save 17.6 % LUT / 21.3 % FF at equal DSP/BRAM/latency, within the frozen accuracy gate (test, previously explored: macro-F1 +0.19 pp, worst recall −0.16 pp); only −2.4 % LUT beyond the exact divider alone | post-route + physically measured (95,404/95,404 = own C-model, 0 misses / 30,000) | `artifacts/research/A06_20261113/summary.json`, `artifacts/results/raw_core_board_*_20261113_21*`, `_220122` |
| 5 | Silent class corruption from single-bit faults is concentrated (93 %) in the output logits and the input token codes; attention, weights and vote state almost never change the class | software-measured (logical emulation, 3,072 + 384 + 256 events) | `artifacts/research/B01_20261113/` |
| 6 | Input-buffer signature + duplicated classifier tail + proven-range check remove 90.8 % of silent class corruptions on a held-out campaign | software-measured (modelled protection) | `artifacts/research/B02_20261113/heldout_campaign/` |
| 7 | These checks cost +1.0 % LUT, +1.2 % FF, 0 DSP, +1.1 ms latency, with bit-identical outputs and 0 false flags on 95,404 clean windows | post-route + physically measured | `artifacts/research/B02_20261113/b02_build/`, `artifacts/results/raw_core_board_*_20261113_18*`, `_191441` |
| 8 | The board-side deadline holds under bounded CPU, memory, cache, SD and network load (0 misses in 7 × 30,000 windows, max 16.91 ms) | physically measured (finite envelope, not WCET) | `artifacts/research/S01_20261113/` |
| 9 | Margin-weighted integer evidence beats k=5 voting at a matched false-switch budget, but does not dominate hysteresis | software-measured | `artifacts/research/T01_20261113/summary.json` |
| 10 | Raw-input quality flags + integer top-two margin cut accepted wrong outputs by 95.5 % under modelled acquisition faults at 90 % clean coverage (validation); 0 accepted false-healthy on the test report | software-measured (bit-exact C-model, modelled faults) | `artifacts/research/Q01_20261113/summary.json` |

Not claimed: physical fault coverage (no hardware injection yet), radiation tolerance or
upset rates, analog front-end behaviour, power/energy (no logging instrument), universal
WCET, generalization to other machines.

## Equipment request (faculty) — kept pending, not rejected

| ID | Equipment | Question it answers | Protocol notes | Status |
|---|---|---|---|---|
| EQ_POWER | Logging DC power meter or logging bench supply for the board's 5 V / 3 A barrel input (≥ 3 A range, ≥ 10 samples/s, PC logging); alternative: calibrated shunt + DAQ | Whole-board idle / overlay-loaded / sustained-inference power and energy per accepted window; canonical vs exact-divider vs integrity overlays | Same input and timing boundary; idle-subtracted and total energy; instrument accuracy and sample rate reported; a non-logging multimeter only gives a steady-state mean | requested, access unconfirmed |
| EQ_FRONTEND | Multichannel ADC/DAQ compatible with the board + calibrated signal generator (or digital replay source) | Real acquisition-chain gain/clipping/lag versus the modelled raw24 input; Q01 flags on real faults | Known references, documented voltage/interface/sample alignment; digital replay kept separate from analog results | requested, access unconfirmed |
| EQ_TIMING | Logic analyzer or oscilloscope (3.3 V PMOD-compatible) | Independent edge timing of release, core start/end and result transfer | Explicit pulse semantics, instrumentation overhead, independent clock or stated uncertainty | optional |
| EQ_THERMAL | Controlled temperature environment within rated limits | Operating envelope under temperature | Declared limits, continuous guard, no overclock/undervoltage | optional |
| EQ_MACHINE | Approved rig with calibrated sensors and documented healthy/fault labels | External transfer evidence | Kept separate from MAFAULDA evaluation; labels and conditions documented | optional |
