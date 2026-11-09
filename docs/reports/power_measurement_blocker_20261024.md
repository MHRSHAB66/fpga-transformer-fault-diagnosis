# Power / energy measurement — BLOCKED (no current measurement available), 2026-10-24

**Result: no physical power or energy value is reported.** Nothing below is a
measurement of power; no value was estimated from temperature or voltage.

## What was inspected (read-only)
Evidence: `artifacts/results/power_inspection_20261024/board_inspection.txt`
(collected over SSH on the MYIR Z-Turn XC7Z020, kernel 4.14.0-xilinx).

| source | present? | usable for power? |
|---|---|---|
| XADC (`iio:device0`): die temperature | yes | no (temperature only) |
| XADC rail voltages VCCINT 1.001 V, VCCAUX 1.799 V, VCCBRAM 1.002 V, VCCPINT 0.998 V, VCCPAUX 1.797 V, VCCODDR 1.479 V | yes | **no — voltage without current gives no power** |
| `curr*_input`, `power*_input`, `energy*_input`, `in_current*`, `in_power*` anywhere under `/sys` | none | — |
| `/sys/class/hwmon`, `/sys/class/power_supply`, `/sys/class/regulator` | empty | — |
| I²C devices | only `sii902x` (HDMI transmitter) on `i2c-0` | no power monitor (no INA2xx / PMBus) |
| PMBus / INA / LTC / UCD kernel modules | none loaded | — |
| USB devices on the board | root hub only | no USB power meter |
| PC-side instrument | none available (confirmed by Mehrdad, 2026-10-24) | — |

## Why this blocks the task
Power needs current (or a calibrated power monitor). The board exposes no current
sense, and no external instrument is connected. Rail voltages only confirm the
supplies are within range.

## What is needed (any one)
1. Inline DC power meter (or shunt + logging DMM) between the power supply and the
   board's main DC input — preferred measurement point (whole board).
2. Bench supply that logs voltage and current (SCPI/USB), powering the board directly.
3. USB power meter — only if the board is actually powered through that USB path.

## Reproducible procedure once an instrument exists
1. Record instrument model, measurement point, sampling interval, resolution and
   accuracy; board configuration (Ethernet + UART connected, no HDMI).
2. Close the temperature dashboard (its SSH polling adds CPU load).
3. For each condition log ≥ 120 s of raw samples (≥ 10 Hz), with the read-only
   preflight before and a `BOARD_IDLE` check after:
   - idle (no overlay loaded since boot);
   - overlay loaded, no inference (`tools/board/fpga_rt_acceptance.py` preload phase
     or a load-only script);
   - sustained inference with the argmax deployment under the guard, e.g.
     `tools/run_hard_rt_20261024.ps1` configuration R1 (48.83 windows/s paced);
   - optionally the live GUI/Ethernet path.
   Repeat one condition.
4. Report mean V, I, P per condition; energy = ∫P dt over the inference interval;
   energy per inference = energy / windows executed; idle-subtracted values only as
   a separately labelled incremental estimate.
