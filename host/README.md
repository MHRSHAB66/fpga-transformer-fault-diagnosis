# PC-side monitor (Stage 10/11)

Receives `FRAME=...` lines from the FPGA over UART and shows the latest fault
class live, with a rolling log and dropped/malformed counters. Display only — it
never re-classifies; the FPGA is the source of truth.

## Files
- `frame_protocol.py` — the single source of truth for the wire format
  `FRAME=<id>,CLASS=<id>,LABEL=<label>,CONF=<conf|NA>`, plus parsing, class/label
  validation, and wrap-aware dropped-frame detection.
- `fault_monitor.py` — receiver + Tkinter UI + headless mode, with three input
  sources: `serial` (real link), `mock` (no hardware), `file` (replay a log).

## Run
```bash
# No hardware — built-in generator, good for a UI/demo dry run:
python -m host.fault_monitor --source mock

# Real link to the FPGA:
python -m host.fault_monitor --source serial --port COM5 --baud 115200

# Replay a captured log, printing instead of GUI:
python -m host.fault_monitor --source file --path capture.log --headless
```

## Protocol
```
FRAME=<id>,CLASS=<class_id>,LABEL=<label>,CONF=<confidence>\n
```
- `id` — 16-bit window counter (wraps at 65536); gaps are counted as dropped.
- `class_id` / `label` — must be consistent (`0 normal, 1 imbalance,
  2 misalignment, 3 bearing_fault`); a mismatch is flagged as malformed.
- `confidence` — `0..1` float, or `NA` when the core does not emit one.

Requires `pyserial` only for the `serial` source (`pip install pyserial`);
`mock`/`file` and the Tkinter UI need nothing beyond the standard library.
