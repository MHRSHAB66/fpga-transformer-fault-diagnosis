"""
End-to-end demo: window -> int10 classify -> UART frame to the PC monitor.

Runs on the board (HardwareBackend + real UART to the PC) OR on any machine
(NumpyBackend, and either a real serial port or --print for a dry run). This is
the glue between Stage 8 (inference) and Stage 10 (FPGA->PC link); pair it with
`python -m host.fault_monitor --source serial --port <PC-COM>` on the PC.

Confidence is a softmax over the integer logits, computed here as a convenience
(the integer core itself emits only class + logits); pass --no-conf to send NA.

Examples
--------
    python -m pynq.run_demo --print                       # dry run, no hardware
    python -m pynq.run_demo --port COM6                    # numpy infer + real UART
    python -m pynq.run_demo --bitstream design.bit --port /dev/ttyPS0   # on board
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pynq.fault_classifier import FaultClassifier, NumpyBackend
from host.frame_protocol import format_frame, LABELS

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_IN = REPO_ROOT / "hls" / "f1_int10" / "golden" / "golden_in.txt"


def softmax_conf(logits: np.ndarray) -> float:
    z = logits.astype(np.float64)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return float(p.max())


def load_golden_windows():
    lines = GOLDEN_IN.read_text().splitlines()
    n, seq, ch = map(int, lines[0].split())
    return [np.array(lines[1 + i].split(), dtype=np.int64).reshape(seq, ch)
            for i in range(n)]


def open_serial(port, baud):
    import serial
    return serial.Serial(port, baud, timeout=1)


def main():
    ap = argparse.ArgumentParser(description="int10 classify -> UART frame demo.")
    ap.add_argument("--bitstream", help="overlay .bit for the HardwareBackend")
    ap.add_argument("--port", help="serial port to the PC (e.g. COM6, /dev/ttyPS0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--print", dest="dry", action="store_true",
                    help="print frames instead of sending over serial")
    ap.add_argument("--no-conf", action="store_true", help="send CONF=NA")
    ap.add_argument("--period", type=float, default=0.5, help="seconds between windows")
    ap.add_argument("--loops", type=int, default=1, help="how many passes over the windows")
    args = ap.parse_args()

    clf = FaultClassifier(bitstream=args.bitstream) if args.bitstream \
        else FaultClassifier(backend=NumpyBackend())
    windows = load_golden_windows()
    ser = None if args.dry else open_serial(args.port, args.baud) if args.port else None
    if not args.dry and ser is None:
        raise SystemExit("give --port <serial> or --print for a dry run")

    fid = 0
    for _ in range(args.loops):
        for w in windows:
            cls, logits = clf.infer_from_codes(w)
            conf = None if args.no_conf else softmax_conf(logits)
            line = format_frame(fid, cls, conf)
            if ser is not None:
                ser.write(line.encode("ascii"))
            print(f"[{fid:5d}] {LABELS[cls]:<13} logits={logits.tolist()} "
                  f"conf={'NA' if conf is None else f'{conf:.3f}'}"
                  f"{'  -> ' + args.port if ser is not None else ''}")
            fid += 1
            time.sleep(args.period)
    if ser is not None:
        ser.close()


if __name__ == "__main__":
    main()
