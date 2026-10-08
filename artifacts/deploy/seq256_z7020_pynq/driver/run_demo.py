"""
End-to-end demo: window -> INT10 classify -> UART frame to the PC monitor.

Runs on the board (HardwareBackend + real UART to the PC) OR on any machine
(NumpyBackend, and either a real serial port or --print for a dry run).

Board usage (avoids the vendor `pynq` package-name clash): run this file as a
SCRIPT from inside the deployment bundle's `driver/` folder so the adjacent
`fault_classifier.py` / `frame_protocol.py` are imported, and `from pynq import
Overlay` inside the driver resolves to the vendor runtime:

    python driver/run_demo.py --bitstream overlay/design_1_wrapper.bit --port /dev/ttyPS0

Repository/dry-run usage (NumPy inference, no board):

    python driver/run_demo.py --print
    python -m pynq.run_demo --print          # repository package form (dev/tests)

The integer hardware core emits only class + 4 logits, so confidence defaults to
NA. `--sw-conf` adds a SOFTWARE-ONLY softmax over the logits as a convenience;
this is NOT a hardware-supported confidence and must not be presented as one.

Golden windows (16 SEQ=256 ingress vectors, each a full 1024x8 INT10 window) are
resolved relative to this file (bundle `golden/` or repo `golden_ingress1024/`);
the backend performs the 4:1 aggregation internally, exactly like the IP.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

# Import adjacent modules when run as a script (board bundle / direct exec), and
# fall back to the repository package layout when imported as `pynq.run_demo`.
try:
    from fault_classifier import FaultClassifier, NumpyBackend, golden_dir
    from frame_protocol import format_frame, LABELS
except ImportError:  # repository package layout (tests / `python -m pynq.run_demo`)
    from pynq.fault_classifier import FaultClassifier, NumpyBackend, golden_dir
    from host.frame_protocol import format_frame, LABELS


def softmax_conf(logits: np.ndarray) -> float:
    """SOFTWARE-ONLY convenience; the hardware emits no confidence."""
    z = logits.astype(np.float64)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return float(p.max())


def load_golden_windows():
    golden_in = golden_dir() / "golden_in.txt"
    lines = golden_in.read_text().splitlines()
    n, seq, ch = map(int, lines[0].split())
    return [np.array(lines[1 + i].split(), dtype=np.int64).reshape(seq, ch)
            for i in range(n)]


def open_serial(port, baud):
    import serial
    return serial.Serial(port, baud, timeout=1)


def main():
    ap = argparse.ArgumentParser(description="INT10 classify -> UART frame demo.")
    ap.add_argument("--bitstream", help="overlay .bit for the HardwareBackend")
    ap.add_argument("--port", help="serial port to the PC (e.g. COM6, /dev/ttyPS0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--print", dest="dry", action="store_true",
                    help="print frames instead of sending over serial")
    ap.add_argument("--sw-conf", action="store_true",
                    help="attach a SOFTWARE-ONLY softmax confidence (default: CONF=NA, "
                         "matching the hardware which emits no confidence)")
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
            conf = softmax_conf(logits) if args.sw_conf else None
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
