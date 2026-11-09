#!/usr/bin/env python3
"""UART result path on the board (Python 3.6, deployed driver): FPGA class -> USB-UART -> PC.

Classifies the windows of a RAM demo set on the FPGA (class from the FPGA logic
when the overlay has the pred_class register) and sends one text frame per window
over the board's PS UART, the same USB-UART link (CP2102, 115200 8N1) that carries
the boot console:

    FRAME=<k>,CLASS=<id>,LABEL=<name>,CONF=<0..1>\\n        (host/frame_protocol.py)

CONF is display-only: the softmax of the FPGA's 4 logits with the deployed logits
scale and the validation-fitted temperature, computed on the ARM; it never changes
the class. Windows are released every <period_s>; a window is never skipped.
Run it inside <DEP>/driver (PYTHONPATH there) under temp_guard.sh, with the serial
console (getty) stopped so the UART carries only frames.

Usage: python3.6 -u fpga_uart_demo.py <bit> <set.i32> <indices.txt> <n_windows> \
           <period_s> <tty> <conf_json> <out_dir>
Writes <out_dir>/sent.jsonl (one record per frame, with timings) and summary.json.
"""
import json
import os
import sys
import termios
import time

import numpy as np

REC = 1024 * 8
XADC = "/sys/bus/iio/devices/iio:device0"


def temp_c():
    try:
        rd = lambda n: float(open(os.path.join(XADC, n)).read().strip())
        return round((rd("in_temp0_raw") + rd("in_temp0_offset")) * rd("in_temp0_scale") / 1000.0, 2)
    except (OSError, ValueError):
        return None


def open_uart(path):
    """Raw 115200 8N1 output, no flow control, no output processing."""
    fd = os.open(path, os.O_WRONLY | os.O_NOCTTY)
    attr = termios.tcgetattr(fd)
    attr[0] = 0                                         # iflag
    attr[1] = 0                                         # oflag: no \n -> \r\n
    attr[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attr[3] = 0                                         # lflag: raw
    attr[4] = attr[5] = termios.B115200
    termios.tcsetattr(fd, termios.TCSANOW, attr)
    termios.tcflush(fd, termios.TCOFLUSH)
    return fd


def confidence(logits, cls, scale, temperature):
    z = np.asarray(logits, dtype=np.float64) * scale / temperature
    e = np.exp(z - z.max())
    return float(e[cls] / e.sum())


def main():
    (bit, set_path, idx_path, n_windows, period, tty, conf_json, out_dir) = sys.argv[1:9]
    n_windows, period = int(n_windows), float(period)
    os.makedirs(out_dir, exist_ok=True)
    windows = np.fromfile(set_path, dtype="<i4").reshape(-1, REC)
    indices = np.loadtxt(idx_path, dtype=np.int64)
    assert len(windows) == len(indices)
    conf_cfg = json.load(open(conf_json))

    from fault_classifier import FaultClassifier, HardwareBackend
    from frame_protocol import format_frame
    clf = FaultClassifier(backend=HardwareBackend(bit, wait_timeout=10.0))
    fd = open_uart(tty)
    log = open(os.path.join(out_dir, "sent.jsonl"), "w")
    t_start = time.monotonic() + 0.5
    lat, tx, late = [], [], 0
    temp_start = temp_c()
    try:
        for k in range(n_windows):
            slot = k % len(windows)
            release = t_start + k * period
            now = time.monotonic()
            if now < release:
                time.sleep(release - now)
            else:
                late += 1
            t0 = time.monotonic()
            cls, logits = clf.infer_from_codes(windows[slot].reshape(1024, 8))
            t1 = time.monotonic()
            conf = confidence(logits, cls, conf_cfg["logits_scale"], conf_cfg["T"])
            line = format_frame(k, int(cls), conf)
            # Queue the frame in the tty driver and return; waiting with tcdrain()
            # per frame costs ~24 ms on this kernel (10 ms jiffy polling), longer
            # than the 20.48 ms period. 52 bytes per 20.48 ms use ~23 % of 115200 baud.
            os.write(fd, line.encode("ascii"))
            t2 = time.monotonic()
            lat.append((t1 - t0) * 1e3)
            tx.append((t2 - t1) * 1e3)
            log.write(json.dumps({"k": k, "dataset_index": int(indices[slot]), "cls": int(cls),
                                  "logits": [int(v) for v in logits], "conf": round(conf, 3),
                                  "line": line.strip(), "release": round(release - t_start, 6),
                                  "infer_ms": round(lat[-1], 3), "write_ms": round(tx[-1], 3)}) + "\n")
    finally:
        log.close()
        termios.tcdrain(fd)                             # all queued frames sent
        os.close(fd)
    summary = {"n_windows": n_windows, "period_s": period, "class_source": clf.class_source,
               "tty": tty, "baud": 115200,
               "infer_ms": {"mean": round(float(np.mean(lat)), 3), "max": round(float(np.max(lat)), 3)},
               "uart_write_ms": {"mean": round(float(np.mean(tx)), 3), "max": round(float(np.max(tx)), 3)},
               "releases_already_late": late, "temp_c": {"start": temp_start, "end": temp_c()},
               "confidence": "softmax(logits * %g / T), T = %g; display only" % (
                   conf_cfg["logits_scale"], conf_cfg["T"])}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
