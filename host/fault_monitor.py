"""
Stage 10/11 -- PC-side receiver + live UI for FPGA fault-classification frames.

Reads FRAME=... lines (see frame_protocol.py) from one of three sources:

  * serial  -- a real UART link to the FPGA (needs pyserial + a COM port);
  * mock    -- a built-in generator, so the whole PC pipeline / demo runs with
               NO hardware attached;
  * file    -- replay a previously captured log of frame lines.

and shows the latest predicted class (plus a rolling log and dropped/malformed
counters) in a small Tkinter window, or prints them with --headless.

The receiver only *displays* results -- it never re-classifies. The class id ->
label mapping is fixed in frame_protocol.LABELS and matches src.config.LABEL_MAP.

Examples
--------
    python -m host.fault_monitor --source mock                 # no hardware
    python -m host.fault_monitor --source serial --port COM5 --baud 115200
    python -m host.fault_monitor --source file --path capture.log --headless
"""
from __future__ import annotations

import argparse
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from host.frame_protocol import Frame, LABELS, format_frame, parse_frame, gap_after, ParseError


# ---------------------------------------------------------------------------
# Sources: each yields raw wire lines (str). Kept as generators so the receiver
# is agnostic to where bytes come from.
# ---------------------------------------------------------------------------
def serial_source(port: str, baud: int) -> Iterator[str]:
    try:
        import serial  # pyserial; lazy so mock/file work without it installed
    except ImportError as e:  # pragma: no cover - depends on environment
        raise SystemExit("pyserial not installed. `pip install pyserial` or use "
                         "--source mock / --source file.") from e
    with serial.Serial(port, baud, timeout=1) as ser:
        buf = b""
        while True:
            buf += ser.read(256)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode("ascii", errors="replace") + "\n"


def file_source(path: str, period: float = 0.2) -> Iterator[str]:
    with open(path, encoding="ascii") as f:
        for line in f:
            if line.strip():
                yield line
                time.sleep(period)


def mock_source(period: float = 0.5, seed: int = 0) -> Iterator[str]:
    """Emit plausible frames (mostly one class with occasional faults + a rare
    dropped/garbled line) so the UI and drop/malformed logic are exercised."""
    rng = random.Random(seed)
    fid = 0
    while True:
        fid = (fid + 1) % (1 << 16)
        if rng.random() < 0.03:          # simulate an occasional dropped frame
            fid = (fid + 1) % (1 << 16)
        if rng.random() < 0.02:          # simulate a garbled line
            yield "FRME=?,CLASS=bad\n"
            time.sleep(period); continue
        cls = rng.choices([0, 1, 2, 3], weights=[5, 2, 2, 3])[0]
        conf = round(rng.uniform(0.80, 0.99), 3)
        yield format_frame(fid, cls, conf)
        time.sleep(period)


# ---------------------------------------------------------------------------
# Receiver: parse lines, track health, hand Frames to a callback.
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    received: int = 0
    dropped: int = 0
    malformed: int = 0
    last_id: Optional[int] = None
    by_class: dict = field(default_factory=lambda: {c: 0 for c in LABELS})


class Receiver:
    def __init__(self, on_frame=None, on_error=None):
        self.stats = Stats()
        self.on_frame = on_frame
        self.on_error = on_error

    def feed_line(self, line: str) -> Optional[Frame]:
        try:
            frame = parse_frame(line)
        except ParseError as e:
            self.stats.malformed += 1
            if self.on_error:
                self.on_error(str(e))
            return None
        self.stats.dropped += gap_after(self.stats.last_id, frame.frame_id)
        self.stats.last_id = frame.frame_id
        self.stats.received += 1
        self.stats.by_class[frame.class_id] += 1
        if self.on_frame:
            self.on_frame(frame)
        return frame

    def run(self, lines: Iterator[str]):
        for line in lines:
            self.feed_line(line)


# ---------------------------------------------------------------------------
# UIs
# ---------------------------------------------------------------------------
def run_headless(lines: Iterator[str]):
    rx = Receiver(
        on_frame=lambda f: print(f"FRAME {f.frame_id:5d}  {f.label:<13} "
                                 f"conf={'  NA' if f.confidence is None else f'{f.confidence:.3f}'}"
                                 f"   [rx={rx.stats.received} drop={rx.stats.dropped} "
                                 f"bad={rx.stats.malformed}]"),
        on_error=lambda msg: print(f"  ! {msg}"),
    )
    try:
        rx.run(lines)
    except KeyboardInterrupt:
        pass
    print(f"\nsummary: {rx.stats}")


def run_gui(lines: Iterator[str], title="FPGA Fault Monitor"):
    import tkinter as tk

    q: "queue.Queue[object]" = queue.Queue()
    rx = Receiver(on_frame=lambda f: q.put(f), on_error=lambda m: q.put(("err", m)))
    threading.Thread(target=rx.run, args=(lines,), daemon=True).start()

    COLORS = {"normal": "#2e7d32", "imbalance": "#f9a825",
              "misalignment": "#ef6c00", "bearing_fault": "#c62828"}

    root = tk.Tk()
    root.title(title)
    root.geometry("460x420")

    label_var = tk.StringVar(value="--")
    sub_var = tk.StringVar(value="waiting for frames...")
    stat_var = tk.StringVar(value="rx=0  dropped=0  malformed=0")

    big = tk.Label(root, textvariable=label_var, font=("Segoe UI", 30, "bold"),
                   fg="white", bg="#455a64", pady=18)
    big.pack(fill="x")
    tk.Label(root, textvariable=sub_var, font=("Segoe UI", 11)).pack(pady=4)
    tk.Label(root, textvariable=stat_var, font=("Consolas", 9), fg="#555").pack()

    tk.Label(root, text="recent predictions", font=("Segoe UI", 9, "bold")).pack(pady=(8, 0))
    log = tk.Text(root, height=12, width=56, font=("Consolas", 9), state="disabled")
    log.pack(padx=8, pady=6, fill="both", expand=True)

    def append_log(text):
        log.configure(state="normal")
        log.insert("end", text + "\n")
        log.see("end")
        log.configure(state="disabled")

    def poll():
        while not q.empty():
            item = q.get_nowait()
            if isinstance(item, Frame):
                big.configure(bg=COLORS.get(item.label, "#455a64"))
                label_var.set(item.label.replace("_", " ").upper())
                conf = "NA" if item.confidence is None else f"{item.confidence:.3f}"
                sub_var.set(f"frame {item.frame_id}   confidence {conf}")
                s = rx.stats
                stat_var.set(f"rx={s.received}  dropped={s.dropped}  malformed={s.malformed}")
                append_log(f"{item.frame_id:5d}  {item.label:<13} conf={conf}")
            else:  # ("err", msg)
                append_log(f"  ! {item[1]}")
        root.after(50, poll)

    root.after(50, poll)
    root.mainloop()


def build_source(args) -> Iterator[str]:
    if args.source == "serial":
        if not args.port:
            raise SystemExit("--source serial requires --port (e.g. --port COM5)")
        return serial_source(args.port, args.baud)
    if args.source == "file":
        if not args.path:
            raise SystemExit("--source file requires --path")
        return file_source(args.path, args.period)
    return mock_source(args.period, args.seed)


def main():
    ap = argparse.ArgumentParser(description="FPGA fault-classification PC monitor.")
    ap.add_argument("--source", choices=["serial", "mock", "file"], default="mock")
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--path", help="log file for --source file")
    ap.add_argument("--period", type=float, default=0.5, help="mock/file inter-frame seconds")
    ap.add_argument("--seed", type=int, default=0, help="mock generator seed")
    ap.add_argument("--headless", action="store_true", help="print instead of GUI")
    args = ap.parse_args()

    lines = build_source(args)
    if args.headless:
        run_headless(lines)
    else:
        run_gui(lines)


if __name__ == "__main__":
    main()
