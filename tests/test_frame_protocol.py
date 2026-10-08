"""Tests for the FPGA->PC text frame protocol (host.frame_protocol).

Pure standard library (no torch / no board). Covers format/parse round-trip, the
NA-confidence sentinel, class/label validation, wrap-aware drop detection, and a
byte-stream "UART-like" loopback over an in-process socket pair (no hardware).
"""
import socket
import unittest

from host.frame_protocol import (
    LABELS, format_frame, parse_frame, Frame, ParseError, gap_after, ID_WRAP,
)


class FrameProtocolTests(unittest.TestCase):
    def test_round_trip_all_classes_na_conf(self):
        for cid, label in LABELS.items():
            line = format_frame(10 + cid, cid, None)
            f = parse_frame(line)
            self.assertEqual((f.frame_id, f.class_id, f.label, f.confidence),
                             (10 + cid, cid, label, None))

    def test_round_trip_with_confidence(self):
        f = parse_frame(format_frame(3, 2, 0.875))
        self.assertEqual(f.class_id, 2)
        self.assertEqual(f.label, "misalignment")
        self.assertAlmostEqual(f.confidence, 0.875, places=3)

    def test_na_sentinel_is_none(self):
        self.assertIsNone(parse_frame("FRAME=0,CLASS=0,LABEL=normal,CONF=NA").confidence)

    def test_frame_id_wraps_at_16bit(self):
        line = format_frame(ID_WRAP + 5, 1, None)
        self.assertIn("FRAME=5,", line)

    def test_rejects_out_of_range_class(self):
        with self.assertRaises(ValueError):
            format_frame(0, 4, None)

    def test_label_class_mismatch_is_malformed(self):
        with self.assertRaises(ParseError):
            parse_frame("FRAME=0,CLASS=0,LABEL=bearing_fault,CONF=NA")

    def test_confidence_out_of_range_is_malformed(self):
        with self.assertRaises(ParseError):
            parse_frame("FRAME=0,CLASS=0,LABEL=normal,CONF=1.5")

    def test_garbage_line_is_malformed(self):
        with self.assertRaises(ParseError):
            parse_frame("hello world")

    def test_gap_after_is_wrap_aware(self):
        self.assertEqual(gap_after(None, 7), 0)
        self.assertEqual(gap_after(4, 5), 0)
        self.assertEqual(gap_after(4, 8), 3)
        self.assertEqual(gap_after(ID_WRAP - 1, 1), 1)  # wrapped, one dropped

    def test_uart_like_loopback_over_socket_pair(self):
        """Frames survive a raw byte-stream transport and re-frame on newlines."""
        tx, rx = socket.socketpair()
        try:
            sent = [format_frame(i, i % 4, None) for i in range(16)]
            tx.sendall("".join(sent).encode("ascii"))
            tx.close()
            buf = b""
            chunks = []
            while True:
                data = rx.recv(4096)
                if not data:
                    break
                chunks.append(data)
            buf = b"".join(chunks).decode("ascii")
        finally:
            rx.close()
        received = [parse_frame(ln) for ln in buf.splitlines()]
        self.assertEqual(len(received), 16)
        self.assertEqual([f.class_id for f in received], [i % 4 for i in range(16)])
        # no drops implied by consecutive ids
        prev = None
        for f in received:
            self.assertEqual(gap_after(prev, f.frame_id), 0)
            prev = f.frame_id


if __name__ == "__main__":
    unittest.main()
