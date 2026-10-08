"""Tests for the Live tab lifecycle without a board or mainloop.

Covers the memory/teardown hardening: bounded queue drops oldest (not unbounded
growth), shutdown is idempotent and cancels the pending after() callback, and
Stop is safe when nothing is running. GUI event loop is not started; Tk is only
used to construct the widgets. Skipped if no display/Tk is available.
"""
import queue
import unittest

try:
    import tkinter as tk
    from tkinter import ttk
    _TK = True
    try:
        _root = tk.Tk(); _root.withdraw(); _root.destroy()
    except Exception:
        _TK = False
except Exception:
    _TK = False


@unittest.skipUnless(_TK, "Tkinter/display not available")
class LiveLifecycleTests(unittest.TestCase):
    def setUp(self):
        from host.fpga_result_viewer import LiveFrame
        self.root = tk.Tk(); self.root.withdraw()
        self.frame = LiveFrame(ttk.Frame(self.root), self.root)

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_bounded_queue_drops_oldest(self):
        cap = self.frame.q.maxsize
        # push well past capacity; queue must stay bounded and count drops
        for i in range(cap + 250):
            self.frame._put({"type": "status", "msg": str(i)})
        self.assertLessEqual(self.frame.q.qsize(), cap)
        self.assertEqual(self.frame.dropped, 250)

    def test_shutdown_is_idempotent_and_cancels_after(self):
        self.assertTrue(self.frame._alive)
        self.assertIsNotNone(self.frame._poll_after)
        self.assertTrue(self.frame.shutdown())   # nothing running -> worker gone
        self.assertFalse(self.frame._alive)
        self.assertIsNone(self.frame._poll_after)
        # calling again must not raise
        self.assertTrue(self.frame.shutdown())

    def test_poll_stops_when_not_alive(self):
        self.frame._alive = False
        self.frame._poll_after = "sentinel"
        self.frame._poll()                        # should return immediately
        self.assertEqual(self.frame._poll_after, "sentinel")  # not rescheduled

    def test_stop_safe_with_no_client(self):
        self.frame.client = None
        self.frame.stop()                         # must not raise

    def test_pause_resume_flags(self):
        self.frame.pause()
        self.assertTrue(self.frame.paused)
        self.frame.resume()
        self.assertFalse(self.frame.paused)


if __name__ == "__main__":
    unittest.main()
