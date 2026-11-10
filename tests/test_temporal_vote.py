"""Temporal majority vote: Python reference properties and C++ (temporal_vote.h) equality."""
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from host.temporal_vote import VOTE_MAX, TemporalVote, vote_sequence

REPO = Path(__file__).resolve().parents[1]
# Xilinx ap_int headers: the second build uses the hardware bit widths.
XILINX_INCLUDE = Path("G:/Xilinx/Vitis_HLS/2020.2/include")


class TemporalVoteTest(unittest.TestCase):
    def test_k1_is_the_window_class(self):
        seq = [3, 1, 1, 0, 2, 2, 3]
        self.assertEqual([c for c, _ in vote_sequence(seq, 1)], seq)

    def test_majority_and_recency_tie_break(self):
        v = TemporalVote()
        self.assertEqual(v.push(1, 3), (1, 1))
        self.assertEqual(v.push(2, 3), (2, 1))      # tie 1:1 -> newest (2)
        self.assertEqual(v.push(1, 3), (1, 2))      # 1,2,1 -> 1 wins
        self.assertEqual(v.push(2, 3), (2, 2))      # 2,1,2 -> 2 wins

    def test_k_is_clamped(self):
        seq = [0] * 20 + [3] * 10
        self.assertEqual(vote_sequence(seq, 99)[-1], vote_sequence(seq, VOTE_MAX)[-1])
        self.assertEqual(vote_sequence(seq, 0), vote_sequence(seq, 1))

    def _cpp_vs_python(self, extra_flags):
        rng = random.Random(7)
        pairs = []
        for _ in range(5000):
            k = rng.choice([1, 2, 3, 5, 9, 15, 0, 20])
            cls = rng.choice([0, 1, 2, 3]) if rng.random() < 0.3 else (pairs[-1][1] if pairs else 0)
            pairs.append((k, cls))
        ref, v = [], TemporalVote()
        for k, c in pairs:
            ref.append(v.push(c, k))
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "tv.exe"
            subprocess.run(["g++", "-O2", "-std=c++17", "-Wno-unknown-pragmas"] + extra_flags +
                           ["-I", str(REPO / "hls" / "f1_int10_seq256" / "vitis"),
                            str(REPO / "hls" / "f1_int10_seq256" / "gpp_csim" / "test_temporal_vote.cpp"),
                            "-o", str(exe)], check=True)
            out = subprocess.run([str(exe)], input="\n".join("%d %d" % p for p in pairs),
                                 capture_output=True, text=True, check=True).stdout.split("\n")
        got = [tuple(int(x) for x in line.split()) for line in out if line.strip()]
        self.assertEqual(got, ref)

    @unittest.skipUnless(shutil.which("g++"), "g++ not on PATH")
    def test_cpp_equals_python(self):
        self._cpp_vs_python([])

    @unittest.skipUnless(shutil.which("g++") and (XILINX_INCLUDE / "ap_int.h").exists(),
                         "g++ or the Xilinx ap_int headers not available")
    def test_cpp_ap_int_equals_python(self):
        self._cpp_vs_python(["-DHLS_USE_AP_INT", "-I", str(XILINX_INCLUDE)])


if __name__ == "__main__":
    unittest.main()
