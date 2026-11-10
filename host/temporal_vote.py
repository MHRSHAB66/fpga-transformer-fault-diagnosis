"""Bit-exact reference of the temporal majority vote inside the FPGA
(hls/f1_int10_seq256/vitis/temporal_vote.h).

The core keeps the classes of the last VOTE_MAX windows it decided. After each
window it votes over the newest m = min(windows seen, k) of them (k from the
AXI-Lite register vote_k, clamped to 1..VOTE_MAX): the class with the most votes
wins; between tied classes the one seen most recently wins. It reports the
winning class and its vote count. There is no reset: the history starts empty at
bitstream load and then follows the continuous window stream, like a sensor that
never stops, so a change of machine condition shows up after a few windows.
Pure Python (no numpy) so it also runs on the board.
"""
VOTE_MAX = 15


class TemporalVote:
    def __init__(self):
        self.history = []          # newest last, at most VOTE_MAX entries

    def push(self, cls, k):
        k = 1 if k < 1 else VOTE_MAX if k > VOTE_MAX else k
        self.history.append(int(cls))
        if len(self.history) > VOTE_MAX:
            self.history.pop(0)
        recent = self.history[-k:]
        counts = [recent.count(c) for c in range(4)]
        best = max(counts)
        for c in reversed(recent):             # newest first
            if counts[c] == best:
                return c, best
        raise AssertionError("unreachable")


def vote_sequence(classes, k):
    """Voted (class, count) for every element of a continuous class stream."""
    v = TemporalVote()
    return [v.push(c, k) for c in classes]
