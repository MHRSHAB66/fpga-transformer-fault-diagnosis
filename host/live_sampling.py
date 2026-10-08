"""Live-demo window-selection plans (software only -- NO hardware run here).

The first ~150 test-set windows are almost all one class (bearing_fault), which
makes a poor multiclass Live demo. This module builds an ordered list of REAL
dataset indices to feed the Live runner so a future on-board demo shows all
classes. It never classifies and never predicts: it only chooses which window
indices to send; the FPGA still produces every class decision, and the UI shows
the true dataset frame id (= the chosen index).

Plans (all deterministic):
  * sequential     -- indices 0..n-1 (back-compat; what Live did before).
  * strided        -- evenly spaced across the whole set (spreads classes out).
  * class_balanced -- round-robin across classes using ground-truth labels, so
                      each class contributes ~n/num_classes windows.
  * explicit       -- a caller-supplied list (validated).

Ground-truth labels are used ONLY to choose a representative index order for the
demo; they are not shown as predictions. Hardware still decides each class.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from host.frame_protocol import LABELS

NUM_CLASSES = len(LABELS)


class SamplingError(ValueError):
    pass


def strided_plan(total: int, n: int) -> List[int]:
    """n indices evenly spaced over [0, total). Order ascending."""
    _check_total_n(total, n)
    if n == total:
        return list(range(total))
    step = total / float(n)
    idx = sorted({min(total - 1, int(round(i * step))) for i in range(n)})
    # round collisions can shrink the set; top up with nearest unused indices
    i = 0
    while len(idx) < n and i < total:
        if i not in idx:
            idx.append(i)
        i += 1
    return sorted(idx)[:n]


def class_balanced_plan(labels: Sequence[int], n: int,
                        seed: int = 0) -> List[int]:
    """~n/num_classes indices per present class, interleaved round-robin.

    Within a class, indices are taken evenly spaced (deterministic) so the demo
    spans the whole file rather than one contiguous run. Classes are visited
    round-robin, so the Live stream alternates classes.
    """
    total = len(labels)
    _check_total_n(total, n)
    by_class = {c: [] for c in range(NUM_CLASSES)}
    for i, lab in enumerate(labels):
        if lab in by_class:
            by_class[lab].append(i)
    present = [c for c in range(NUM_CLASSES) if by_class[c]]
    if not present:
        raise SamplingError("no labelled windows to sample from")
    per = max(1, n // len(present))
    picked = {c: _evenly(by_class[c], per) for c in present}
    # round-robin interleave, preserving each class's ascending order
    order: List[int] = []
    cursors = {c: 0 for c in present}
    while len(order) < n:
        advanced = False
        for c in present:
            if len(order) >= n:
                break
            cur = cursors[c]
            if cur < len(picked[c]):
                order.append(picked[c][cur])
                cursors[c] += 1
                advanced = True
        if not advanced:
            break
    return order[:n]


def explicit_plan(indices: Sequence[int], total: int) -> List[int]:
    _check_total_n(total, len(indices))
    out = []
    for i in indices:
        i = int(i)
        if not (0 <= i < total):
            raise SamplingError("index {} out of range 0..{}".format(i, total - 1))
        out.append(i)
    return out


def build_plan(mode: str, total: int, n: int,
               labels: Optional[Sequence[int]] = None,
               indices: Optional[Sequence[int]] = None,
               seed: int = 0) -> List[int]:
    """Dispatch to a plan. Returns a list of real dataset indices (frame ids)."""
    if mode == "sequential":
        _check_total_n(total, n)
        return list(range(n))
    if mode == "strided":
        return strided_plan(total, n)
    if mode == "class_balanced":
        if labels is None:
            raise SamplingError("class_balanced needs ground-truth labels")
        return class_balanced_plan(labels, n, seed=seed)
    if mode == "explicit":
        if indices is None:
            raise SamplingError("explicit needs an indices list")
        return explicit_plan(indices, total)
    raise SamplingError("unknown sampling mode: {!r}".format(mode))


def _evenly(seq: Sequence[int], k: int) -> List[int]:
    """k evenly spaced, order-preserving picks from an ascending index list."""
    m = len(seq)
    if k >= m:
        return list(seq)
    step = m / float(k)
    return [seq[min(m - 1, int(round(j * step)))] for j in range(k)]


def _check_total_n(total: int, n: int) -> None:
    if total <= 0:
        raise SamplingError("empty dataset")
    if n <= 0:
        raise SamplingError("requested count must be >= 1")
    if n > total:
        raise SamplingError("requested {} windows but only {} available".format(n, total))
