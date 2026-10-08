"""Stage 11 -- replay data + metrics for the FPGA full-test result set.

This is the *display/scoring* layer behind the result viewer. It never
classifies anything: predictions come verbatim from the FPGA run
(``fpga_pred.txt``) and ground truth from ``labels.txt``. It only pairs them up,
flags correct/wrong, and aggregates the same metrics the board run already
reported in ``fpga_fulltest_metrics.json`` -- so the viewer can be checked
against that file byte-for-byte (within rounding).

Pure standard library (no numpy / torch) so the PC viewer stays dependency-light
and the parsing/metrics are trivially unit-testable.

Confusion-matrix convention matches ``fpga_fulltest_metrics.json``:
rows = TRUE class, columns = PREDICTED class (``cm[true][pred]``).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from host.frame_protocol import LABELS  # single source of truth for class<->label

NUM_CLASSES = len(LABELS)


class DataError(Exception):
    """Raised for malformed / mismatched / out-of-range replay inputs."""


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------
def load_int_file(path: str) -> List[int]:
    """Read a one-integer-per-line class file (blank lines ignored).

    Raises DataError with a line number on any non-integer or out-of-range code.
    """
    if not os.path.isfile(path):
        raise DataError("file not found: {}".format(path))
    out: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            s = raw.strip()
            if not s:
                continue
            try:
                v = int(s)
            except ValueError:
                raise DataError("non-integer class {!r} at {}:{}".format(s, path, lineno))
            if v not in LABELS:
                raise DataError("class {} out of range 0..{} at {}:{}"
                                .format(v, NUM_CLASSES - 1, path, lineno))
            out.append(v)
    return out


def load_meta(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Record pairing
# ---------------------------------------------------------------------------
class ReplayDataset:
    """Paired (prediction, ground-truth) records for navigation/replay.

    ``preds`` and ``labels`` must be equal-length lists of valid class ids.
    """

    def __init__(self, preds: List[int], labels: List[int],
                 meta: Optional[dict] = None):
        if len(preds) != len(labels):
            raise DataError("prediction/label count mismatch: {} preds vs {} labels"
                            .format(len(preds), len(labels)))
        if not preds:
            raise DataError("empty dataset (0 records)")
        self.preds = preds
        self.labels = labels
        self.meta = meta or {}

    @property
    def n(self) -> int:
        return len(self.preds)

    @classmethod
    def from_dir(cls, result_dir: str) -> "ReplayDataset":
        """Load the canonical fpga_pred.txt / labels.txt / meta.json triple."""
        preds = load_int_file(os.path.join(result_dir, "fpga_pred.txt"))
        labels = load_int_file(os.path.join(result_dir, "labels.txt"))
        meta = load_meta(os.path.join(result_dir, "meta.json"))
        return cls(preds, labels, meta)

    def record(self, i: int) -> Dict[str, object]:
        """One display record. Confidence is always None (FPGA emits none)."""
        if not (0 <= i < self.n):
            raise IndexError("frame {} out of range 0..{}".format(i, self.n - 1))
        pred = self.preds[i]
        true = self.labels[i]
        return {
            "frame": i,
            "class_id": pred,
            "label": LABELS[pred],
            "true_class": true,
            "true_label": LABELS[true],
            "correct": pred == true,
            "conf": None,  # hardware produced no confidence -> UI shows N/A
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _macro(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class MetricsAccumulator:
    """Incremental confusion matrix + counters, matching the board's metrics.

    Fed either per record (``add``) for the live/stepping UI, or in one shot
    (``add_all``) for Fast evaluation. ``finalize`` reproduces accuracy, macro
    and weighted precision/recall/F1, per-class values, support and the
    confusion matrix exactly as in fpga_fulltest_metrics.json.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.k = num_classes
        self.cm = [[0] * num_classes for _ in range(num_classes)]  # cm[true][pred]
        self.correct = 0
        self.total = 0
        self.by_pred = [0] * num_classes

    def add(self, true: int, pred: int) -> None:
        self.cm[true][pred] += 1
        self.by_pred[pred] += 1
        self.total += 1
        if true == pred:
            self.correct += 1

    def add_all(self, trues: List[int], preds: List[int]) -> None:
        for t, p in zip(trues, preds):
            self.add(t, p)

    @property
    def running_accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def incorrect(self) -> int:
        return self.total - self.correct

    def predicted_counts(self) -> Dict[int, int]:
        return {c: self.by_pred[c] for c in range(self.k)}

    def finalize(self) -> dict:
        k = self.k
        support = [sum(self.cm[t]) for t in range(k)]          # true totals (rows)
        pred_tot = [sum(self.cm[t][p] for t in range(k)) for p in range(k)]  # cols
        tp = [self.cm[c][c] for c in range(k)]
        precision = [(tp[c] / pred_tot[c]) if pred_tot[c] else 0.0 for c in range(k)]
        recall = [(tp[c] / support[c]) if support[c] else 0.0 for c in range(k)]
        f1 = [(2 * precision[c] * recall[c] / (precision[c] + recall[c]))
              if (precision[c] + recall[c]) else 0.0 for c in range(k)]
        total = sum(support)
        accuracy = (sum(tp) / total) if total else 0.0

        def weighted(vals: List[float]) -> float:
            return (sum(vals[c] * support[c] for c in range(k)) / total) if total else 0.0

        return {
            "test_windows": total,
            "correct": sum(tp),
            "accuracy": accuracy,
            "macro_precision": _macro(precision),
            "macro_recall": _macro(recall),
            "macro_f1": _macro(f1),
            "weighted_precision": weighted(precision),
            "weighted_recall": weighted(recall),
            "weighted_f1": weighted(f1),
            "per_class_precision": precision,
            "per_class_recall": recall,
            "per_class_f1": f1,
            "support": support,
            "confusion_matrix": [row[:] for row in self.cm],
        }


def compare_to_reference(computed: dict, reference: dict, tol: float = 1e-6) -> List[str]:
    """Return a list of human-readable discrepancies (empty == perfect match).

    Compares the scalar/array metrics present in both dicts within ``tol``.
    Unexpected differences are surfaced, never hidden.
    """
    diffs: List[str] = []
    scalar_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                   "weighted_precision", "weighted_recall", "weighted_f1",
                   "test_windows", "correct"]
    for key in scalar_keys:
        if key not in reference:
            continue
        a, b = computed.get(key), reference.get(key)
        if a is None:
            continue
        if abs(float(a) - float(b)) > tol:
            diffs.append("{}: computed={} reference={} (|d|={:.3g})"
                         .format(key, a, b, abs(float(a) - float(b))))
    for key in ["per_class_precision", "per_class_recall", "per_class_f1", "support"]:
        if key not in reference:
            continue
        ca, cb = computed.get(key) or [], reference.get(key) or []
        if len(ca) != len(cb):
            diffs.append("{}: length {} vs {}".format(key, len(ca), len(cb)))
            continue
        for i, (a, b) in enumerate(zip(ca, cb)):
            if abs(float(a) - float(b)) > tol:
                diffs.append("{}[{}]: computed={} reference={}".format(key, i, a, b))
    if "confusion_matrix" in reference and "confusion_matrix" in computed:
        if computed["confusion_matrix"] != reference["confusion_matrix"]:
            diffs.append("confusion_matrix differs from reference")
    return diffs
