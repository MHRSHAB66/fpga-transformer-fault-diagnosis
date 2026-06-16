from __future__ import annotations

import numpy as np


def precision_recall_fbeta_from_confusion(
    confusion: np.ndarray,
    class_names: list[str],
    beta: float = 2.0,
) -> dict:
    """
    Compute per-class precision/recall/F-beta plus macro/weighted averages.
    """
    matrix = np.asarray(confusion, dtype=np.int64)
    beta2 = float(beta) ** 2
    n = matrix.shape[0]

    per_class = {}
    precisions = []
    recalls = []
    fbetas = []
    supports = []

    for idx in range(n):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        support = int(matrix[idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = (beta2 * precision) + recall
        fbeta = ((1.0 + beta2) * precision * recall / denom) if denom > 0 else 0.0

        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            f"f{beta:g}": float(fbeta),
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        fbetas.append(fbeta)
        supports.append(support)

    total = int(sum(supports))
    weighted = {
        "precision": float(np.average(precisions, weights=supports)) if total else 0.0,
        "recall": float(np.average(recalls, weights=supports)) if total else 0.0,
        f"f{beta:g}": float(np.average(fbetas, weights=supports)) if total else 0.0,
    }

    return {
        "per_class": per_class,
        "macro": {
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            f"f{beta:g}": float(np.mean(fbetas)) if fbetas else 0.0,
        },
        "weighted": weighted,
    }


def fault_vs_normal_summary(
    confusion: np.ndarray,
    normal_class_idx: int = 0,
) -> dict:
    """
    Collapse a multiclass confusion matrix into fault-vs-normal safety metrics.

    captured_faults: actual fault windows predicted as any fault class.
    missed_faults: actual fault windows predicted as normal.
    false_alarms: actual normal windows predicted as any fault class.
    """
    matrix = np.asarray(confusion, dtype=np.int64)
    all_indices = np.arange(matrix.shape[0])
    fault_indices = all_indices[all_indices != normal_class_idx]

    captured_faults = int(matrix[np.ix_(fault_indices, fault_indices)].sum())
    missed_faults = int(matrix[fault_indices, normal_class_idx].sum())
    false_alarms = int(matrix[normal_class_idx, fault_indices].sum())
    actual_faults = captured_faults + missed_faults
    actual_normal = int(matrix[normal_class_idx, :].sum())

    return {
        "captured_faults": captured_faults,
        "missed_faults": missed_faults,
        "false_alarms": false_alarms,
        "actual_faults": actual_faults,
        "actual_normal": actual_normal,
        "fault_recall": (
            float(captured_faults / actual_faults) if actual_faults > 0 else 0.0
        ),
        "false_alarm_rate": (
            float(false_alarms / actual_normal) if actual_normal > 0 else 0.0
        ),
    }
