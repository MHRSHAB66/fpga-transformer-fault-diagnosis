"""Optional confidence derived from the FPGA's own INT10 logits (PC side).

    confidence = max_c softmax(s_logits * logits_int)[c]

* ``logits_int``: the 4 signed integers streamed out by the deployed core.
* ``s_logits``  : the exported logits quantization scale of that exact model
  (deployment bundle config/integer_config_scales.json -> scales.logits).

This is the model's own (uncalibrated) softmax probability of the class the
FPGA chose; it does not decide or change the class. Measured calibration (ECE,
reliability) is in
artifacts/results/board_vs_reference_20261022/confidence/confidence_calibration.json.

Calibrated display (2026-10-24): ``display_confidence`` divides the scaled logits
by a single temperature T fitted on the VALIDATION split only
(artifacts/calibration/conf_temperature_v1/, test evaluated afterwards). T > 0
never changes the arg-max, i.e. the FPGA's class. The calibration is used only if
its recorded logits scale equals the deployed one; otherwise the uncalibrated
value is shown and labelled as such.
When no logits are available (older runner, replay of runs that did not record
logits) the caller must keep CONF=NA.
"""
import json
import math
import os
from typing import Optional, Sequence

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Scales of the CURRENTLY deployed overlay (argmax build, 2026-10-22). Its
# logits scale is identical to the dma16 bundle's and to the HLS export
# (artifacts/quantized/f_seq256_int10/integer_config.json); tests/test_confidence.py
# checks that, so a re-export with different scales cannot slip through.
SCALES_PATH = os.path.join(_REPO, "artifacts", "deploy", "seq256_argmax_z7020_20261022",
                           "config", "integer_config_scales.json")
_SCALE = None
CALIBRATION_PATH = os.path.join(_REPO, "artifacts", "calibration", "conf_temperature_v1",
                                "temperature.json")
_CAL = {}


def logits_scale() -> float:
    global _SCALE
    if _SCALE is None:
        with open(SCALES_PATH, encoding="utf-8") as f:
            _SCALE = float(json.load(f)["scales"]["logits"])
    return _SCALE


def load_calibration(path: Optional[str] = None) -> Optional[dict]:
    """Frozen temperature calibration, or None if missing/invalid/mismatched."""
    path = path or CALIBRATION_PATH
    if path in _CAL:
        return _CAL[path]
    cal = None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        T = float(d["T"])
        if T > 0 and math.isfinite(T) and float(d["s_logits"]) == logits_scale():
            cal = {"T": T, "path": path}
    except (OSError, ValueError, KeyError, TypeError):
        cal = None
    _CAL[path] = cal
    return cal


def display_confidence(logits: Sequence[int], cls: int,
                       calibration_path: Optional[str] = None):
    """(value, kind) for display: kind is "calibrated_T=<T>" when the frozen
    calibration applies to the deployed model, else "uncalibrated"."""
    cal = load_calibration(calibration_path)
    if cal is None:
        return confidence_from_logits(logits, cls), "uncalibrated"
    return (confidence_from_logits(logits, cls, temperature=cal["T"]),
            "calibrated_T=%.4f" % cal["T"])


def confidence_from_logits(logits: Sequence[int], cls: int,
                           scale: Optional[float] = None,
                           temperature: float = 1.0) -> float:
    """Softmax probability of ``cls`` from the 4 hardware integer logits.

    ``cls`` must be the FPGA-path class (lowest-index argmax of the same
    logits); a mismatch means the inputs do not belong together -> ValueError.
    """
    if len(logits) != 4:
        raise ValueError("expected 4 logits, got %d" % len(logits))
    vals = [int(v) for v in logits]
    best = max(range(4), key=lambda c: (vals[c], -c))
    if best != cls:
        raise ValueError("class %d is not the argmax of logits %s" % (cls, vals))
    s = (logits_scale() if scale is None else float(scale)) / float(temperature)
    m = max(vals)
    ex = [math.exp((v - m) * s) for v in vals]
    return ex[cls] / sum(ex)
