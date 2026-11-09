"""Per-window comparison: physical board vs HLS C-model vs Python INT10 references.

All sources were evaluated on the SAME input file the board consumed
(artifacts/results/fpga_fulltest_20261008_141335/fpga_testset_in.i32.gz):

  board        fpga_pred.txt (2026-10-08 physical run; classes only)
  board_logits optional: per-window logits from a later physical run
               (tools/board/fpga_stage_bench.py logits_i32.bin)
  csim         g++ C-model of the deployed HLS source (dsp_margin top)
  csim_argmax  g++ C-model of the argmax variant (logits + pred_class register)
  ref_export   Python INT10 reference, export scales (20-file calibration,
               the scales the HLS weights / bitstream were generated from)
  ref_calib12  Python INT10 reference, 12-file calibration (the run behind the
               previously reported 97.7359 % "software INT10 reference")

Writes <dir>/comparison.json and <dir>/disagreements_board_vs_calib12.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "artifacts" / "results" / "fpga_fulltest_20261008_141335"


def _logits(path: Path, cols: int = 4) -> np.ndarray:
    a = np.loadtxt(path, dtype=np.int64)
    assert (a[:, 0] == np.arange(len(a))).all(), path
    return a[:, 1:1 + cols] if cols else a[:, 1:]


def _metrics(pred, y):
    cm = np.zeros((4, 4), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {"n": int(len(y)), "correct": int(tp.sum()), "accuracy": round(tp.sum() / len(y), 6),
            "macro_f1": round(float(f1.mean()), 6), "confusion_matrix": cm.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=ROOT / "artifacts" / "results" / "board_vs_reference_20261022")
    ap.add_argument("--board-logits", type=Path, default=None)
    args = ap.parse_args()
    d = args.dir
    y = np.loadtxt(FT / "labels.txt", dtype=np.int64)
    board = np.loadtxt(FT / "fpga_pred.txt", dtype=np.int64)
    csim = _logits(d / "csim_dsp_margin" / "csim_logits.txt")
    cam = _logits(d / "csim_argmax" / "csim_logits.txt", cols=5)
    rex = _logits(d / "numpy_reference" / "numpy_export_logits.txt")
    r12 = _logits(d / "numpy_reference" / "numpy_calib12_logits.txt")
    n = len(y)
    assert all(len(a) == n for a in (board, csim, cam, rex, r12))

    def cls(lg):
        return lg.argmax(1)  # lowest index wins ties (numpy.argmax)

    tied = (rex == rex.max(1, keepdims=True)).sum(1) > 1
    s_ex = json.loads((d / "numpy_reference" / "numpy_export_scales.json").read_text())
    s_12 = json.loads((d / "numpy_reference" / "numpy_calib12_scales.json").read_text())
    res = {
        "input": str((FT / "fpga_testset_in.i32.gz").relative_to(ROOT)),
        "n_windows": n,
        "same_input_all_sources": True,
        "metrics": {
            "board_2026-10-08": _metrics(board, y),
            "csim_deployed_top": _metrics(cls(csim), y),
            "csim_argmax_register": _metrics(cam[:, 4], y),
            "ref_export_scales": _metrics(cls(rex), y),
            "ref_calib12_scales": _metrics(cls(r12), y),
        },
        "agreement": {
            "board_class_vs_csim_class": int((board == cls(csim)).sum()),
            "board_class_vs_ref_export_class": int((board == cls(rex)).sum()),
            "csim_logits_vs_ref_export_logits_all4": int((csim == rex).all(1).sum()),
            "argmax_top_logits_vs_deployed_top_logits_all4": int((cam[:, :4] == csim).all(1).sum()),
            "argmax_register_vs_board_class": int((cam[:, 4] == board).sum()),
            "argmax_register_vs_numpy_argmax_of_logits": int((cam[:, 4] == cls(cam[:, :4])).sum()),
            "board_class_vs_ref_calib12_class": int((board == cls(r12)).sum()),
        },
        "tied_max_logit_windows_export": int(tied.sum()),
        "calibration_difference": {
            "scales_differing": sorted(k for k in s_ex if abs(s_ex[k] - s_12[k]) > 1e-12 * abs(s_ex[k])),
            "input_scale_equal": s_ex["input"] == s_12["input"],
        },
    }
    if args.board_logits is not None:
        bl = np.fromfile(args.board_logits, dtype=np.int32).reshape(-1, 4).astype(np.int64)
        m = len(bl)
        res["board_logits_physical"] = {
            "source": str(args.board_logits), "n": m,
            "vs_csim_logits_all4": int((bl == csim[:m]).all(1).sum()),
            "vs_ref_export_logits_all4": int((bl == rex[:m]).all(1).sum()),
            "class_vs_board_2026-10-08": int((bl.argmax(1) == board[:m]).sum()),
        }
    dis = np.where(board != cls(r12))[0]
    bc = int((board[dis] == y[dis]).sum())
    kc = int((cls(r12)[dis] == y[dis]).sum())
    res["board_vs_calib12_disagreements"] = {
        "count": int(len(dis)), "board_correct": bc, "calib12_correct": kc,
        "neither_correct": int(len(dis)) - bc - kc, "net_correct_difference": bc - kc,
    }
    with open(d / "disagreements_board_vs_calib12.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("window,true,board,ref_export,ref_calib12,export_logits,calib12_logits\n")
        for i in dis:
            f.write("%d,%d,%d,%d,%d,%s,%s\n" % (i, y[i], board[i], cls(rex)[i], cls(r12)[i],
                                                ";".join(map(str, rex[i])), ";".join(map(str, r12[i]))))
    (d / "comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["agreement"], indent=2))
    print(json.dumps(res["board_vs_calib12_disagreements"], indent=2))
    if "board_logits_physical" in res:
        print(json.dumps(res["board_logits_physical"], indent=2))


if __name__ == "__main__":
    main()
