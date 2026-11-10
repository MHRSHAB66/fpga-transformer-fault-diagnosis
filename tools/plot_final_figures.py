"""Final evaluation figures of the deployed FPGA system (thesis / report).

Every figure is drawn from recorded evidence; nothing is re-measured:

  roc_ovr_board              one-vs-rest ROC + AUC from the 4 physical board
                             logits of all 95,404 test windows
  confusion_matrix_board     board predictions vs labels (row-normalised)
  reliability_confidence     calibration of CONF before/after temperature T
  rt_response_ccdf           hard real-time runs: P(response > x) vs deadline
  e2e_latency_profile        TCP end-to-end mode: board stage vs GUI render
  throughput_vs_requirement  measured inferences/s vs the 48.83 /s requirement
  fpga_resources             post-route utilisation of the argmax build
  cpu_vs_fpga_latency        same INT10 model: board ARM and laptop CPU vs FPGA

    python -m tools.plot_final_figures
    -> docs/figures/final/<name>.png + .pdf
    -> artifacts/results/final_figures_20261109/figure_data.json (all plotted numbers)

The hard real-time raw timings (rt_raw.npz) are gitignored; when they are
missing that figure is skipped and reported, the others are still drawn.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.metrics import (auc, precision_recall_fscore_support,  # noqa: E402
                             roc_auc_score, roc_curve)

from host.confidence import load_calibration, logits_scale  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "artifacts" / "results"
OUT_FIG = REPO / "docs" / "figures" / "final"
OUT_DATA = RES / "final_figures_20261109"

CLASSES = ["normal", "imbalance", "misalignment", "bearing_fault"]
PERIOD_MS = 20.48  # 50 kHz sample rate / 1024-sample stride (src/config.py)

# Evidence files
LABELS = RES / "fpga_fulltest_20261008_141335" / "labels.txt"
BOARD_PRED = RES / "fpga_fulltest_20261008_141335" / "fpga_pred.txt"
BOARD_LOGITS = RES / "fpga_realtime_bench_20261022" / "board_full_pipelined" / "logits_i32.bin"
ARGMAX_FULLTEST = RES / "fpga_argmax_fulltest_20261023" / "live_evidence.json"
REALTIME_VERDICT = RES / "fpga_realtime_bench_20261022" / "realtime_verdict.json"
HARD_RT = RES / "hard_rt_20261024"
E2E_RUNS = {"run 1": RES / "e2e_demo_gui_20261024_231859" / "e2e_verification.json",
            "run 2": RES / "e2e_demo_gui_20261024_233000" / "e2e_verification.json"}
CPU_BASELINE = RES / "cpu_baseline_20261109" / "summary.json"
RAW_RT = RES / "raw_core_board_rt_20261109_234531" / "out" / "summary.json"
UTILIZATION = (RES / "seq256_dma_argmax_z7020_20261022" / "vivado_reports"
               / "post_route_utilization.rpt")

# Reference palette (light mode, validated with the dataviz validator: categorical
# slots 1-4 pass the adjacent CVD/normal-vision gates; aqua/yellow need labels,
# which every chart here carries in its legend or as direct labels).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


# ---------------------------------------------------------------------------
# styling helpers
# ---------------------------------------------------------------------------
def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 24,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK,
        "legend.handlelength": 3.0,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
    })


def subtitle(ax, text):
    """Secondary line between the title (axes.titlepad) and the plot area."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
                textcoords="offset points", ha="left", va="bottom",
                color=INK_2, fontsize=9)


def save(fig, name):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT_FIG / name).relative_to(REPO))


def hbar(ax, y, value, height, color, radius_px=4):
    """Horizontal bar from 0 with a rounded data end and a square baseline."""
    if value <= 0:
        return
    # Convert the 4 px radius to data units; mutation_aspect makes the corner
    # circular on screen although x and y data units differ.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ext = ax.get_window_extent()
    px_per_x = ext.width / abs(x1 - x0)
    px_per_y = ext.height / abs(y1 - y0)
    r = min(radius_px / px_per_x, value / 2)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), value, height, linewidth=0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=px_per_x / px_per_y,
                                facecolor=color))
    ax.add_patch(Rectangle((0, y - height / 2), value - r, height, linewidth=0,
                           facecolor=color))


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_labels():
    return np.loadtxt(LABELS, dtype=np.int64)


def load_board_logits(n):
    logits = np.fromfile(BOARD_LOGITS, dtype="<i4").reshape(-1, 4)
    assert logits.shape[0] == n, (logits.shape, n)
    return logits.astype(np.float64)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_roc(y, logits, data):
    s = logits_scale()
    probs = softmax(logits * s)
    onehot = np.eye(4, dtype=np.int64)[y]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    per_class = {}
    for c, name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(onehot[:, c], probs[:, c])
        a = auc(fpr, tpr)
        per_class[name] = round(float(a), 6)
        # log-x needs FPR > 0; the first point sits at FPR=0 by construction.
        keep = fpr > 0
        ax.plot(fpr[keep], tpr[keep], color=SERIES[c],
                label=f"{name}  (AUC {a:.4f})")
    macro = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    micro = roc_auc_score(onehot.ravel(), probs.ravel())
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0.5, 1.005)
    ax.set_xlabel("False positive rate (log scale)")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC of the deployed FPGA model, one class vs rest")
    subtitle(ax, f"95,404 test windows, physical board logits · macro AUC {macro:.4f} · "
                 f"micro AUC {micro:.4f}")
    ax.legend(loc="lower right")
    save(fig, "roc_ovr_board")
    data["roc_auc"] = {
        "score": "softmax(s_logits * board_logits), s_logits = %.17g (uncalibrated)" % s,
        "per_class_ovr": per_class,
        "macro_ovr": round(float(macro), 6),
        "micro": round(float(micro), 6),
        "n_windows": int(len(y)),
        "logits_source": str(BOARD_LOGITS.relative_to(REPO)).replace("\\", "/"),
    }


def fig_confusion(y, pred, data):
    cm = np.zeros((4, 4), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    rows = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            v = rows[i, j]
            k = int(round(v * (len(BLUE_RAMP) - 1)))
            color = BLUE_RAMP[k] if cm[i, j] else SURFACE
            ax.add_patch(Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, facecolor=color,
                                   linewidth=0))
            ink = "#ffffff" if k >= 6 else INK
            ax.text(j + 0.5, i + 0.45, f"{cm[i, j]:,}", ha="center", va="center",
                    color=ink, fontsize=10, fontweight="semibold")
            ax.text(j + 0.5, i + 0.68, f"{100 * v:.2f} %", ha="center", va="center",
                    color=ink, fontsize=8)
    ax.set_xlim(0, 4)
    ax.set_ylim(4, 0)
    ax.set_xticks(np.arange(4) + 0.5, CLASSES, rotation=15)
    ax.set_yticks(np.arange(4) + 0.5, CLASSES)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Predicted class (decided in FPGA logic)")
    ax.set_ylabel("True class")
    acc = np.trace(cm) / cm.sum()
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)
    ax.set_title("Confusion matrix of the deployed FPGA system")
    subtitle(ax, f"95,404 test windows · accuracy {100 * acc:.2f} % · macro-F1 "
                 f"{100 * f.mean():.2f} % · cell shade = share of the true class")
    save(fig, "confusion_matrix_board")
    data["confusion_matrix"] = {
        "classes": CLASSES, "counts": cm.tolist(),
        "accuracy": round(float(acc), 6),
        "per_class": {CLASSES[c]: {"precision": round(float(p[c]), 6),
                                   "recall": round(float(r[c]), 6),
                                   "f1": round(float(f[c]), 6),
                                   "support": int(sup[c])} for c in range(4)},
        "macro_f1": round(float(f.mean()), 6),
        "pred_source": str(BOARD_PRED.relative_to(REPO)).replace("\\", "/"),
    }


def _reliability(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        c, a, n = conf[m].mean(), correct[m].mean(), int(m.sum())
        ece += n / len(conf) * abs(a - c)
        rows.append({"bin": b, "n": n, "mean_conf": float(c), "accuracy": float(a)})
    return rows, ece


def fig_reliability(y, logits, data):
    cal = load_calibration()
    s = logits_scale()
    T = float(cal["T"]) if cal else 1.0
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0.25, 1], [0.25, 1], color=AXIS, linewidth=1, label="perfect calibration")
    out = {}
    for k, (name, temp) in enumerate((("uncalibrated", 1.0), (f"calibrated, T = {T:.4f}", T))):
        probs = softmax(logits * s / temp)
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        rows, ece = _reliability(conf, (pred == y).astype(np.float64))
        shown = [r for r in rows if r["n"] >= 30]
        ax.plot([r["mean_conf"] for r in shown], [r["accuracy"] for r in shown],
                color=SERIES[k], marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=f"{name}  (ECE {100 * ece:.2f} %)")
        out[name] = {"temperature": temp, "ece_15_bins": round(float(ece), 6), "bins": rows}
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.25, 1.02)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title("Calibration of the displayed confidence (test split)")
    subtitle(ax, "95,404 windows · 15 equal-width bins, bins with n < 30 hidden · "
                 "T fitted on validation only")
    ax.legend(loc="upper left")
    save(fig, "reliability_confidence")
    data["reliability"] = out


def fig_rt_ccdf(data):
    runs = [("R0_normal_prio", "default scheduling, 30,000 windows"),
            ("R1_fifo50_cpu1", "SCHED_FIFO 50 + CPU 1, 30,000 windows"),
            ("L1_fifo50_cpu1_100k", "SCHED_FIFO 50 + CPU 1, 100,000 windows")]
    loaded = []
    for run, label in runs:
        npz = HARD_RT / run / "remote" / "out" / "rt_raw.npz"
        summ = json.loads((HARD_RT / run / "remote" / "out" / "summary.json").read_text())
        if not npz.exists():
            print("skip rt_response_ccdf: missing", npz.relative_to(REPO))
            data["rt_response_ccdf"] = {"skipped": "rt_raw.npz not present (gitignored)"}
            return
        z = np.load(npz)
        warm = int(summ["warmup"])
        resp = (z["done"][warm:] - z["release"][warm:]) * 1e3
        loaded.append((run, label, np.sort(resp), summ))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    out = {}
    for k, (run, label, resp, summ) in enumerate(loaded):
        n = len(resp)
        ccdf = (n - np.arange(n)) / n          # P(response >= x), last point 1/n
        ax.plot(resp, ccdf, color=SERIES[k], label=label, drawstyle="steps-post")
        misses = int((resp > PERIOD_MS).sum())
        out[run] = {"windows": n, "misses_recomputed": misses,
                    "misses_summary": summ["deadline_misses"],
                    "max_ms": round(float(resp[-1]), 3),
                    "p50_ms": round(float(np.percentile(resp, 50)), 3),
                    "p99_9_ms": round(float(np.percentile(resp, 99.9)), 3)}
        assert misses == summ["deadline_misses"], (run, misses, summ["deadline_misses"])
    ax.axvline(PERIOD_MS, color=INK_2, linewidth=1)
    ax.text(PERIOD_MS + 0.3, 2e-5, "deadline 20.48 ms", color=INK_2, fontsize=9, va="center")
    ax.set_yscale("log")
    ax.set_xlim(15.5, 45)
    ax.set_ylim(5e-6, 1.2)
    ax.set_xlabel("Response time: scheduled release to FPGA class ready (ms)")
    ax.set_ylabel("Share of windows with response ≥ x")
    ax.set_title("Board-side hard real-time test")
    r0 = out["R0_normal_prio"]
    subtitle(ax, f"Prepared windows in RAM, one every 20.48 ms · misses: "
                 f"{r0['misses_recomputed']} with default scheduling, 0 with SCHED_FIFO")
    ax.legend(loc="upper right")
    save(fig, "rt_response_ccdf")
    data["rt_response_ccdf"] = out


def fig_e2e_profile(data):
    keys = ["p50", "p95", "p99", "p99.9", "p99.99", "max"]
    runs = {name: json.loads(p.read_text()) for name, p in E2E_RUNS.items()}
    ev = runs["run 2"]
    board = [ev["stage1_release_to_ready_ms"][k] for k in keys]
    e2e = [ev["e2e_release_to_render_ms"][k] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, (vals, label) in enumerate(((board, "board: release to FPGA class ready"),
                                       (e2e, "end to end: release to GUI render"))):
        ax.plot(x, vals, color=SERIES[k], marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.text(x[-1] + 0.12, vals[-1], f"{vals[-1]:.1f} ms", color=INK_2, fontsize=9,
                va="center")
    ax.axhline(PERIOD_MS, color=INK_2, linewidth=1)
    ax.text(len(keys) - 1.55, PERIOD_MS * 1.04, "deadline 20.48 ms", color=INK_2,
            fontsize=9, va="bottom", ha="right")
    ax.set_yscale("log")
    ticks = [15, 20, 30, 50, 100, 150]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(14, 170)
    ax.set_xticks(x, keys)
    ax.set_xlim(-0.3, len(keys) - 0.4)
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_xlabel("Percentile over 30,000 windows")
    ax.set_title("TCP end-to-end demo mode: board stage vs full path")
    late = ev.get("e2e_misses_with_bound")
    subtitle(ax, f"Run 2 · board stage 0 misses · {late:,} of 30,000 windows late at the GUI "
                 "(Ethernet + Windows + Tk not time-bounded)")
    ax.legend(loc="upper left")
    save(fig, "e2e_latency_profile")
    data["e2e_latency_profile"] = {
        name: {"stage1_release_to_ready_ms": r["stage1_release_to_ready_ms"],
               "e2e_release_to_render_ms": r["e2e_release_to_render_ms"],
               "late_with_clock_bound": r.get("e2e_misses_with_bound"),
               "max_clock_err_ms": r.get("max_clock_err_ms")} for name, r in runs.items()}


def fig_throughput(data):
    v = json.loads(REALTIME_VERDICT.read_text())
    board = v["measured_board_only_full_test"]
    e2e = v["measured_end_to_end"]["runs"]
    argmax = json.loads(ARGMAX_FULLTEST.read_text())
    serial_e2e = [e2e[k]["pc_receive_throughput_inf_s"]
                  for k in ("before_head_runner", "before_head_runner_repeat")]
    overlap_e2e = [e2e[k]["pc_receive_throughput_inf_s"]
                   for k in ("after_pipelined_runner", "final_runner_sequential10000")]
    rows = [
        ("Board only, serial loop (95,404 windows)", board["baseline_serial_loop"]["throughput_inf_s"]),
        ("Board only, overlapped read (95,404 windows)", board["pipelined_loop"]["throughput_inf_s"]),
        ("Board only, FPGA argmax overlay (95,404 windows)", argmax["throughput_inferences_per_second"]),
        ("Board to PC over SSH, serial (mean of 2 × 10,000)", float(np.mean(serial_e2e))),
        ("Board to PC over SSH, overlapped (mean of 2 × 10,000)", float(np.mean(overlap_e2e))),
    ]
    req = v["requirement"]["required_windows_per_s"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 72)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    fig.canvas.draw()
    for i, (label, val) in enumerate(rows):
        hbar(ax, i, val, 0.42, SERIES[0])
        ax.text(val + 0.8, i, f"{val:.1f}", va="center", color=INK, fontsize=9)
    ax.axvline(req, color=INK_2, linewidth=1)
    ax.text(req - 0.6, -0.55, f"required {req:.2f} /s", color=INK_2, fontsize=9,
            ha="right", va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Windows classified per second (sustained average)")
    ax.set_title("Measured throughput on the real board vs the real-time requirement")
    subtitle(ax, "Requirement = 50 kHz / 1,024-sample stride; prepared windows, Tk rendering excluded")
    save(fig, "throughput_vs_requirement")
    data["throughput_inf_s"] = {"required": req, "rows": dict(rows),
                                "e2e_serial_runs": serial_e2e, "e2e_overlap_runs": overlap_e2e}


def fig_resources(data):
    text = UTILIZATION.read_text()
    wanted = [("LUT", r"\| Slice LUTs\s+\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|"),
              ("FF", r"\| Slice Registers\s+\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|"),
              ("BRAM tiles", r"\| Block RAM Tile\s+\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|"),
              ("DSP48E1", r"\| DSPs\s+\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|")]
    rows = []
    for name, pat in wanted:
        m = re.search(pat, text)
        used, avail = float(m.group(1)), float(m.group(2))
        rows.append((name, used, avail, 100 * used / avail))
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.set_xlim(0, 112)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    fig.canvas.draw()
    for i, (name, used, avail, pct) in enumerate(rows):
        hbar(ax, i, pct, 0.42, SERIES[0])
        u = f"{used:,.1f}" if used % 1 else f"{used:,.0f}"
        ax.text(pct + 1, i, f"{pct:.1f} %  ({u} of {avail:,.0f})", va="center",
                color=INK, fontsize=9)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of the XC7Z020 device used (%)")
    ax.set_title("FPGA resources of the deployed design (post-route)")
    subtitle(ax, "Transformer core + AXI DMA + Zynq PS glue, 100 MHz, timing met (WNS +0.250 ns)")
    save(fig, "fpga_resources")
    data["fpga_resources_post_route"] = {r[0]: {"used": r[1], "available": r[2],
                                                "percent": round(r[3], 2)} for r in rows}


def fig_cpu_vs_fpga(data):
    cpu = json.loads(CPU_BASELINE.read_text())["runs"]
    r1 = json.loads((HARD_RT / "R1_fifo50_cpu1" / "remote" / "out" / "summary.json").read_text())
    raw = json.loads(RAW_RT.read_text())
    rows = [  # (label, p50 ms, is_fpga)
        ("Board ARM Cortex-A9, 1 core, -O2", cpu["board/O2_single"]["timing_us"]["p50_us"] / 1e3, False),
        ("Board ARM Cortex-A9, 1 core, -O3 + NEON", cpu["board/O3neon_single"]["timing_us"]["p50_us"] / 1e3, False),
        ("FPGA, INT10 input (incl. DMA)", r1["response_ms"]["p50"], True),
        ("FPGA, raw input, preprocessing on chip (incl. DMA)", raw["response_ms"]["p50"], True),
        ("Laptop i7-13620H, -O2 (reference)", cpu["pc/O2_single"]["timing_us"]["p50_us"] / 1e3, False),
        ("Laptop i7-13620H, -O3 native (reference)", cpu["pc/O3native_single"]["timing_us"]["p50_us"] / 1e3, False),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(0, 260)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    fig.canvas.draw()
    for i, (label, ms, is_fpga) in enumerate(rows):
        hbar(ax, i, ms, 0.42, SERIES[0] if is_fpga else SERIES[1])
        ax.text(ms + 2, i, "%.1f ms" % ms, va="center", color=INK, fontsize=9, zorder=3,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.axvline(PERIOD_MS, color=INK_2, linewidth=1, zorder=1)
    ax.text(PERIOD_MS + 2, -0.55, "deadline 20.48 ms", color=INK_2, fontsize=9, va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time per window, median (ms)")
    ax.set_title("Same INT10 model: FPGA vs CPUs")
    subtitle(ax, "Identical integer computation: logits equal on all 256 class-balanced windows")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=SERIES[0]),
                       Rectangle((0, 0), 1, 1, facecolor=SERIES[1])],
              labels=["FPGA (board)", "CPU"], loc="lower right", handlelength=1.2)
    save(fig, "cpu_vs_fpga_latency")
    data["cpu_vs_fpga_latency_p50_ms"] = {r[0]: round(r[1], 3) for r in rows}


def main():
    style()
    data = {"generated_by": "tools/plot_final_figures.py"}
    y = load_labels()
    pred = np.loadtxt(BOARD_PRED, dtype=np.int64)
    logits = load_board_logits(len(y))
    assert (logits.argmax(axis=1) == pred).all(), "board logits and predictions disagree"
    fig_roc(y, logits, data)
    fig_confusion(y, pred, data)
    fig_reliability(y, logits, data)
    fig_rt_ccdf(data)
    fig_e2e_profile(data)
    fig_throughput(data)
    fig_resources(data)
    fig_cpu_vs_fpga(data)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "figure_data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", (OUT_DATA / "figure_data.json").relative_to(REPO))


if __name__ == "__main__":
    main()
