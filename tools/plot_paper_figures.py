"""Paper-analysis figures (2026-11-10), drawn from recorded evidence only:

  noise_robustness       accuracy and normal-class recall vs SNR: the deployed INT10
                         model (C-model, equal to the board) and, when
                         noise_eval.json exists, the float control and the
                         noise-augmented float model on the same noisy windows
  dead_sensor            accuracy of the deployed model when one channel reads 0
  generalization_splits  same architecture and recipe retrained on stricter splits
  temporal_vote          majority vote over the last k windows (board predictions,
                         per recording); FPGA vote result added when available

    python -m tools.plot_paper_figures
    -> docs/figures/paper/<name>.png + .pdf
    -> artifacts/results/paper_figures_20261110/figure_data.json (all plotted numbers)
Figures whose evidence is missing are skipped and reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from tools.plot_final_figures import INK, INK_2, MUTED, SERIES, SURFACE, hbar, style, subtitle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "artifacts" / "results"
OUT_FIG = REPO / "docs" / "figures" / "paper"
OUT_DATA = RES / "paper_figures_20261110"

ROBUSTNESS = RES / "robustness_20261110" / "summary.json"
GENERALIZATION = RES / "generalization_20261110"
VOTE = RES / "paper_analysis_20261110" / "temporal_vote.json"

SNR_ORDER = ["clean", "snr30", "snr20", "snr10", "snr5", "snr0", "snr-5"]
CHANNELS = ["tachometer", "bearing 1 axial", "bearing 1 radial", "bearing 1 tangential",
            "bearing 2 axial", "bearing 2 radial", "bearing 2 tangential", "microphone"]
SPLITS = [("random", "Project split\n(control)"), ("speed_mid", "Unseen speed\nband 37-47 Hz"),
          ("sev_mid", "Unseen middle\nseverity"), ("sev_max_nmid", "Unseen highest\nseverity"),
          ("sev_max", "Unseen highest\nseverity; normal\nat unseen speeds")]


def save(fig, name):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT_FIG / name).relative_to(REPO))


def snr_label(cond):
    return "clean" if cond == "clean" else cond[3:].replace("-", "−")


def fig_noise(data):
    dep = json.loads(ROBUSTNESS.read_text())
    series = [("Deployed INT10 (= board)", {c: dep[c] for c in SNR_ORDER if c in dep}, SERIES[0], "o")]
    noise_eval = GENERALIZATION / "noise_eval.json"
    if noise_eval.exists():
        ne = json.loads(noise_eval.read_text())
        if "random" in ne:
            series.append(("Float, retrained (control)", ne["random"], SERIES[1], "s"))
        if "random_noise" in ne:
            series.append(("Float, noise-augmented training", ne["random_noise"], SERIES[2], "D"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharex=True)
    for ax, (title, get) in zip(axes, (("Accuracy", lambda r: r["accuracy"]),
                                       ("Recall of the normal class", lambda r: r["recall_per_class"][0]))):
        for label, rows, color, marker in series:
            xs = [i for i, c in enumerate(SNR_ORDER) if c in rows]
            ys = [100 * get(rows[c]) for c in SNR_ORDER if c in rows]
            ax.plot(xs, ys, color=color, marker=marker, markersize=4.5, label=label)
        ax.set_xticks(range(len(SNR_ORDER)), [snr_label(c) for c in SNR_ORDER])
        ax.set_ylim(0, 102)
        ax.set_title(title)
        ax.set_xlabel("SNR per channel (dB)")
    axes[0].set_ylabel("%")
    subtitle(axes[0], "White Gaussian noise on the raw ADC codes, all 95,404 test windows")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "noise_robustness")
    data["noise_robustness"] = {label: {c: {"accuracy": rows[c]["accuracy"],
                                            "normal_recall": rows[c]["recall_per_class"][0]}
                                        for c in SNR_ORDER if c in rows}
                                for label, rows, _, _ in series}


def fig_dead(data):
    dep = json.loads(ROBUSTNESS.read_text())
    rows = [(CHANNELS[c], dep["dead%d" % c]["accuracy"] * 100) for c in range(8) if "dead%d" % c in dep]
    if "dead0+7" in dep:
        rows.append(("tachometer + microphone", dep["dead0+7"]["accuracy"] * 100))
    clean = dep["clean"]["accuracy"] * 100
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.set_xlim(0, 105)
    ax.set_ylim(len(rows) + 0.1, -0.6)
    ax.grid(axis="y", visible=False)
    fig.canvas.draw()
    for i, (label, acc) in enumerate(rows):
        hbar(ax, i, acc, 0.5, SERIES[0])
        ax.text(acc + 1, i, "%.1f %%" % acc, va="center", color=INK, fontsize=9, zorder=3,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
    ax.axvline(clean, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(clean - 1, len(rows) - 0.25, "all sensors %.1f %%" % clean, color=INK_2, fontsize=9,
            ha="right", va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Accuracy when one sensor reads zero (deployed model, no retraining)")
    subtitle(ax, "C-model bit-exact with the board; all 95,404 test windows")
    save(fig, "dead_sensor")
    data["dead_sensor_accuracy_pct"] = {r[0]: round(r[1], 2) for r in rows}
    data["dead_sensor_accuracy_pct"]["all sensors"] = round(clean, 2)


def fig_generalization(data):
    have = [(k, label, json.loads((GENERALIZATION / k / "result.json").read_text()))
            for k, label in SPLITS if (GENERALIZATION / k / "result.json").exists()]
    if not have:
        print("skipped generalization_splits: no result.json")
        return
    metrics = [("Accuracy", lambda t: t["accuracy"]), ("Macro-F1", lambda t: t["macro_f1"]),
               ("Normal-class recall", lambda t: t["recall_per_class"][0])]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    width = 0.24
    for m, (name, get) in enumerate(metrics):
        xs = [i + (m - 1) * width for i in range(len(have))]
        vals = [100 * get(r["test"]) for _, _, r in have]
        ax.bar(xs, vals, width * 0.92, color=SERIES[m], label=name, zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, "%.1f" % v, ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(have)), [label for _, label, _ in have])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Test (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Same architecture and recipe, stricter file-level splits (float)")
    subtitle(ax, "Normal files have no severity: severity splits hold out a speed band of the normal class "
                 "(middle band; ≥ 52 Hz in the last group)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, ncol=3)
    save(fig, "generalization_splits")
    data["generalization_splits"] = {k: {"label": label.replace("\n", " "), "files": r["files"], "best_epoch": r["best_epoch"],
                                         **{n: round(g(r["test"]), 6) for n, g in metrics}}
                                     for k, label, r in have}


def board_vote():
    """Newest board full test that read the FPGA vote registers."""
    for p in sorted(RES.glob("raw_core_board_fulltest_*/fulltest_check.json"), reverse=True):
        r = json.loads(p.read_text())
        if "temporal_vote" in r:
            return p, r["temporal_vote"]
    return None, None


def fig_vote(data):
    v = json.loads(VOTE.read_text())
    ks = sorted(int(k[2:]) for k in v if k.startswith("k="))
    acc = [v["k=%d" % k]["accuracy"] for k in ks]
    normal = [v["k=%d" % k]["recall_per_class"][0] for k in ks]
    ms = [v["k=%d" % k]["decision_window_ms"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.plot(range(len(ks)), acc, color=SERIES[0], marker="o", markersize=4.5, label="Accuracy")
    ax.plot(range(len(ks)), normal, color=SERIES[1], marker="s", markersize=4.5, label="Normal-class recall")
    ax.set_xticks(range(len(ks)), ["k=%d\n%.0f ms" % (k, m) for k, m in zip(ks, ms)])
    ax.set_ylim(85, 100.8)
    ax.set_ylabel("%")
    ax.set_xlabel("Windows in the vote (decision window)")
    ax.set_title("Majority vote over the last k windows (physical board predictions)")
    subtitle(ax, "Causal vote inside each recording; ties go to the newest window")
    plotted = {"per_recording": {str(k): {"accuracy": a, "normal_recall": n, "decision_window_ms": m}
                                 for k, a, n, m in zip(ks, acc, normal, ms)}}
    src, b = board_vote()
    if b is not None and b.get("k") in ks:
        x = ks.index(b["k"])
        ax.plot([x], [100 * b["system_accuracy"]], linestyle="none", marker="*", markersize=11,
                color=SERIES[2], label="FPGA vote, one continuous stream (k=%d)" % b["k"])
        plotted["fpga_continuous_stream"] = dict(b, evidence=str(src.relative_to(REPO)))
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "temporal_vote")
    data["temporal_vote"] = plotted


def main():
    style()
    data = {"generated_by": "tools/plot_paper_figures.py"}
    for f in (fig_noise, fig_dead, fig_generalization, fig_vote):
        f(data)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "figure_data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", (OUT_DATA / "figure_data.json").relative_to(REPO))


if __name__ == "__main__":
    main()
