"""
Direct-run diagnostic for the data pipeline.

Builds the train/val/test DataLoaders via ``build_dataloaders``, prints the stats
dict and timing, saves lightweight summary bar charts (file counts, sample counts,
timing) under ``artifacts/results/``, and shows one training batch.

This was extracted from ``src/data/make_dataloaders.py`` so that module stays a
clean importable library (no matplotlib / plotting concerns). Behavior is
identical to running ``python -m src.data.make_dataloaders`` previously.

Usage
-----
    python -m src.tools.inspect_dataloaders
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT
from src.data.make_dataloaders import build_dataloaders


def _save_bar_plot(
    values: dict,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    display_labels: Optional[dict] = None,
    figure_size=(6, 4),
    rotation: int = 0,
) -> None:
    """
    Save a simple bar chart for the provided mapping.
    """
    if not values:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(values.keys())
    numeric_values = list(values.values())
    tick_labels = [display_labels.get(label, label) for label in labels] if display_labels else labels

    fig, ax = plt.subplots(figsize=figure_size)
    bars = ax.bar(labels, numeric_values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=rotation, ha="right" if rotation else "center")

    for label, bar, value in zip(labels, bars, numeric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            (
                f"{value:.6f}"
                if label == "second_train_batch_seconds" and isinstance(value, float)
                else f"{value:.3f}" if isinstance(value, float) else str(value)
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_direct_run_plots(stats: dict, timings: dict) -> None:
    """
    Save lightweight summary plots for direct execution only.
    """
    output_dir = PROJECT_ROOT / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_bar_plot(
        {
            "train": stats["train_files_kept"],
            "val": stats["val_files_kept"],
            "test": stats["test_files_kept"],
        },
        output_dir / "dataloader_file_counts.png",
        "Kept Files Per Split",
        "Files",
    )

    _save_bar_plot(
        {
            "train": stats["train_samples"],
            "val": stats["val_samples"],
            "test": stats["test_samples"],
        },
        output_dir / "dataloader_sample_counts.png",
        "Samples Per Split",
        "Samples",
    )

    available_timings = {
        key: value
        for key, value in timings.items()
        if value is not None
    }
    if available_timings:
        _save_bar_plot(
            available_timings,
            output_dir / "dataloader_timing_seconds.png",
            "DataLoader Build And Batch Timing",
            "Seconds",
            display_labels={
                "build_dataloaders_seconds": "build_dataloaders",
                "first_train_batch_seconds": "first_train_batch",
                "second_train_batch_seconds": "second_train_batch",
            },
            figure_size=(7.5, 4),
            rotation=0,
        )


def main() -> None:
    t0 = time.perf_counter()

    train_loader, val_loader, test_loader, stats = build_dataloaders()

    t1 = time.perf_counter()

    train_iter = iter(train_loader)
    x_batch, y_batch = next(train_iter)

    t2 = time.perf_counter()

    try:
        _ = next(train_iter)
        t3 = time.perf_counter()
    except StopIteration:
        t3 = t2

    print("Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nTiming:")
    print(f"build_dataloaders_seconds: {t1 - t0:.3f}")
    print(f"first_train_batch_seconds: {t2 - t1:.3f}")
    print(f"second_train_batch_seconds: {t3 - t2:.10f}")
    print(f"total_seconds: {t3 - t0:.3f}")

    _save_direct_run_plots(
        stats,
        {
            "build_dataloaders_seconds": t1 - t0,
            "first_train_batch_seconds": t2 - t1,
            "second_train_batch_seconds": t3 - t2,
        },
    )

    print("\nNumber of batches:")
    print("train:", len(train_loader))
    print("val:", len(val_loader))
    print("test:", len(test_loader))

    print("\nBatch X shape:", x_batch.shape)
    print("Batch y shape:", y_batch.shape)
    print("Batch labels:", y_batch.tolist())


if __name__ == "__main__":
    main()
