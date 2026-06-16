from __future__ import annotations

import argparse
import json

from src.config import DEFAULT_RUN_NAME
from src.train_transformer_baseline import train


ABLATIONS = {
    "A0": {
        "description": "final candidate: per_window, 1024, batch 64, filters/aug off",
        "normalization_scope": "per_window",
        "window_size": 1024,
        "stride": 1024,
        "batch_size": 64,
        "num_epochs": 30,
    },
    "A1": {
        "description": "global train-fitted normalization ablation",
        "normalization_scope": "global_train",
        "window_size": 1024,
        "stride": 1024,
        "batch_size": 64,
        "num_epochs": 30,
    },
    "A2": {
        "description": "512-sample window FPGA-cost ablation",
        "normalization_scope": "per_window",
        "window_size": 512,
        "stride": 512,
        "batch_size": 64,
        "num_epochs": 30,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepared A0/A1/A2 ablation runner. Defaults to smoke mode."
    )
    parser.add_argument("--experiments", nargs="+", default=["A0"])
    parser.add_argument("--max-files", type=int, default=24)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = [exp for exp in args.experiments if exp not in ABLATIONS]
    if unknown:
        raise ValueError(f"Unknown ablation ids: {unknown}. Valid: {sorted(ABLATIONS)}")

    if args.full and not args.confirm_full:
        raise RuntimeError(
            "Full ablation runs are expensive. Re-run with --full --confirm-full "
            "only when intentionally launching the full experiment."
        )

    max_files = None if args.full else args.max_files
    summaries = []
    for exp_id in args.experiments:
        spec = ABLATIONS[exp_id]
        run_name = f"{DEFAULT_RUN_NAME}_{exp_id.lower()}_{spec['normalization_scope']}"
        print(f"[ABLATION] Starting {exp_id}: {json.dumps(spec, indent=2)}")
        summary = train(
            data_dir=args.data_dir,
            max_files=max_files,
            run_name=run_name,
            num_epochs=spec["num_epochs"],
            batch_size=spec["batch_size"],
            window_size=spec["window_size"],
            stride=spec["stride"],
            normalization_scope=spec["normalization_scope"],
            normalization_method="zscore",
        )
        summary["ablation_id"] = exp_id
        summary["ablation_spec"] = spec
        summary["mode"] = "full" if args.full else "smoke"
        summaries.append(summary)

    print("[ABLATION] Completed summaries:")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
