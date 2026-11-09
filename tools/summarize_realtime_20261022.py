"""Assemble artifacts/results/fpga_realtime_bench_20261022/realtime_verdict.json
from the measurement files in that directory (no new measurement is made here).

Requirement: R_req = f_s / stride (one classification per window, every window
of a continuous stream). From src/config.py: f_s = 50,000 Hz, window = stride =
1024 samples (no overlap) -> R_req = 48.828 windows/s, period 20.48 ms.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import MAFAULDA_SAMPLE_RATE, WINDOW_SIZE, WINDOW_STRIDE

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "artifacts" / "results" / "fpga_realtime_bench_20261022"


def _board(tag):
    s = json.loads((D / tag / "summary.json").read_text())
    t = np.load(D / tag / "timing.npz")
    it = t["t_iter"]
    period = WINDOW_STRIDE / MAFAULDA_SAMPLE_RATE
    return {
        "windows": s["executed"], "aborted": s["aborted"], "wall_s": s["wall_s"],
        "throughput_inf_s": s["throughput_inf_s"],
        "stage_ms_mean": {"input_read_parse": round(float(t["t_read"].mean()) * 1e3, 3),
                          "blocked_on_dma_core": round(float(t["t_core"].mean()) * 1e3, 3),
                          "loop_iteration": round(float(it.mean()) * 1e3, 3)},
        "iterations_over_window_period": int((it > period).sum()),
        "iterations_over_100ms": int((it > 0.1).sum()),
        "max_iteration_ms": round(float(it.max()) * 1e3, 1),
        "decile_throughput_inf_s": [round(len(b) / float(it[b].sum()), 1)
                                    for b in np.array_split(np.arange(len(it)), 10)],
        "temp_c": s["temp_c"],
    }


def main() -> None:
    r_req = MAFAULDA_SAMPLE_RATE / WINDOW_STRIDE
    e2e = {p.name: json.loads((p / "summary.json").read_text())
           for p in sorted((D / "e2e").iterdir()) if (p / "summary.json").exists()}
    runner_of = {"before_head_runner": "HEAD e42ec4f (serial)",
                 "before_head_runner_repeat": "HEAD e42ec4f (serial)",
                 "after_pipelined_runner": "20261022a (overlap in all modes)",
                 "after_pipelined_class_balanced120": "20261022a (overlap in all modes)",
                 "final_runner_sequential3000": "20261022b = repository version",
                 "final_runner_sequential10000": "20261022b = repository version",
                 "final_runner_class_balanced120": "20261022b = repository version"}
    base, pipe = _board("board_full_baseline"), _board("board_full_pipelined")
    out = {
        "requirement": {
            "formula": "R_req = sample_rate / stride",
            "sample_rate_hz": MAFAULDA_SAMPLE_RATE, "window_size": WINDOW_SIZE,
            "stride": WINDOW_STRIDE, "overlap": WINDOW_SIZE - WINDOW_STRIDE,
            "required_windows_per_s": round(r_req, 3),
            "window_period_ms": round(1e3 / r_req, 3),
            "source": "src/config.py",
        },
        "measured_board_only_full_test": {
            "what": "on-board loop: gzip read of prepared INT10 codes -> one 32,768-byte MM2S "
                    "frame -> core -> 4 logits -> argmax on PS -> prediction file; no SSH per window",
            "baseline_serial_loop": base, "pipelined_loop": pipe,
            "predictions_identical_to_2026-10-08_run": True,
            "logits_identical_between_modes": True,
        },
        "measured_end_to_end": {
            "what": "board runner -> SSH/Ethernet -> PC production parser (parse_line -> FRAME line); "
                    "Tk GUI rendering excluded. Sequential runs use contiguous windows from 0; "
                    "class_balanced runs use the validated 120-window demo plan (gzip skipping "
                    "dominates their wall time, so their throughput is not an inference rate)",
            "runs": {t: dict({k: e2e[t].get(k) for k in
                              ("sample_mode", "received_results", "pc_receive_throughput_inf_s",
                               "pc_inter_arrival_ms", "board_lat_ms", "pc_parse_us",
                               "match_recorded_fpga_pred", "accuracy_vs_labels", "conf_present")},
                             runner=runner_of.get(t, "?")) for t in e2e},
        },
        "estimates_not_measurements": {
            "hls_csynth_interval_ms_at_100mhz": 14.90527,
            "source": "artifacts/results/seq256_dma_system_dsp_margin_dma16_z7020_20261008/summary.json",
        },
        "not_covered": [
            "sensor acquisition and on-board normalization/quantization (inputs are INT10 codes "
            "prepared offline by tools/export_fpga_testset.py)",
            "Tk GUI rendering cost",
            "hard per-window deadline: individual iterations exceed 20.48 ms (see counts); "
            "a live stream would need an input buffer to absorb them",
        ],
        "verdict": {
            "sustained_average": "board-only pipelined %.2f inf/s over 95,404 windows; end-to-end "
                                 "%.2f and %.2f inf/s over 10,000 windows (two runs); all exceed "
                                 "%.3f windows/s (prepared-window replay, not a live sensor)."
                                 % (pipe["throughput_inf_s"],
                                    e2e["after_pipelined_runner"]["pc_receive_throughput_inf_s"],
                                    e2e["final_runner_sequential10000"]["pc_receive_throughput_inf_s"], r_req),
            "before_change": "board-only serial %.2f inf/s (95,404) and end-to-end %.2f / %.2f inf/s "
                             "(10,000, two runs) also exceeded the requirement today; the 2026-10-08 "
                             "46.4 inf/s full-test average (slowdown after ~41k windows) did not reproduce."
                             % (base["throughput_inf_s"],
                                e2e["before_head_runner"]["pc_receive_throughput_inf_s"],
                                e2e["before_head_runner_repeat"]["pc_receive_throughput_inf_s"]),
            "short_run_variance": "a 3,000-window run of the final runner gave %.2f inf/s (stalls weigh "
                                  "more in short runs)" % e2e["final_runner_sequential3000"]["pc_receive_throughput_inf_s"],
            "hard_deadline": "not demonstrated (%d of %d pipelined iterations > period)"
                             % (pipe["iterations_over_window_period"], pipe["windows"]),
        },
    }
    (D / "realtime_verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["verdict"], indent=2))


if __name__ == "__main__":
    main()
