"""One table: stock kernel (R0/R1/R2/L1) vs one-shot PREEMPT_RT kernel (K1/K2/K3).

Reads only the independent verifier outputs and the board summary.json files.
Usage: python -m tools.compare_hard_rt_kernels_20261024
Writes artifacts/results/hard_rt_preempt_rt_20261024/comparison.{json,md}
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOCK = ROOT / "artifacts/results/hard_rt_20261024"
RT = ROOT / "artifacts/results/hard_rt_preempt_rt_20261024"


def rows(base: Path, vfile: str, kernel: str):
    v = json.loads((base / vfile).read_text(encoding="utf-8"))["runs"]
    for tag, r in sorted(v.items()):
        s = json.loads((base / tag / "remote/out/summary.json").read_text())
        yield {"run": tag, "kernel": kernel, "sched": s["sched"],
               "accepted": r["acceptance_windows"], "misses": r["deadline_misses"],
               **{k: r["response_ms"][k] for k in ("p50", "p99", "p99.9", "p99.99", "max")},
               "start_lag_max_ms": r["start_lag_ms_max"], "rate": r["throughput_windows_per_s"],
               "classes_ok": "%d/%d" % (r["class_eq_recorded"], r["executed"]),
               "logits_ok": "%d/%d" % (r["logits_eq_reference_all4"], r["executed"]),
               "invol_ctx_sw": s.get("involuntary_ctx_switches"),
               "temp_start": round(s["temp_c"]["start"], 1), "temp_max": round(s["temp_c"]["max"], 1),
               "guard_rc": r["guard_rc"], "workload_rc": r["workload_rc"], "formal_pass": r["hard_rt_pass"]}


def main():
    t = list(rows(STOCK, "verification_v2.json", "4.14.0-xilinx SMP PREEMPT")) + \
        list(rows(RT, "verification.json", "4.14.0-rt1-rt-xilinx+ PREEMPT_RT"))
    (RT / "comparison.json").write_text(json.dumps(t, indent=2), encoding="utf-8")
    cols = ["run", "kernel", "accepted", "misses", "p50", "p99", "p99.9", "p99.99", "max",
            "start_lag_max_ms", "rate", "classes_ok", "logits_ok", "invol_ctx_sw", "temp_start",
            "temp_max", "guard_rc", "workload_rc", "formal_pass"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in t]
    (RT / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
