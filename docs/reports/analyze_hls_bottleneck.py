"""
Analytical latency attribution for the F_1layer @ int10 HLS core.

WHY ANALYTICAL: Vitis HLS is not installed on this development machine, so the
per-function HLS latency breakdown (transformer_core_csynth.rpt sub-instances)
cannot be regenerated here. This script builds a transparent cycle model FROM
THE ACTUAL LOOP STRUCTURE AND PRAGMAS in hls/f1_int10/vitis/transformer_top.cpp
and reconciles it against the ONE measured number that IS preserved: the
top-level latency of 74,246,781 cycles from the frozen csynth report.

Every count here is analytical unless explicitly tagged "measured". The model's
purpose is to attribute WHERE the cycles go and to bound the expected effect of
the bit-exact context optimization (transformer_top_opt.cpp), NOT to replace an
HLS measurement. The optimized figure is a projection; it must be MEASURED with
run_top_opt_verify_and_synth.tcl on the laptop before any hardware claim.

Run:
    python docs/reports/analyze_hls_bottleneck.py
Writes:
    artifacts/results/hls_attention_bottleneck_analysis.json
"""
from __future__ import annotations

import json
from pathlib import Path

SEQ, CH, D, NHEAD, DFF, CLS_HID, NUM_CLASSES = 1024, 8, 32, 4, 64, 16, 4
HD = D // NHEAD  # 8

MEASURED_TOTAL_CYCLES = 74_246_781  # csynth report, xc7z020clg400-1, 10 ns target
TARGET_CLOCK_NS = 10.0

# --- cycle model -----------------------------------------------------------
# Rule applied per loop: a loop carrying "#pragma HLS PIPELINE" costs ~trip
# cycles at II=1; a sub-loop nested inside a pipelined loop is fully unrolled
# (parallel), so it does not add cycles but widens the datapath. Pipeline
# fill/drain (a few cycles per loop) and handshake are collected in a single
# reconciled "residual" term rather than guessed per loop.
LN = 95   # layernorm(1 row): D-sum + fixed rsqrt loop + D-scale (approx)


def linear_cycles(rows, dout):
    # linear(): o-loop (rows*dout) is pipelined II=1; inner Din loop unrolled.
    return rows * dout


def per_token_cycles():
    c = {}
    c["q_proj"] = linear_cycles(1, D)                    # 32
    # scores: j-loop (SEQ) pipelined, HD dot unrolled -> SEQ cyc/head
    c["attn_scores"] = NHEAD * SEQ                       # 4096  (8 MAC/cyc)
    c["attn_rowmax"] = NHEAD * SEQ                       # 4096
    c["attn_exp_denom"] = NHEAD * SEQ                    # 4096
    c["attn_softmax_div"] = NHEAD * SEQ                  # 4096  (II=1 ideal; see residual)
    # context BASELINE: d-outer (HD) x j-inner (SEQ) pipelined -> HD*SEQ cyc/head
    c["attn_context_baseline"] = NHEAD * HD * SEQ        # 32768 (1 MAC/cyc)
    c["out_proj"] = linear_cycles(1, D)                  # 32
    c["finish_token"] = (
        D + LN + linear_cycles(1, DFF) + DFF + linear_cycles(1, D) + D + LN
    )                                                    # ~414
    return c


def one_time_cycles():
    c = {}
    c["input_proj"] = linear_cycles(SEQ, D)              # 32768
    c["pe_add"] = SEQ * D                                # 32768
    c["kv_proj"] = 2 * linear_cycles(SEQ, D)             # 65536
    c["pooling"] = D * SEQ                               # 32768
    c["classifier_tail"] = LN + linear_cycles(1, CLS_HID) + CLS_HID + linear_cycles(1, NUM_CLASSES)
    return c


def build():
    pt = per_token_cycles()
    ot = one_time_cycles()

    context_baseline = pt["attn_context_baseline"] * SEQ
    # optimized context: j-outer pipelined, HD lanes unrolled -> SEQ cyc/head
    context_opt_per_token = NHEAD * SEQ
    context_opt = context_opt_per_token * SEQ

    per_token_total = sum(pt.values())
    ideal_token_loop = per_token_total * SEQ
    ideal_one_time = sum(ot.values())
    ideal_total = ideal_token_loop + ideal_one_time

    residual = MEASURED_TOTAL_CYCLES - ideal_total  # div-II + fill/drain + handshake

    # projected optimized ideal: swap baseline context for optimized context
    ideal_total_opt = ideal_total - context_baseline + context_opt
    # two projection bounds for the MEASURED optimized latency:
    #  - conservative: residual (division/overhead) unchanged
    #  - optimistic:   residual scales with ideal work (same overhead fraction)
    proj_measured_conservative = ideal_total_opt + residual
    proj_measured_optimistic = ideal_total_opt * (MEASURED_TOTAL_CYCLES / ideal_total)

    def wps(cycles):
        return 1.0 / (cycles * TARGET_CLOCK_NS * 1e-9)

    # component cycle shares (of measured total), baseline
    token_scaled = {k: v * SEQ for k, v in pt.items()}
    all_components = {**token_scaled, **ot}
    shares = {
        k: {
            "cycles": v,
            "percent_of_measured_total": round(100.0 * v / MEASURED_TOTAL_CYCLES, 4),
        }
        for k, v in sorted(all_components.items(), key=lambda kv: -kv[1])
    }

    out = {
        "scope": "F_1layer int10 HLS core, transformer_top.cpp, SEQ=1024, xc7z020clg400-1, 10 ns",
        "method": "analytical cycle model from loop structure + pragmas, reconciled to the one measured top-level latency",
        "measured": {
            "source": "artifacts/results/hls_z7020_q40_bounded_snapshot/transformer_top_csynth.rpt",
            "total_latency_cycles": MEASURED_TOTAL_CYCLES,
            "note": "This is the only measured latency available on this machine; per-function HLS latencies were not regenerated (Vitis HLS not installed here).",
        },
        "analytical_cycle_model": {
            "per_token_cycles": pt,
            "per_token_total": per_token_total,
            "token_loop_trip": SEQ,
            "one_time_cycles": ot,
            "ideal_token_loop_cycles": ideal_token_loop,
            "ideal_one_time_cycles": ideal_one_time,
            "ideal_total_cycles_II1": ideal_total,
            "reconciliation_residual_cycles": residual,
            "reconciliation_residual_percent_of_measured": round(100.0 * residual / MEASURED_TOTAL_CYCLES, 4),
            "residual_attribution": [
                "runtime integer division in softmax normalization (qattn = numerator/denom, variable denominator; a 32-bit divider does not pipeline at II=1 on this device)",
                "pipeline fill/drain across the many short per-token inner loops (context restarts the j-pipeline HD times per head in the baseline)",
                "per-token module handshake / control overhead (finish_token is INLINE off)",
            ],
        },
        "component_cycle_shares_baseline": shares,
        "headline_finding": {
            "text": "In MAC COUNT scores and context are symmetric (33,554,432 each, 88.58% combined). In CYCLES they are NOT: the score loop unrolls the HD=8 dot product (8 MAC/cyc) while the baseline context loop does not (1 MAC/cyc), so context costs 8x more cycles than scores.",
            "attn_context_baseline_cycles": context_baseline,
            "attn_context_baseline_percent_of_measured": round(100.0 * context_baseline / MEASURED_TOTAL_CYCLES, 4),
            "attn_scores_cycles": pt["attn_scores"] * SEQ,
            "attn_scores_percent_of_measured": round(100.0 * (pt["attn_scores"] * SEQ) / MEASURED_TOTAL_CYCLES, 4),
            "context_to_scores_cycle_ratio": context_baseline // (pt["attn_scores"] * SEQ),
        },
        "optimization_projection": {
            "change": "context accumulation restructured to j-outer / HD-unrolled (bit-exact), v cyclic-partitioned factor 8",
            "file": "hls/f1_int10/vitis/transformer_top_opt.cpp",
            "functional_status": "bit-exact vs baseline: verified by g++ C-simulation on 16/16 golden vectors (byte-identical logits); ap_int csim + csynth still PENDING on laptop",
            "context_opt_cycles": context_opt,
            "context_cycle_reduction": context_baseline - context_opt,
            "ideal_total_cycles_opt": ideal_total_opt,
            "ideal_speedup": round(ideal_total / ideal_total_opt, 3),
            "projected_measured_latency_cycles_conservative": int(proj_measured_conservative),
            "projected_measured_latency_cycles_optimistic": int(proj_measured_optimistic),
            "projected_speedup_range": [
                round(MEASURED_TOTAL_CYCLES / proj_measured_conservative, 3),
                round(MEASURED_TOTAL_CYCLES / proj_measured_optimistic, 3),
            ],
            "projected_throughput_wps_range": [
                round(wps(proj_measured_conservative), 3),
                round(wps(proj_measured_optimistic), 3),
            ],
            "required_realtime_wps": round(1.0 / (SEQ / 50000.0), 6),
            "note": "PROJECTION ONLY. Speedup and throughput must be MEASURED by csynth (run_top_opt_verify_and_synth.tcl). Even the optimistic projection stays well below the 48.83 wps continuous non-overlapping input rate; this optimization is necessary but not sufficient for real-time.",
        },
        "corrected_summary_json_note": {
            "field": "quadratic_attention_share_percent in artifacts/results/hls_z7020_current_summary.json",
            "stored_value": 88.58329943,
            "recomputed_value": round(100.0 * 67108864 / 75760192, 8),
            "impact": "immaterial (both round to 88.58%); flagged for correction",
        },
    }
    return out


def main():
    root = Path(__file__).resolve().parents[2]
    out = build()
    dst = root / "artifacts" / "results" / "hls_attention_bottleneck_analysis.json"
    dst.write_text(json.dumps(out, indent=2))
    m = out["analytical_cycle_model"]
    print(f"measured total cycles     : {MEASURED_TOTAL_CYCLES:,}")
    print(f"ideal (II=1) total cycles : {m['ideal_total_cycles_II1']:,}")
    print(f"reconciliation residual   : {m['reconciliation_residual_cycles']:,} "
          f"({m['reconciliation_residual_percent_of_measured']}% of measured)")
    hf = out["headline_finding"]
    print(f"context baseline cycles   : {hf['attn_context_baseline_cycles']:,} "
          f"({hf['attn_context_baseline_percent_of_measured']}% of measured)")
    print(f"scores cycles             : {hf['attn_scores_cycles']:,} "
          f"({hf['attn_scores_percent_of_measured']}% of measured)")
    print(f"context/scores cyc ratio  : {hf['context_to_scores_cycle_ratio']}x")
    op = out["optimization_projection"]
    print(f"ideal speedup (context opt): {op['ideal_speedup']}x")
    print(f"projected speedup range   : {op['projected_speedup_range']}x")
    print(f"projected throughput wps  : {op['projected_throughput_wps_range']}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
