"""
Stage 12 -- assemble the final project report from committed artifacts.

Collects the software accuracy (float vs int10), the quantization bit-width
sweep, and the hardware export config into a single Markdown report plus a
machine-readable JSON. Hardware resource/latency numbers are left as explicit
`TBD` placeholders to be filled from the Vitis reports after synthesis (this
desktop has no FPGA toolchain).

Reads (all committed):
  - kaggle/int10_shard/int10_fulltest_combined.json   (int10 full-test accuracy)
  - kaggle/sweep_multibit/integer_multibit_sweep_F.json (bit-width sweep)
  - hls/f1_int10/model_manifest.json                  (HW geometry / scales)

Run
---
    python -m src.reporting.generate_final_report
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.config import PROJECT_ROOT

INT10_FULLTEST = PROJECT_ROOT / "kaggle" / "int10_shard" / "int10_fulltest_combined.json"
BIT_SWEEP      = PROJECT_ROOT / "kaggle" / "sweep_multibit" / "integer_multibit_sweep_F.json"
HW_MANIFEST    = PROJECT_ROOT / "hls" / "f1_int10" / "model_manifest.json"

OUT_MD   = PROJECT_ROOT / "docs" / "final_report.md"
OUT_JSON = PROJECT_ROOT / "artifacts" / "results" / "final_report.json"

# Float full-test accuracy of the DEPLOYED F_1layer checkpoint. Source: day15
# Kaggle tuning kernel v10 test evaluation on all 95,404 test windows (the int10
# number below is measured on the same split). Not in a committed JSON, so it is
# recorded here as a documented constant.
FLOAT_FULLTEST_ACC = 0.9755


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def build():
    int10 = _load(INT10_FULLTEST)
    sweep = _load(BIT_SWEEP)
    manifest = _load(HW_MANIFEST)

    report = {
        "project": "Real-Time Machinery Condition Monitoring with a Transformer on FPGA",
        "generated": str(date.today()),
        "deployed_model": "F_1layer (1 encoder layer, d_model=32, 4 heads, ffn=64, 9,492 params)",
        "dataset": "MAFAULDA, 8 channels, window=1024, 4 classes "
                   "(normal/imbalance/misalignment/bearing_fault)",
        "software_accuracy": {
            "float_fulltest": FLOAT_FULLTEST_ACC,
            "int10_fulltest": int10["int_accuracy"] if int10 else None,
            "int10_per_class": int10["per_class"] if int10 else None,
            "test_windows": int10["full_test_windows"] if int10 else None,
        },
        "bit_width_sweep": sweep["results"] if sweep else None,
        "hardware_config": manifest,
        "hardware_resources": {  # fill from Vitis reports after synthesis
            "part": "TBD -- final FPGA board/device not selected",
            "LUT": "TBD", "FF": "TBD", "BRAM": "TBD", "DSP": "TBD",
            "latency_cycles": "TBD", "II": "TBD", "Fmax_MHz": "TBD",
        },
        "pipeline_status": {
            "software_training": "done",
            "quantization_int10": "done (full-test 97.11%)",
            "hls_export_and_verify": "done (verify_export PASS 16/16 bit-exact)",
            "hls_synthesis": "TBD (see hls/f1_int10/vitis/SYNTHESIS_GUIDE.md)",
            "pc_uart_receiver_and_ui": "done (host/, tested end-to-end)",
            "pynq_driver": "done (numpy backend verified; hardware backend template)",
            "on_board_uart_demo": "TBD (needs synthesized bitstream + board)",
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report)
    print(f"[report] -> {OUT_MD}")
    print(f"[report] -> {OUT_JSON}")


def _pct(x):
    return "N/A" if x is None else f"{x*100:.2f}%"


def _write_markdown(r):
    L = []
    L.append("# Final Project Report\n")
    L.append(f"*{r['project']}*  \nGenerated {r['generated']} -- auto-assembled by "
             "`src/reporting/generate_final_report.py` from committed artifacts.\n")
    L.append("## 1. Model & data\n")
    L.append(f"- **Deployed model:** {r['deployed_model']}")
    L.append(f"- **Dataset:** {r['dataset']}")
    L.append("- **FPGA numeric format:** int10 (activations, weights, attention all 10-bit)\n")

    sa = r["software_accuracy"]
    L.append("## 2. Software accuracy (full test set)\n")
    L.append(f"Measured on {sa['test_windows']:,} unseen windows.\n" if sa["test_windows"] else "")
    L.append("| precision | accuracy |")
    L.append("|-----------|----------|")
    L.append(f"| float (reference) | {_pct(sa['float_fulltest'])} |")
    L.append(f"| **int10 (FPGA)** | **{_pct(sa['int10_fulltest'])}** |")
    if sa["int10_per_class"]:
        L.append("\n**int10 per-class accuracy**\n")
        L.append("| class | n | accuracy |")
        L.append("|-------|---|----------|")
        for name, d in sa["int10_per_class"].items():
            L.append(f"| {name} | {d['n']:,} | {_pct(d['accuracy'])} |")
    L.append("")

    if r["bit_width_sweep"]:
        L.append("## 3. Quantization bit-width sweep (F_1layer, 4000-window subset)\n")
        L.append("| bits | int accuracy | drop vs float | agreement |")
        L.append("|------|--------------|---------------|-----------|")
        for res in r["bit_width_sweep"]:
            L.append(f"| {res['bits']} | {_pct(res['int_accuracy'])} | "
                     f"{res['accuracy_drop_vs_float']*100:+.2f}% | "
                     f"{_pct(res['prediction_agreement_vs_float'])} |")
        L.append("\nint10 is the smallest near-lossless width; int8 collapses the "
                 "imbalance class. Hence `ap_int<10>` on hardware.\n")

    m = r["hardware_config"] or {}
    L.append("## 4. Hardware export config\n")
    if m:
        L.append(f"- geometry: seq={m.get('seq_len')}, in_ch={m.get('in_channels')}, "
                 f"d_model={m.get('d_model')}, heads={m.get('nhead')}, "
                 f"layers={m.get('num_layers')}, ffn={m.get('dim_feedforward')}, "
                 f"classes={m.get('num_classes')}")
        L.append(f"- bit-widths: act={m.get('act_bits')}, weight={m.get('weight_bits')}, "
                 f"attn={m.get('attn_bits')}; input_scale={m.get('input_scale')}")
        g = m.get("golden", {})
        L.append(f"- golden self-check: {g.get('correct')}/{g.get('count')} classified; "
                 "bit-exact verify PASS 16/16 (`hls/f1_int10/verify_export.py`)\n")

    hw = r["hardware_resources"]
    L.append("## 5. Hardware resources & latency (fill after Vitis synthesis)\n")
    L.append(f"- target part: {hw['part']}")
    L.append("")
    L.append("| metric | value |")
    L.append("|--------|-------|")
    for k in ["LUT", "FF", "BRAM", "DSP", "latency_cycles", "II", "Fmax_MHz"]:
        L.append(f"| {k} | {hw[k]} |")
    L.append("\nSee `hls/f1_int10/vitis/SYNTHESIS_GUIDE.md` for the synthesis flow.\n")

    L.append("## 6. End-to-end pipeline status\n")
    L.append("| stage | status |")
    L.append("|-------|--------|")
    for k, v in r["pipeline_status"].items():
        L.append(f"| {k.replace('_', ' ')} | {v} |")
    L.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    build()
