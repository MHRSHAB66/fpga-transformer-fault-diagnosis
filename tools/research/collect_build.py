"""Collect a Vivado/Vitis research build into a tracked build package (the layout that
tools/package_seq256_pynq_deploy.py --pkg expects):

  <out>/bitstream/{design_1_wrapper.bit, design_1.hwh, seq256_dma_system.xsa}
  <out>/hls_reports/transformer_top_csynth.rpt (+ transformer_core_csynth.rpt)
  <out>/vivado_reports/post_route_*.rpt, post_synth_*.rpt
  <out>/build_summary.json  LUT/FF/BRAM/DSP, WNS/TNS, failing endpoints, HLS latency

    python -m tools.research.collect_build --proj vb04 --hls seq256_b04 --out artifacts/research/B04_20261114/b04_build
Refuses to overwrite an existing package (failed implementations are kept, never replaced).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def util(rpt):
    t = rpt.read_text()
    out = {}
    for key, name in (("Slice LUTs", "LUT"), ("Slice Registers", "FF"), ("Block RAM Tile", "BRAM"), ("DSPs", "DSP")):
        m = re.search(r"^\| %s\s*\|\s*([\d.]+)" % re.escape(key), t, re.M)
        out[name] = float(m.group(1)) if m else None
    return out


def timing(rpt):
    t = rpt.read_text()
    m = re.search(r"WNS\(ns\).*?\n\s*-+.*?\n\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", t, re.S)
    met = "All user specified timing constraints are met." in t
    return {"WNS_ns": float(m.group(1)), "TNS_ns": float(m.group(2)), "failing_endpoints": int(m.group(3)),
            "constraints_met": met, "note": "WNS 0.000 means met with no reported margin" if m and float(m.group(1)) == 0 else ""}


def hls_latency(rpt):
    m = re.search(r"Latency \(cycles\).*?\n.*?\n.*?\n\s*\|\s*(\d+)\|\s*(\d+)\|", rpt.read_text(), re.S)
    return {"min": int(m.group(1)), "max": int(m.group(2))} if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--proj", required=True, help="Vivado project dir name under tmp/ (e.g. vb04)")
    ap.add_argument("--hls", required=True, help="Vitis HLS project dir at the repo root (e.g. seq256_b04)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    proj, hls, out = REPO / "tmp" / args.proj, REPO / args.hls, args.out
    if out.exists():
        raise SystemExit("%s exists: refusing to overwrite a build package" % out)
    hwh = next(proj.rglob("hw_handoff/design_1.hwh"))
    copies = {proj / ("%s.runs/impl_1/design_1_wrapper.bit" % args.proj): "bitstream/design_1_wrapper.bit",
              hwh: "bitstream/design_1.hwh", proj / "seq256_dma_system.xsa": "bitstream/seq256_dma_system.xsa",
              hls / "sol1/syn/report/transformer_top_csynth.rpt": "hls_reports/transformer_top_csynth.rpt",
              hls / "sol1/syn/report/transformer_core_csynth.rpt": "hls_reports/transformer_core_csynth.rpt"}
    for r in sorted(proj.glob("post_*.rpt")):
        copies[r] = "vivado_reports/" + r.name
    for src, dst in copies.items():
        if src.exists():
            (out / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / dst)
    summ = {"vivado_project": "tmp/" + args.proj, "hls_project": args.hls,
            "post_route": util(out / "vivado_reports/post_route_utilization.rpt"),
            "timing": timing(out / "vivado_reports/post_route_timing_summary.rpt"),
            "hls_latency_cycles": hls_latency(out / "hls_reports/transformer_top_csynth.rpt")}
    (out / "build_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
