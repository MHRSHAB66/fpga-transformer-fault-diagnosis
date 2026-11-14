"""Truthful per-variant deployment contracts for the research raw-input bundles.

One JSON per bundle (artifacts/research/contracts/<bundle>.json) records the real
numeric contract (precision, signedness, qmax of every stage that differs), the
raw24 input representation and on-FPGA preprocessing, the reference that the
bundle must match (and how to reconstruct it), the golden set actually used
(count, ties, sha256), the source/build identity and the physical validation
evidence. tools/package_seq256_pynq_deploy.py --contract renders the bundle's
README, config/model_manifest.json, config/integer_config_scales.json and
config/variant_contract.json from it, so regenerating a bundle cannot restore
the generic SEQ=256 ingress text (audit 2026-11-13, finding 3).

    python -m tools.research.variant_contracts            # write all contracts
    python tools/package_seq256_pynq_deploy.py --pkg <build> --bundle artifacts/deploy/<bundle> \
        --golden <golden dir> --contract artifacts/research/contracts/<bundle>.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts/research/contracts"
CANON_GOLDEN = "hls/f1_int10_seq256/golden_raw1024_argmax"
CANON_CMODEL = "artifacts/results/onchip_preprocess_20261109/csim_raw_argmax/csim_logits.txt"
RES = "artifacts/results/raw_core_board_%s"

INT10 = {"bits": 10, "signed": True, "qmax": 511}
CANONICAL_NUMERIC = {
    "weights": INT10, "activations": INT10,
    "attention_probabilities": {"bits": 10, "signed": True, "qmax": 511, "scale": "1/511",
                                "note": "softmax probabilities are non-negative; signed 10-bit code"},
    "softmax_quotient": "integer division (num*qmax + den/2) / den",
    "scales": "artifacts/quantized/f_seq256_int10/integer_config.json (export scales)",
}
INPUT = {
    "representation": "raw 24-bit ADC codes, one per little-endian int32 word; 8192 words per window, "
                      "row-major [sample, channel] (word = sample*8 + channel), TLAST on word 8191",
    "contract": "artifacts/preprocessing/raw_input_contract_v1.json",
    "on_fpga_preprocessing": "per-window per-channel integer z-score -> INT10 codes (+-511) -> 4:1 "
                             "non-overlapping mean aggregation (1024 -> 256 tokens) inside transformer_top",
    "host_side_normalisation": "none (the host sends raw codes)",
    "channels": "8 MAFAULDA channels in src/config.py order",
}
CLASS_OUT = {"stream": "4 int32 logits (S2MM, TLAST on logit 3)",
             "pred_class": "AXI-Lite register: FPGA argmax, lowest index on ties; class map 0 normal, "
                           "1 imbalance, 2 misalignment, 3 bearing_fault"}
INTEGRITY_B02 = ("integrity_status (AXI-Lite, metadata, never a class): bit0 input-buffer XOR signature "
                 "(after ingress vs after projection), bit1 classifier tail evaluated twice and compared "
                 "(time-multiplexed on ONE compute unit: detects transient upsets of one evaluation, not "
                 "persistent/common-mode faults), bit2 projection accumulator outside its proven bound "
                 "|acc+b| <= 719,316")
INTEGRITY_B04 = INTEGRITY_B02 + ("; bit3 per-row projection checksum (ABFT): sum_o acc[r][o] == "
                                 "sum_i x[r][i]*WSUM[i], exact integers")
INJECTION = ("inj_site/inj_word/inj_bit (AXI-Lite, research only): 0 off, 1 input word bit (10-bit code), "
             "2 projection accumulator bit (32-bit), 3 logit of the first tail evaluation; written while the "
             "core is idle, applied to the next ap_start only (transient, one window)")


def canonical_ref(note="bit-identical to the canonical raw-input core (exact implementation change)"):
    return {"bit_identical_to_canonical": True, "numpy_backend": "canonical", "golden": CANON_GOLDEN,
            "cmodel_logits": CANON_CMODEL, "note": note}


VARIANTS = {
    "research_divq9_z7020_20261113": {
        "id": "A05", "title": "Exact 9-bit-quotient softmax divider (div_q9)",
        "numeric": dict(CANONICAL_NUMERIC, softmax_quotient="exact 9-step restoring divider div_q9 "
                        "(quotient <= 511 proven; 367,657,004 reachable pairs checked)"),
        "manifest_overrides": {}, "reference": canonical_ref(),
        "outputs": dict(CLASS_OUT),
        "source": {"hls_top": "hls/research/a04_softmax_div/transformer_top_seq256_raw1024_divq9_argmax.cpp",
                   "generator": "hls/research/a04_softmax_div/make_divq9_top.py",
                   "build": "powershell -File tools/build_raw_core.ps1 -Variant divq9",
                   "build_package": "artifacts/research/A05_20261113/divq9_build"},
        "validation": [
            ("board golden", "41/41 exact incl. 8 ties", RES % "golden_20261113_144159"),
            ("board full test (95,404 clean test windows)", "95,404/95,404 logits and class = canonical C-model",
             RES % "fulltest_20261113_144220"),
            ("hard real time (RAM windows, 20.48 ms period, SCHED_FIFO 50 CPU1)",
             "0 misses / 30,000, max 16.63 ms, 30,200 outputs = canonical", RES % "rt_20261113_151033")],
        "timing": "post-route WNS +0.132 ns (default strategy)"},
    "research_b02_z7020_20261113": {
        "id": "B02", "title": "Selective integrity checks (input signature, repeated tail, projection range)",
        "numeric": CANONICAL_NUMERIC, "manifest_overrides": {}, "reference": canonical_ref(),
        "outputs": dict(CLASS_OUT, integrity_status=INTEGRITY_B02),
        "source": {"hls_top": "hls/research/b02_integrity/transformer_top_seq256_raw1024_b02_argmax.cpp",
                   "generator": "hls/research/b02_integrity/make_b02_top.py",
                   "build": "powershell -File tools/build_raw_core.ps1 -Variant b02",
                   "build_package": "artifacts/research/B02_20261113/b02_build"},
        "validation": [
            ("board golden", "41/41 exact incl. 8 ties, integrity 0", RES % "golden_20261113_184354"),
            ("board full test", "95,404/95,404 = canonical C-model, integrity_status 0 on all",
             RES % "fulltest_20261113_184417"),
            ("hard real time", "0 misses / 30,000, max 17.71 ms (+1.1 ms: range check stopped loop flattening)",
             RES % "rt_20261113_191441")],
        "timing": "post-route WNS +0.202 ns (default strategy)"},
    "research_b03_z7020_20261113": {
        "id": "B03", "title": "B02 + programmable logical fault injection (research overlay)",
        "numeric": CANONICAL_NUMERIC, "manifest_overrides": {},
        "reference": canonical_ref("injection off: bit-identical to the canonical core; injection rows: "
                                   "64-bit-long C-model (hls/research/b03_inject/make_lp64_cmodel.py)"),
        "outputs": dict(CLASS_OUT, integrity_status=INTEGRITY_B02, injection=INJECTION),
        "source": {"hls_top": "hls/research/b03_inject/transformer_top_seq256_raw1024_b03_argmax.cpp",
                   "generator": "hls/research/b03_inject/make_b03_top.py",
                   "build": "powershell -File tools/build_raw_core.ps1 -Variant b03; vivado -mode batch -source "
                            "tools/vivado/rerun_impl_strategy.tcl -tclargs tmp/vb03/vb03.xpr "
                            "Performance_ExplorePostRoutePhysOpt",
                   "build_package": "artifacts/research/B03_20261113/b03_build"},
        "validation": [
            ("board golden (injection off)", "41/41 exact incl. 8 ties", RES % "golden_20261113_221324"),
            ("board injection campaign", "1,408/1,408 rows = 64-bit-long C-model; 88/93 class-changing "
             "faults flagged, 5 silent", "artifacts/research/B03_20261113/summary.json")],
        "timing": "default implementation WNS -0.437 ns (failed, kept in b03_build/first_impl_timing_fail); "
                  "Performance_ExplorePostRoutePhysOpt: WNS 0.000 ns (met, no reported margin)"},
    "research_a06_z7020_20261113": {
        "id": "A06", "title": "Unsigned 8-bit attention probabilities + exact 8-bit divider (W10/A10/P8u)",
        "numeric": dict(CANONICAL_NUMERIC,
                        attention_probabilities={"bits": 8, "signed": False, "qmax": 255, "scale": "1/255"},
                        softmax_quotient="exact 8-step restoring divider div_q8 (quotient <= 255)",
                        context_requant={"name": "M_l0_ctx_p8u_q40", "q40": 4322686742,
                                         "m": 0.00393146068905375,
                                         "rule": "m = (1/255 * s_v) / s_ctx, Q40 round-half-even"},
                        note="weights and activations stay INT10 (this is NOT a W8/A8 implementation)"),
        "manifest_overrides": {"attn_bits": 8, "attn_signed": False, "attn_qmax": 255},
        "reference": {"bit_identical_to_canonical": False, "numpy_backend": "unsupported",
                      "reconstruct": "python -m tools.check_variant_cmodel --ckpt artifacts/experiments/"
                                     "transformer_f1_seq256_pool4/kaggle_v3/seq256_pool4/scratch/best.pt --scales "
                                     "artifacts/quantized/f_seq256_int10/integer_config.json --exe "
                                     "tmp/csim/batch_raw_p8u.exe --n 600 --conditions clean snr0 "
                                     "--research-bits 10,10,8,1",
                      "golden": "artifacts/research/A06_20261113/golden_raw1024_p8u",
                      "cmodel_logits": "artifacts/research/A06_20261113/csim_p8u_fulltest/csim_logits.txt",
                      "note": "own reference: tools/research/precision_ablation.py make_model(state, 10, 10, 8, True)"},
        "outputs": dict(CLASS_OUT),
        "source": {"hls_top": "hls/research/a06_p8u/transformer_top_seq256_raw1024_p8u_argmax.cpp",
                   "generator": "hls/research/a06_p8u/make_a06_top.py",
                   "build": "powershell -File tools/build_raw_core.ps1 -Variant a06",
                   "build_package": "artifacts/research/A06_20261113/a06_build"},
        "validation": [
            ("board golden (own goldens)", "41/41 exact incl. 3 ties", RES % "golden_20261113_212942"),
            ("board full test", "95,404/95,404 = A06 C-model, accuracy 97.8848 %", RES % "fulltest_20261113_213019"),
            ("hard real time", "0 misses / 30,000, max 16.69 ms, 30,200 outputs = A06 C-model",
             RES % "rt_20261113_220122")],
        "timing": "post-route WNS +0.100 ns (default strategy)"},
}
B04_COMMON = {
    "numeric": CANONICAL_NUMERIC, "manifest_overrides": {},
    "reference": canonical_ref("logits/class bit-identical to the canonical core; checks add metadata only"),
}
VARIANTS["research_b04_z7020_20261114"] = dict(
    B04_COMMON, id="B04", title="Single-pipeline checked projection + per-row checksum (no injection hardware)",
    outputs=dict(CLASS_OUT, integrity_status=INTEGRITY_B04),
    source={"hls_top": "hls/research/b04_integrity_fast/transformer_top_seq256_raw1024_b04_argmax.cpp",
            "generator": "hls/research/b04_integrity_fast/make_b04_top.py",
            "build": "powershell -File tools/build_raw_core.ps1 -Variant b04; vivado -mode batch -source "
                     "tools/vivado/rerun_impl_strategy.tcl -tclargs tmp/vb04/vb04.xpr Performance_ExplorePostRoutePhysOpt",
            "build_package": "artifacts/research/B04_20261114/b04_build"},
    validation=[], timing="default implementation WNS -0.737 ns (failed, kept in "
                          "artifacts/research/B04_20261114/b04_build_first_impl_timing_fail); "
                          "Performance_ExplorePostRoutePhysOpt: WNS +0.024 ns")
VARIANTS["research_b04i_z7020_20261114"] = dict(
    B04_COMMON, id="B04i", title="B04 + programmable logical fault injection (research overlay, -DB04_INJECT)",
    outputs=dict(CLASS_OUT, integrity_status=INTEGRITY_B04, injection=INJECTION),
    source={"hls_top": "hls/research/b04_integrity_fast/transformer_top_seq256_raw1024_b04_argmax.cpp (-DB04_INJECT)",
            "generator": "hls/research/b04_integrity_fast/make_b04_top.py",
            "build": "powershell -File tools/build_raw_core.ps1 -Variant b04i",
            "build_package": "artifacts/research/B04_20261114/b04i_build"},
    validation=[], timing="default WNS -1.731 ns; re-runs NetDelay_high -0.620, ExtraTimingOpt -0.805 (all kept as '*_timing_fail'); Performance_ExplorePostRoutePhysOpt: WNS +0.007 ns (met, minimal margin)")
QUALITY = ("quality_detail (bits 0-7 clip flag per channel, 8-15 constant-run flag, 16-23 low-bit-occupancy flag), "
           "quality_status (bit0 any clip, bit1 any run, bit2 any low-bit, bit3 logit margin < 45), logit_margin "
           "(top1 - top2 integer logit); thresholds from clean validation (Q01v2, frozen); metadata only, the class "
           "is the FPGA argmax")
VARIANTS["research_q02_z7020_20261114"] = dict(
    numeric=CANONICAL_NUMERIC, manifest_overrides={},
    reference=canonical_ref("logits/class bit-identical to the canonical core; quality registers = Python reference "
                            "(tools/research/q02_check.py)"),
    id="Q02", title="Input-quality and decision-status metadata registers (digital replay; no analog claim)",
    outputs=dict(CLASS_OUT, quality=QUALITY),
    source={"hls_top": "hls/research/q02_quality/transformer_top_seq256_raw1024_q02_argmax.cpp",
            "generator": "hls/research/q02_quality/make_q02_top.py",
            "build": "powershell -File tools/build_raw_core.ps1 -Variant q02",
            "build_package": "artifacts/research/Q02_20261114/q02_build"},
    validation=[], timing="post-route WNS +0.147 ns (default strategy)")

EXTRA_EVIDENCE = OUT / "extra_evidence.json"   # appended by later runs: {bundle: [[what, result, path], ...]}


def sha(p):
    return hashlib.sha256((REPO / p).read_bytes()).hexdigest()


def golden_info(gdir):
    lines = (REPO / gdir / "golden_in.txt").read_text().splitlines()
    n = int(lines[0].split()[0])
    exp = [l.split() for l in (REPO / gdir / "golden_expected.txt").read_text().splitlines()[1:] if l.strip()]
    ties = sum(1 for e in exp if [int(v) for v in e[3:7]].count(max(int(v) for v in e[3:7])) > 1)
    assert len(exp) == n
    return {"dir": gdir, "count": n, "ties": ties, "sha256_in": sha(gdir + "/golden_in.txt"),
            "sha256_expected": sha(gdir + "/golden_expected.txt"),
            "format": "golden_in.txt: header 'count 1024 8', then one window of raw24 codes per line"}


def build(name):
    v = json.loads(json.dumps(VARIANTS[name]))
    extra = json.loads(EXTRA_EVIDENCE.read_text()).get(name, []) if EXTRA_EVIDENCE.exists() else []
    val = [{"what": a, "result": b, "evidence": c} for a, b, c in v.pop("validation") + [tuple(e) for e in extra]]
    top = v["source"]["hls_top"].split(" ")[0]
    v["source"]["hls_top_sha256"] = sha(top)
    contract = {"bundle": name, "schema": "research variant contract v1", **v, "input": INPUT,
                "golden": golden_info(v["reference"]["golden"]), "physical_validation": val,
                "status": "physically validated on the MYIR Z-Turn XC7Z020" if val else "built; physical validation pending",
                "board": "MYIR Z-Turn MYS-7Z020-C-S, XC7Z020-1CLG400C, 100 MHz, PYNQ 2.4 / Python 3.6"}
    return contract


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        c = build(name)
        (OUT / (name + ".json")).write_text(json.dumps(c, indent=1) + "\n", encoding="utf-8", newline="\n")
        print("wrote", name, "golden", c["golden"]["count"], "evidence", len(c["physical_validation"]))


if __name__ == "__main__":
    main()
