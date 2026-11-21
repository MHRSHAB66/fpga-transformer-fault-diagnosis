"""Write the Day-2 evidence manifest (paths, SHA-256, sources, commands) and the
numeric-test execution record. Read-only with respect to all evidence files.

    python -m pytest tests/test_int10_numeric_contract.py -q \
        --junitxml artifacts/experiments/day2_calibration_sensitivity_20261120/audit/pytest_int10_contract.xml
    python -m tools.day2_evidence_manifest
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "artifacts" / "experiments" / "day2_calibration_sensitivity_20261120"
AUDIT = EXP / "audit"
XILINX_INC = Path("G:/Xilinx/Vitis_HLS/2020.2/include")

# Kaggle runs that produced kaggle_val/ (queried with `kaggle kernels list --mine` /
# `kaggle kernels status` on 2026-11-21; versions from the `kaggle kernels push` output).
KAGGLE = [
    {"kernel": f"mehrdadsheikhabbasi/fpga-fault-day2-calib-sens-shard-{s}", "version": v,
     "status": "COMPLETE", "last_run_time_utc": t, "machine": "CPU, os.cpu_count()=4",
     "output": f"kaggle_val/val_shard{s}.json.gz", "log": f"kaggle_val/shard{s}.kaggle.log",
     "summary_line": line}
    for s, v, t, line in (
        (0, 2, "2026-11-20 17:24:01", "[eval] val shard 0/4: 49 files, 11956 windows, 1572s"),
        (1, 1, "2026-11-20 17:29:13", "[eval] val shard 1/4: 49 files, 11956 windows, 1579s"),
        (2, 1, "2026-11-20 17:29:18", "[eval] val shard 2/4: 49 files, 11956 windows, 826s"),
        (3, 1, "2026-11-20 17:29:22", "[eval] val shard 3/4: 48 files, 11712 windows, 1511s"))]

COMMANDS = {
    "tests": "python -m pytest tests/test_int10_numeric_contract.py -q",
    "calibrate": "python -m src.quantize.calibration_sensitivity calibrate --seeds 42 1 2 --sizes 12 20 40 80 --workers 4 --out " + EXP.relative_to(ROOT).as_posix(),
    "kaggle_build": "python kaggle/day2_calib_sensitivity/build_kernels.py",
    "kaggle_push": "kaggle kernels push -p kaggle/day2_calib_sensitivity/shard{0,1,2,3}",
    "combine": "python -m src.quantize.calibration_sensitivity combine --calib {e}/calibration_sets.json --shards {e}/kaggle_val/val_shard*.json.gz --out {e}".format(e=EXP.relative_to(ROOT).as_posix()),
    "audit_recompute": "python -m tools.audit_day2_calibration --reproduce-files 2",
    "audit_q40_alternatives": "python -m tools.audit_day2_q40_alternatives --scan-files 12",
    "manifest": "python -m tools.day2_evidence_manifest",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tool_version(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
    except Exception as exc:                      # noqa: BLE001 - record, never fail
        return f"unavailable ({type(exc).__name__})"


def test_record():
    xml = AUDIT / "pytest_int10_contract.xml"
    if not xml.exists():
        return {"status": "NOT RUN (no junit xml)"}
    root = ET.parse(xml).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = []
    for tc in suite.iter("testcase"):
        state = "passed"
        for tag in ("failure", "error", "skipped"):
            if tc.find(tag) is not None:
                state = tag
        cases.append({"test": f"{tc.get('classname').split('.')[-1]}::{tc.get('name')}",
                      "seconds": float(tc.get("time", 0)), "result": state})
    import numpy
    sys.path.insert(0, str(ROOT))
    from tests.test_int10_numeric_contract import _Harness
    built = {k: _Harness.get(k) is not None for k in ("native", "ap")}
    return {
        "junit": xml.relative_to(EXP).as_posix(), "junit_sha256": sha(xml),
        "timestamp": suite.get("timestamp"), "tests": int(suite.get("tests")),
        "failures": int(suite.get("failures")), "errors": int(suite.get("errors")),
        "skipped": int(suite.get("skipped")), "seconds": float(suite.get("time")),
        "harness_builds_available": built,
        "gxx": tool_version([shutil.which("g++") or "g++", "--version"]),
        "sizeof_long_on_this_gxx": 4 if platform.system() == "Windows" else 8,
        "xilinx_include": str(XILINX_INC) if (XILINX_INC / "ap_int.h").exists() else None,
        "python": sys.version.split()[0], "numpy": numpy.__version__,
        "platform": platform.platform(), "cases": cases}


def main():
    files = sorted(p for p in EXP.rglob("*") if p.is_file() and p.name != "evidence_manifest.json")
    manifest = {
        "experiment": "day2_calibration_sensitivity_20261120",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": "docs/reports/day2_quantization_validation_20261120.md",
        "deployed_reference": {
            "integer_config": "artifacts/quantized/f_seq256_int10/integer_config.json",
            "integer_config_sha256_lf": hashlib.sha256(
                (ROOT / "artifacts/quantized/f_seq256_int10/integer_config.json").read_bytes()
                .replace(b"\r\n", b"\n")).hexdigest(),
            "weights_header_sha256": sha(ROOT / "hls/f1_int10_seq256/vitis/weights_int10.h"),
            "hls_top_sha256": sha(ROOT / "hls/f1_int10_seq256/vitis/transformer_top_seq256_ingress1024_dsp_margin.cpp"),
        },
        "kaggle_runs": KAGGLE,
        "local_runs": {"calibration_sets.json": "logs/calibrate_with_n80.log (12-config run that wrote it); "
                                                "logs/calibrate.log is the earlier 9-config run (identical scales)",
                       "combine": "logs/combine_val.log"},
        "commands": COMMANDS,
        "numeric_tests": test_record(),
        "files": [{"path": p.relative_to(EXP).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
                  for p in files],
    }
    (AUDIT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {len(files)} files; tests: {manifest['numeric_tests'].get('tests')}")


if __name__ == "__main__":
    main()
