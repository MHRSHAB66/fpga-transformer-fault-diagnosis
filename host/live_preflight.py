"""Non-destructive preflight checks for Live FPGA mode.

Run BEFORE Start to catch problems early with a clear message. It is strictly
read-only: it checks files, versions, the DMA contract and temperature
availability, and imports the driver MODULE (which only defines classes) -- it
NEVER loads the overlay, programs the FPGA, or runs any inference.

Design: ``run_probe`` does the single SSH round-trip and returns a facts dict;
``evaluate`` is a PURE function (facts -> list of checks) so the validation logic
is unit-testable with a mock facts dict and no board.
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

from host.live_fpga import (
    _ssh_base, BOARD_HOST, BOARD_USER, SSH_KEY, DEP_DIR, BIT, IN_GZ,
)

AVAILABLE_WINDOWS = 95404          # test-set size staged on the board
TEMP_STOP_C = 75.0                 # same threshold as the runner and the guard
# Size of fpga_testset_in.i32.gz (artifacts/results/fpga_fulltest_20261008_141335);
# a truncated copy would only fail late, mid-run.
EXPECT_INPUT_BYTES = 1128562912
# Local copy of the deployed bundle; its MANIFEST.sha256 is the identity
# reference for the BIT/HWH on the board.
LOCAL_BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                            "artifacts", "deploy", "seq256_argmax_z7020_20261022")


def expected_overlay_hashes(bundle: str = LOCAL_BUNDLE) -> Dict[str, Optional[str]]:
    """sha256 of overlay/design_1_wrapper.{bit,hwh} from the bundle manifest."""
    out = {"bit": None, "hwh": None}
    try:
        with open(os.path.join(bundle, "MANIFEST.sha256"), encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1].endswith("overlay/design_1_wrapper.bit"):
                    out["bit"] = parts[0]
                elif len(parts) == 2 and parts[1].endswith("overlay/design_1_wrapper.hwh"):
                    out["hwh"] = parts[0]
    except OSError:
        pass
    return out
EXPECT_DMA_MAX_BYTES = 65536       # dma_sg_length_width=16 (DMA-fixed build)
EXPECT_FRAME_BYTES = 32768         # 8192 int32 words = one full window


def _probe_script(dep_dir: str, bit: str, in_gz: str) -> str:
    hwh = bit[:-4] + ".hwh" if bit.endswith(".bit") else bit + ".hwh"
    contract = dep_dir + "/config/hwh_contract.json"
    drv = dep_dir + "/driver"
    return (
        'echo PY=$(python3.6 -V 2>&1); '
        'echo DEP_EXISTS=$([ -d "%(dep)s" ] && echo 1 || echo 0); '
        'echo BIT_EXISTS=$([ -f "%(bit)s" ] && echo 1 || echo 0); '
        'echo HWH_EXISTS=$([ -f "%(hwh)s" ] && echo 1 || echo 0); '
        'echo IN_EXISTS=$([ -f "%(in)s" ] && echo 1 || echo 0); '
        'echo IN_BYTES=$(stat -c %%s "%(in)s" 2>/dev/null); '
        'echo DRIVER_IMPORT=$(cd "%(drv)s" && PYTHONPATH="%(drv)s" python3.6 -c '
        '"import fault_classifier; print(\'ok\')" 2>/dev/null || echo fail); '
        'echo BIT_SHA=$(sha256sum "%(bit)s" 2>/dev/null | cut -d" " -f1); '
        'echo HWH_SHA=$(sha256sum "%(hwh)s" 2>/dev/null | cut -d" " -f1); '
        'echo XADC_RAW=$(cat /sys/bus/iio/devices/iio:device0/in_temp0_raw 2>/dev/null); '
        "echo TEMP_C=$(awk -v r=\"$(cat /sys/bus/iio/devices/iio:device0/in_temp0_raw 2>/dev/null)\" "
        "-v o=\"$(cat /sys/bus/iio/devices/iio:device0/in_temp0_offset 2>/dev/null)\" "
        "-v s=\"$(cat /sys/bus/iio/devices/iio:device0/in_temp0_scale 2>/dev/null)\" "
        "'BEGIN{if (r ~ /^-?[0-9]+$/ && o ~ /^-?[0-9.]+$/ && s ~ /^[0-9.]+$/) printf \"%%.3f\",(r+o)*s/1000}'); "
        # Other runners/benches/guards (incl. the RAM real-time demo and the hard-RT
        # acceptance runner). Each alternative is bracketed so the
        # pattern cannot match this probe's own command line.
        "echo OTHER=$(ps -eo pid=,args= | grep -E "
        "'python3[.]6 .*([f]pga_live_runner|[f]pga_stage_bench|[f]pga_fulltest_bin_runner|[f]pga_ram_rt_demo|[f]pga_rt_acceptance)|[t]emp_guard[.]sh' "
        "| awk '{print $1}' | paste -sd, -); "
        # bracket trick ([f]pga...) so this pgrep does not match its own shell
        # command line, which literally contains the pattern string
        'echo RUNNER=$(pgrep -f "[f]pga_live_runner.py" | head -1); '
        'echo CONTRACT_EXISTS=$([ -f "%(ct)s" ] && echo 1 || echo 0); '
        'echo DMA_MAX=$(grep -o \'"dma_max_transfer_bytes"[^,]*\' "%(ct)s" | grep -o \'[0-9]*\'); '
        'echo FRAME_BYTES=$(grep -o \'"input_frame_bytes"[^,]*\' "%(ct)s" | grep -o \'[0-9]*\')'
        % {"dep": dep_dir, "bit": bit, "hwh": hwh, "in": in_gz, "drv": drv, "ct": contract}
    )


def run_probe(host=BOARD_HOST, user=BOARD_USER, key=SSH_KEY, port=22,
              dep_dir=DEP_DIR, bit=BIT, in_gz=IN_GZ, timeout=20) -> Dict[str, str]:
    """One read-only SSH round-trip; returns a facts dict (adds ``reachable``)."""
    cf = (0x08000000 if os.name == "nt" else 0)
    try:
        r = subprocess.run(_ssh_base(host, user, key, port, timeout=timeout) +
                           [_probe_script(dep_dir, bit, in_gz)],
                           capture_output=True, text=True, timeout=timeout + 8,
                           creationflags=cf)
    except Exception as e:
        return {"reachable": "0", "error": str(e)}
    if r.returncode != 0:
        return {"reachable": "0", "error": (r.stderr or r.stdout or "ssh failed").strip()[:300]}
    facts = {"reachable": "1"}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            facts[k.strip()] = v.strip()
    return facts


def evaluate(facts: Dict[str, str], n: int,
             available: int = AVAILABLE_WINDOWS,
             expected_hashes: Optional[Dict[str, Optional[str]]] = None) -> List[Dict]:
    """Pure: facts -> ordered list of {check, ok, detail}. No I/O."""
    def chk(name, ok, detail):
        return {"check": name, "ok": bool(ok), "detail": detail}

    out: List[Dict] = []
    reachable = facts.get("reachable") == "1"
    out.append(chk("ssh_connectivity", reachable,
                   "reachable" if reachable else facts.get("error", "unreachable")))
    if not reachable:
        return out  # nothing else is meaningful without a connection

    py = facts.get("PY", "")
    out.append(chk("python3.6", py.startswith("Python 3.6"), py or "python3.6 not found"))
    out.append(chk("deployment_dir", facts.get("DEP_EXISTS") == "1", DEP_DIR))
    out.append(chk("bitstream_present", facts.get("BIT_EXISTS") == "1", "design_1_wrapper.bit"))
    out.append(chk("hwh_present", facts.get("HWH_EXISTS") == "1", "design_1_wrapper.hwh"))
    exp = expected_hashes if expected_hashes is not None else expected_overlay_hashes()
    for key, name in (("bit", "BIT_SHA"), ("hwh", "HWH_SHA")):
        got = facts.get(name, "")
        out.append(chk("%s_identity" % key, bool(exp.get(key)) and got == exp.get(key),
                       "board %s, expected %s" % (got[:12] or "?", (exp.get(key) or "?")[:12])))
    out.append(chk("input_present", facts.get("IN_EXISTS") == "1", IN_GZ))
    in_bytes = _to_int(facts.get("IN_BYTES"))
    out.append(chk("input_complete", in_bytes == EXPECT_INPUT_BYTES,
                   "%s bytes (expect %d)" % (in_bytes, EXPECT_INPUT_BYTES)))
    out.append(chk("driver_import", facts.get("DRIVER_IMPORT") == "ok",
                   "import fault_classifier (no overlay load)"))

    dma_max = _to_int(facts.get("DMA_MAX"))
    out.append(chk("dma_contract_65536B", dma_max == EXPECT_DMA_MAX_BYTES,
                   "dma_max_transfer_bytes=%s (expect %d)" % (dma_max, EXPECT_DMA_MAX_BYTES)))
    frame = _to_int(facts.get("FRAME_BYTES"))
    out.append(chk("input_frame_32768B", frame == EXPECT_FRAME_BYTES,
                   "input_frame_bytes=%s (expect %d)" % (frame, EXPECT_FRAME_BYTES)))

    valid_n = isinstance(n, int) and 1 <= n <= available
    out.append(chk("requested_n_valid", valid_n,
                   "n=%s (allowed 1..%d)" % (n, available)))

    raw = _to_int(facts.get("XADC_RAW"))
    out.append(chk("xadc_temperature", raw is not None,
                   "in_temp0_raw=%s" % facts.get("XADC_RAW", "")))
    temp = _to_float(facts.get("TEMP_C"))
    out.append(chk("temperature_below_stop", temp is not None and temp < TEMP_STOP_C,
                   "%s C (start only below %.0f C)" % (facts.get("TEMP_C") or "invalid", TEMP_STOP_C)))
    other = (facts.get("OTHER") or "").strip(",")
    out.append(chk("no_other_workload_or_guard", not other,
                   "clear" if not other else "pids %s (runner/bench/guard) still running" % other))
    no_runner = not facts.get("RUNNER")
    out.append(chk("no_prior_runner", no_runner,
                   "clear" if no_runner else "fpga_live_runner.py pid %s running" % facts.get("RUNNER")))
    return out


def all_ok(checks: List[Dict]) -> bool:
    return all(c["ok"] for c in checks)


def _to_float(s: Optional[str]):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def _to_int(s: Optional[str]):
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def preflight(host=BOARD_HOST, user=BOARD_USER, key=SSH_KEY, port=22, n=150,
              dep_dir=DEP_DIR, bit=BIT, in_gz=IN_GZ) -> List[Dict]:
    """Convenience: probe + evaluate in one call."""
    facts = run_probe(host, user, key, port, dep_dir, bit, in_gz)
    return evaluate(facts, n)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Non-destructive Live-FPGA preflight (read-only).")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--host", default=BOARD_HOST)
    args = ap.parse_args(argv)
    checks = preflight(host=args.host, n=args.n)
    print("Live FPGA preflight (read-only; no overlay load, no inference):")
    for c in checks:
        print("  [%s] %-22s %s" % ("OK " if c["ok"] else "XX ", c["check"], c["detail"]))
    ok = all_ok(checks)
    print("RESULT:", "PASS -- safe to Start" if ok else "FAIL -- fix the XX items before Start")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
