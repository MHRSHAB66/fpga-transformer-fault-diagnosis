#!/usr/bin/env python3
"""Build a small integrity manifest (path + size + SHA-256) for the delivery
artifacts, WITHOUT copying any multi-GB file into git.

It records the key hardware/evidence artifacts -- the deployment overlay
binaries (BIT/HWH/XSA), the full-test metrics/logs, the UART boot summary, the
Live evidence, and the UI screenshots -- so their integrity can be checked even
when the large binaries themselves stay untracked (gitignored) and live only on
disk / on the board.

    python tools/build_artifact_manifest.py
    -> artifacts/manifests/artifact_manifest.json
"""
import datetime
import hashlib
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# (glob-free) explicit paths relative to the repo root. Missing paths are
# recorded as present=false rather than failing, so the manifest is honest.
TARGETS = [
    # deployment bundle (dma16 = the working DMA-fixed build)
    "artifacts/deploy/seq256_z7020_pynq_dma16_py36_20261008/overlay/design_1_wrapper.bit",
    "artifacts/deploy/seq256_z7020_pynq_dma16_py36_20261008/overlay/design_1_wrapper.hwh",
    "artifacts/deploy/seq256_z7020_pynq_dma16_py36_20261008/overlay/seq256_dma_system.xsa",
    "artifacts/deploy/seq256_z7020_pynq_dma16_py36_20261008/config/hwh_contract.json",
    "artifacts/deploy/seq256_z7020_pynq_dma16_py36_20261008/config/model_manifest.json",
    "artifacts/deploy/seq256_z7020_pynq_dma16_py36_20261008/MANIFEST.sha256",
    # canonical physical-validation evidence dir (DMA-fixed)
    "artifacts/results/seq256_dma_system_dsp_margin_dma16_z7020_20261008/bitstream/design_1_wrapper.bit",
    "artifacts/results/seq256_dma_system_dsp_margin_dma16_z7020_20261008/bitstream/design_1.hwh",
    "artifacts/results/seq256_dma_system_dsp_margin_dma16_z7020_20261008/summary.json",
    # full physical test evidence (small files; the >1GB inputs are excluded)
    "artifacts/results/fpga_fulltest_20261008_141335/fpga_fulltest_metrics.json",
    "artifacts/results/fpga_fulltest_20261008_141335/fpga_pred.txt",
    "artifacts/results/fpga_fulltest_20261008_141335/labels.txt",
    "artifacts/results/fpga_fulltest_20261008_141335/meta.json",
    "artifacts/results/fpga_fulltest_20261008_141335/fpga_run.log",
    "artifacts/results/fpga_fulltest_20261008_141335/fpga_run.status",
    # the big inputs: recorded for integrity but intentionally NOT committed
    "artifacts/results/fpga_fulltest_20261008_141335/fpga_testset_in.i32.gz",
    # UART boot test
    "artifacts/results/uart_boot_test_20261008_194030/boot_uart.log",
    "artifacts/results/uart_boot_test_20261008_194030/summary.json",
    # Live FPGA evidence
    "artifacts/results/fpga_live_test_20261008_200509/live_evidence.json",
    "artifacts/results/fpga_live_test_20261008_200509/live_results.txt",
    # UI screenshots
    "artifacts/results/ui_smoke_20261008/replay_tab.png",
    "artifacts/results/ui_smoke_20261008/live_tab.png",
]

# Paths that are intentionally untracked (gitignored) -- flagged in the manifest.
GITIGNORED_HINT = (".bit", ".xsa", "fpga_testset_in.")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    rows = []
    for rel in TARGETS:
        p = os.path.join(REPO, rel.replace("/", os.sep))
        if os.path.isfile(p):
            size = os.path.getsize(p)
            rows.append({
                "path": rel,
                "present": True,
                "size_bytes": size,
                "sha256": sha256(p),
                "committed_in_git": not any(h in rel for h in GITIGNORED_HINT),
            })
        else:
            rows.append({"path": rel, "present": False})
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": ("Integrity manifest for delivery artifacts. Binary/large items "
                 "(.bit/.xsa and the multi-GB fpga_testset_in.*) are gitignored and "
                 "recorded here only (committed_in_git=false); canonical copies live "
                 "on disk and on the board."),
        "repo_relative": True,
        "artifacts": rows,
    }
    out_dir = os.path.join(REPO, "artifacts", "manifests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "artifact_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    present = sum(1 for r in rows if r.get("present"))
    print("wrote %s (%d/%d artifacts present)" % (out_path, present, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
