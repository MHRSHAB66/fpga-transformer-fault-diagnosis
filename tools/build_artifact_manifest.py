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
import subprocess

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
    # argmax overlay (class decided in FPGA logic) = live deployment since 2026-10-23
    "artifacts/deploy/seq256_argmax_z7020_20261022/overlay/design_1_wrapper.bit",
    "artifacts/deploy/seq256_argmax_z7020_20261022/overlay/design_1_wrapper.hwh",
    "artifacts/deploy/seq256_argmax_z7020_20261022/overlay/seq256_dma_system.xsa",
    "artifacts/deploy/seq256_argmax_z7020_20261022/config/hwh_contract.json",
    "artifacts/deploy/seq256_argmax_z7020_20261022/MANIFEST.sha256",
    "artifacts/results/seq256_dma_argmax_z7020_20261022/bitstream/design_1_wrapper.bit",
    "artifacts/results/seq256_dma_argmax_z7020_20261022/bitstream/design_1.hwh",
    "artifacts/results/seq256_dma_argmax_z7020_20261022/bitstream/seq256_dma_system.xsa",
    # board vs reference, physical logits and real-time bench (2026-10-22)
    "artifacts/results/board_vs_reference_20261022/comparison.json",
    "artifacts/results/fpga_realtime_bench_20261022/realtime_verdict.json",
    "artifacts/results/fpga_realtime_bench_20261022/board_full_pipelined/logits_i32.bin",
    "artifacts/results/fpga_realtime_bench_20261022/board_full_pipelined/timing.npz",
    "artifacts/results/fpga_realtime_bench_20261022/board_full_baseline/timing.npz",
    # argmax overlay full test (2026-10-23)
    "artifacts/results/fpga_argmax_fulltest_20261023/live_evidence.json",
    "artifacts/results/fpga_argmax_fulltest_20261023/live_frames.csv",
    "artifacts/results/fpga_argmax_fulltest_20261023/live_frames.jsonl",
    # confidence calibration (validation-only fit)
    "artifacts/calibration/conf_temperature_v1/temperature.json",
    # hard real-time raw timings (stock kernel, then PREEMPT_RT one-shot boot)
    "artifacts/results/hard_rt_20261024/R0_normal_prio/remote/out/rt_raw.npz",
    "artifacts/results/hard_rt_20261024/R1_fifo50_cpu1/remote/out/rt_raw.npz",
    "artifacts/results/hard_rt_20261024/R2_fifo50_cpu1_repeat/remote/out/rt_raw.npz",
    "artifacts/results/hard_rt_20261024/L1_fifo50_cpu1_100k/remote/out/rt_raw.npz",
    "artifacts/results/hard_rt_preempt_rt_20261024/K1_rtkernel_fifo50_cpu1/remote/out/rt_raw.npz",
    "artifacts/results/hard_rt_preempt_rt_20261024/K2_rtkernel_fifo50_cpu1_repeat/remote/out/rt_raw.npz",
    "artifacts/results/hard_rt_preempt_rt_20261024/K3_rtkernel_fifo50_cpu1_100k/remote/out/rt_raw.npz",
    # RAM real-time demo set and runs, TCP end-to-end criterion runs
    "artifacts/ram_demo_sets/class_balanced_256_v1/set.i32",
    "artifacts/ram_demo_sets/class_balanced_256_v1/provenance.json",
    "artifacts/results/ram_rt_demo_20261024_criterion30k/remote/out/rt_raw.npz",
    "artifacts/results/e2e_demo_gui_20261024_231859/pc_evidence.json",
    "artifacts/results/e2e_demo_gui_20261024_231859/frames.tsv",
    "artifacts/results/e2e_demo_gui_20261024_231859/remote/out/rt_raw.npz",
    "artifacts/results/e2e_demo_gui_20261024_233000/pc_evidence.json",
    "artifacts/results/e2e_demo_gui_20261024_233000/frames.tsv",
    "artifacts/results/e2e_demo_gui_20261024_233000/remote/out/rt_raw.npz",
    # tail-latency study raw outputs (2026-10-23)
    "artifacts/results/fpga_timing_20261023/A_legacy_unpaced_full/remote/workload.out",
    "artifacts/results/fpga_timing_20261023/B_prefetch128_unpaced_full/remote/workload.out",
    "artifacts/results/fpga_timing_20261023/B2_proc128_unpaced_full/remote/workload.out",
    "artifacts/results/fpga_timing_20261023/C_prefetch128_paced48828_full/remote/workload.out",
    "artifacts/results/fpga_timing_20261023/S1_L_stream_30k/workload_stream.out",
    "artifacts/results/fpga_timing_20261023/S2_RA32_stream_30k/workload_stream.out",
    "artifacts/results/fpga_timing_20261023/S3_L_stream_30k/workload_stream.out",
    "artifacts/results/fpga_timing_20261023/S4_RA32_stream_30k/workload_stream.out",
    # board boot rollback package and the PREEMPT_RT candidate build
    "artifacts/rollback/board_boot_backup_20261024/BOOT.BIN",
    "artifacts/rollback/board_boot_backup_20261024/image.ub",
    "artifacts/rollback/board_boot_backup_20261024/sd_p1_boot_files.tar",
    "artifacts/rollback/board_boot_backup_20261024/qspi_mtd0.bin",
    "artifacts/rollback/board_boot_backup_20261024/qspi_mtd1.bin",
    "artifacts/rollback/board_boot_backup_20261024/qspi_mtd2.bin",
    "artifacts/rollback/board_boot_backup_20261024/qspi_mtd3.bin",
    "artifacts/preempt_rt_candidate_20261024/image_rt.ub",
    "artifacts/preempt_rt_candidate_20261024/modules_rt.tar.gz",
    # final evaluation figures (numbers behind docs/figures/final/)
    "artifacts/results/final_figures_20261109/figure_data.json",
    # raw-input core: preprocessing inside the FPGA (2026-11-09)
    "artifacts/preprocessing/raw_input_contract_v1.json",
    "artifacts/results/fpga_rawinput_testset_20261109/fpga_testset_raw.i32.gz",
    "artifacts/results/seq256_dma_raw_argmax_z7020_20261109/bitstream/design_1_wrapper.bit",
    "artifacts/results/seq256_dma_raw_argmax_z7020_20261109/bitstream/design_1.hwh",
    "artifacts/results/seq256_dma_raw_argmax_z7020_20261109/bitstream/seq256_dma_system.xsa",
    "artifacts/deploy/seq256_raw_argmax_z7020_20261109/overlay/design_1_wrapper.bit",
    "artifacts/deploy/seq256_raw_argmax_z7020_20261109/overlay/seq256_dma_system.xsa",
    "artifacts/deploy/seq256_raw_argmax_z7020_20261109/MANIFEST.sha256",
    "artifacts/ram_demo_sets/class_balanced_256_raw_v1/set.i32",
    "artifacts/results/raw_core_board_fulltest_20261109_231348/fulltest.out",
    "artifacts/results/raw_core_board_fulltest_20261109_231348/fulltest_check.json",
    "artifacts/results/raw_core_board_rt_20261109_234531/out/rt_raw.npz",
    # UART result path and CPU baseline (2026-11-09)
    "artifacts/results/uart_results_20261109_205416/uart_result_check.json",
    "artifacts/results/cpu_baseline_20261109/summary.json",
    # raw-input core + temporal vote in FPGA logic (2026-11-11 build)
    "artifacts/results/seq256_dma_vote_z7020_20261110/bitstream/design_1_wrapper.bit",
    "artifacts/results/seq256_dma_vote_z7020_20261110/bitstream/design_1.hwh",
    "artifacts/results/seq256_dma_vote_z7020_20261110/bitstream/seq256_dma_system.xsa",
    "artifacts/deploy/seq256_vote_z7020_20261110/overlay/design_1_wrapper.bit",
    "artifacts/deploy/seq256_vote_z7020_20261110/overlay/seq256_dma_system.xsa",
    "artifacts/deploy/seq256_vote_z7020_20261110/MANIFEST.sha256",
    "artifacts/results/raw_core_board_fulltest_20261111_130927/fulltest.out",
    "artifacts/results/raw_core_board_fulltest_20261111_130927/fulltest_check.json",
    "artifacts/results/raw_core_board_rt_20261111_133749/out/rt_raw.npz",
    # paper study: noise-augmented weights on the vote core (second bitstream)
    "artifacts/results/seq256_dma_vote_noiseaug_z7020_20261111/bitstream/design_1_wrapper.bit",
    "artifacts/results/seq256_dma_vote_noiseaug_z7020_20261111/bitstream/seq256_dma_system.xsa",
    "artifacts/deploy/seq256_vote_noiseaug_z7020_20261111/overlay/design_1_wrapper.bit",
    "artifacts/deploy/seq256_vote_noiseaug_z7020_20261111/MANIFEST.sha256",
    "artifacts/results/noisy_testsets_20261111/fpga_testset_raw_snr0.i32.gz",
    "artifacts/results/raw_core_board_rt_20261111_141343/out/rt_raw.npz",
    # board runs of the noise-augmented second bitstream and of both models at 0 dB
    "artifacts/results/raw_core_board_fulltest_20261111_155757/fulltest.out",
    "artifacts/results/raw_core_board_fulltest_20261111_163200/fulltest.out",
    "artifacts/results/raw_core_board_fulltest_20261111_170226/fulltest.out",
    # 10-class study (Kaggle): per-window test predictions and logits
    "artifacts/results/tenclass_kaggle_20261111/seq256_10class/test_predictions.npz",
    "artifacts/results/raw_core_board_rt_20261111_173322/out/rt_raw.npz",
    # research A05: raw-input core with the exact 9-bit softmax divider
    "artifacts/research/A05_20261113/divq9_build/bitstream/design_1_wrapper.bit",
    "artifacts/research/A05_20261113/divq9_build/bitstream/seq256_dma_system.xsa",
    "artifacts/deploy/research_divq9_z7020_20261113/overlay/design_1_wrapper.bit",
    "artifacts/deploy/research_divq9_z7020_20261113/MANIFEST.sha256",
    "artifacts/results/raw_core_board_fulltest_20261113_144220/fulltest.out",
    "artifacts/results/raw_core_board_rt_20261113_151033/out/rt_raw.npz",
    # research B02: selective integrity checks
    "artifacts/research/B02_20261113/b02_build/bitstream/design_1_wrapper.bit",
    "artifacts/research/B02_20261113/b02_build/bitstream/seq256_dma_system.xsa",
    "artifacts/deploy/research_b02_z7020_20261113/overlay/design_1_wrapper.bit",
    "artifacts/deploy/research_b02_z7020_20261113/MANIFEST.sha256",
    "artifacts/results/raw_core_board_fulltest_20261113_184417/fulltest.out",
    "artifacts/results/raw_core_board_rt_20261113_191441/out/rt_raw.npz",
    "artifacts/research/S01_20261113/quiet/out/rt_raw.npz",
    "artifacts/research/S01_20261113/cpu0_busy/out/rt_raw.npz",
    "artifacts/research/S01_20261113/mem_copy/out/rt_raw.npz",
    "artifacts/research/S01_20261113/cache_thrash/out/rt_raw.npz",
    "artifacts/research/S01_20261113/sd_io/out/rt_raw.npz",
    "artifacts/research/S01_20261113/net_rx/out/rt_raw.npz",
    "artifacts/research/S01_20261113/combined/out/rt_raw.npz",
    # research A02: per-variant validation logits and attention diagnostics
    "artifacts/research/A02_20261113/int10/val_outputs.npz",
    "artifacts/research/A02_20261113/int8/val_outputs.npz",
    "artifacts/research/A02_20261113/w8/val_outputs.npz",
    "artifacts/research/A02_20261113/a8/val_outputs.npz",
    "artifacts/research/A02_20261113/p8s/val_outputs.npz",
    "artifacts/research/A02_20261113/p9s/val_outputs.npz",
    "artifacts/research/A02_20261113/p8u/val_outputs.npz",
    "artifacts/research/A02_20261113/w8a8_p9s/val_outputs.npz",
    "artifacts/research/A02_20261113/w8a8_p10s/val_outputs.npz",
    "artifacts/research/A02_20261113/w8a8_p8u/val_outputs.npz",
    # research Q01: raw validation windows for the bit-exact C-model
    "artifacts/results/fpga_rawinput_valset_20261113/fpga_valset_raw.i32.gz",
    # research A06: unsigned 8-bit probabilities + exact 8-bit divider
    "artifacts/research/A06_20261113/a06_build/bitstream/design_1_wrapper.bit",
    "artifacts/research/A06_20261113/a06_build/bitstream/seq256_dma_system.xsa",
    "artifacts/deploy/research_a06_z7020_20261113/overlay/design_1_wrapper.bit",
    "artifacts/deploy/research_a06_z7020_20261113/MANIFEST.sha256",
    "artifacts/results/raw_core_board_fulltest_20261113_213019/fulltest.out",
    "artifacts/results/raw_core_board_rt_20261113_220122/out/rt_raw.npz",
    # research B03: logical fault-injection overlay (timing-clean re-implementation + first failing one)
    "artifacts/research/B03_20261113/b03_build/bitstream/design_1_wrapper.bit",
    "artifacts/research/B03_20261113/b03_build/bitstream/seq256_dma_system.xsa",
    "artifacts/research/B03_20261113/b03_build/first_impl_timing_fail/bitstream/design_1_wrapper.bit",
    "artifacts/deploy/research_b03_z7020_20261113/overlay/design_1_wrapper.bit",
    "artifacts/deploy/research_b03_z7020_20261113/MANIFEST.sha256",
]


def tracked_paths():
    """Paths git tracks (committed_in_git is read from git, not guessed)."""
    out = subprocess.run(["git", "-C", REPO, "ls-files", "-z"], capture_output=True,
                         check=True).stdout.decode("utf-8")
    return set(p for p in out.split("\0") if p)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def blob_digest(rel):
    """Size and SHA-256 of the committed (index) bytes of a tracked file.

    Tracked text files are checked out with CRLF on Windows (core.autocrlf), so
    hashing the working copy would make the manifest depend on the platform.
    """
    data = subprocess.run(["git", "-C", REPO, "cat-file", "blob", ":" + rel],
                          capture_output=True, check=True).stdout
    return len(data), hashlib.sha256(data).hexdigest()


def main():
    rows = []
    tracked = tracked_paths()
    for rel in TARGETS:
        p = os.path.join(REPO, rel.replace("/", os.sep))
        if rel in tracked:
            size, digest = blob_digest(rel)
        elif os.path.isfile(p):
            size, digest = os.path.getsize(p), sha256(p)
        else:
            rows.append({"path": rel, "present": False})
            continue
        rows.append({
            "path": rel,
            "present": True,
            "size_bytes": size,
            "sha256": digest,
            "committed_in_git": rel in tracked,
        })
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": ("Integrity manifest for delivery artifacts. Binary/large items "
                 "(.bit/.xsa, the multi-GB fpga_testset_in.*, raw .npz timings, "
                 "multi-MB run dumps, board boot binaries) are gitignored and "
                 "recorded here only (committed_in_git=false); copies live on disk, "
                 "in the off-repo backup and on the board."),
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
