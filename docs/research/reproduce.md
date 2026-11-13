# Research program — reproduction guide (2026-11-13)

All commands run from the repository root (Windows, Git Bash or PowerShell), Python
3.12 with the repository requirements, g++ 15 (`/c/Users/mine_pc/scoop/apps/gcc/current/bin`),
Vitis HLS / Vivado 2020.2 at `G:/Xilinx`. Board: MYIR Z-Turn XC7Z020 at 192.168.2.99
(root SSH key `~/.ssh/id_zturn`). After every board reboot pin the Ethernet link to
100 Mb/s (`ssh root@192.168.2.99 'ethtool -s eth0 advertise 0x008'`); at 1 Gb/s the
link drops under heavy traffic. Board jobs run under `tools/board/temp_guard.sh` (75 C stop).
Status, outcomes and limitations of every experiment: `docs/research/experiment_ledger.json`;
claims and equipment: `docs/research/claims_evidence_equipment.md`.

## F00 — baseline identity and gates
    python -m tools.research.freeze_baseline --out artifacts/research/F00_20261113
Gates (frozen before results): `artifacts/research/F00_20261113/preregistered_protocol.json`.

## H01 — reporting hygiene
    python -m tools.raw_core_board_test recheck-votes --run artifacts/results/raw_core_board_fulltest_20261111_<run>
    python -m tools.plot_paper_figures

## A01/A02 — attention precision ablation (validation)
Needs the window cache `D:/fpga_cache/seq256_all` (`python -m tools.generalization_study cache`).
    python -m tools.research.precision_ablation toy --out artifacts/research/A01_20261113
    python -m tools.research.precision_ablation selfcheck --out artifacts/research/A02_20261113
    OMP_NUM_THREADS=1 python -m tools.research.precision_ablation run --out artifacts/research/A02_20261113 --workers 8
    python -m tools.research.precision_ablation summarize --out artifacts/research/A02_20261113

## A03/A04/A05 — exact 9-bit softmax divider
    g++ -O2 -std=c++17 -I hls/f1_int10_seq256/vitis -I hls/research/a04_softmax_div \
        hls/research/a04_softmax_div/test_div_q9_exhaustive.cpp -o tmp/a04_exh.exe && tmp/a04_exh.exe
    python hls/research/a04_softmax_div/make_divq9_top.py
    cd tmp && vitis_hls -f ../hls/research/a04_softmax_div/run_a04_probe.tcl      # single-operator HLS
    vivado -mode batch -source hls/research/a04_softmax_div/ooc_impl.tcl -tclargs <rtl dir> a04_probe <out> 10
    powershell -File tools/build_raw_core.ps1 -Variant divq9                     # integrated build
    python tools/package_seq256_pynq_deploy.py --pkg artifacts/research/A05_20261113/divq9_build \
        --bundle artifacts/deploy/research_divq9_z7020_20261113 --golden hls/f1_int10_seq256/golden_raw1024_argmax
    python -m tools.raw_core_board_test deploy --bundle artifacts/deploy/research_divq9_z7020_20261113
    python -m tools.raw_core_board_test golden|fulltest|rt --name research_divq9_z7020_20261113

## A06 — unsigned 8-bit probabilities + exact 8-bit divider (own reference)
    python hls/research/a06_p8u/make_a06_top.py
    python -m tools.check_variant_cmodel --ckpt <deployed best.pt> --scales artifacts/quantized/f_seq256_int10/integer_config.json \
        --exe tmp/csim/batch_raw_p8u.exe --n 600 --conditions clean snr0 --research-bits 10,10,8,1
    python -m tools.make_variant_golden --ckpt <deployed best.pt> --scales artifacts/quantized/f_seq256_int10/integer_config.json \
        --dst artifacts/research/A06_20261113/golden_raw1024_p8u --research-bits 10,10,8,1
    powershell -File tools/build_raw_core.ps1 -Variant a06
    (board: fulltest with --cmodel artifacts/research/A06_20261113/csim_p8u_fulltest/csim_logits.txt)

## B01/B02/B03 — logical fault injection and selective integrity
    python -m tools.research.fault_injection campaign --out artifacts/research/B01_20261113
    python -m tools.research.fault_injection weights|vote --out artifacts/research/B01_20261113
    python -m tools.research.fault_injection campaign --seed 20261114 --out artifacts/research/B02_20261113/heldout_campaign
    python hls/research/b02_integrity/make_b02_top.py && powershell -File tools/build_raw_core.ps1 -Variant b02
    python hls/research/b03_inject/make_b03_top.py  && powershell -File tools/build_raw_core.ps1 -Variant b03
    python -m tools.research.b03_campaign manifest|cmodel|analyze --out artifacts/research/B03_20261113
    python -m tools.research.b03_campaign board --out artifacts/research/B03_20261113 --name research_b03_z7020_20261113
    # default implementation missed timing (-0.437 ns); re-implemented with:
    vivado -mode batch -source hls/f1_int10_seq256/vitis/rerun_impl_strategy.tcl -tclargs tmp/vb03/vb03.xpr Performance_ExplorePostRoutePhysOpt
    # 64-bit-long C-model (MinGW long is 32 bit): see hls/research/b03_inject/make_lp64_cmodel.py, then
    python -m tools.research.b03_campaign cmodel --out artifacts/research/B03_20261113 --lp64

## T01 — temporal policies (offline)
    python -m tools.research.temporal_policies --out artifacts/research/T01_20261113

## Q01 — input quality and margin (offline, bit-exact C-model)
    python -m tools.export_fpga_testset_raw export --split val --out artifacts/results/fpga_rawinput_valset_20261113
    python -m tools.research.q01_quality --out artifacts/research/Q01_20261113

## S01 — interference envelope (board)
    python -m tools.research.s01_interference --name seq256_vote_z7020_20261110 --out artifacts/research/S01_20261113

## Archive
Large binaries (BIT/XSA, rt_raw.npz, fulltest.out, raw gz inputs, per-variant npz) are
gitignored and hashed in `artifacts/manifests/artifact_manifest.json`
(`python tools/build_artifact_manifest.py`); copy them with the git bundle to the
off-repo backup before the board is returned.
