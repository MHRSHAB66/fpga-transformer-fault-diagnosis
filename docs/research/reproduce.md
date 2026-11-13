# Research program — reproduction guide (2026-11-13, corrected 2026-11-14)

All commands run from the repository root on the PC (Windows; Git Bash or PowerShell), Python 3.12
with `requirements.txt`, g++ 15 (`C:/Users/mine_pc/scoop/apps/gcc/current/bin/g++`), Vitis HLS and
Vivado 2020.2 at `G:/Xilinx`. Board: MYIR Z-Turn XC7Z020 at `root@192.168.2.99`, SSH key
`~/.ssh/id_zturn`, PYNQ 2.4 / Python 3.6. After every board reboot pin the Ethernet link:
`ssh -i ~/.ssh/id_zturn root@192.168.2.99 "ethtool -s eth0 advertise 0x008"` (at 1 Gb/s the link
dropped under load). Board jobs run under `tools/board/temp_guard.sh` (stop at 75 C); the board
tool refuses to start while another board job runs. Every board run writes `run_meta.json`
(deployment, bitstream sha256, command, git HEAD).

Status, outcomes and limitations: `docs/research/experiment_ledger.json`. Claims:
`docs/research/claims_evidence_equipment.md`. Audit resolution: `docs/research/audit_resolution_20261114.md`.
Variant contracts (numeric/input/reference/golden/evidence per research bundle):
`artifacts/research/contracts/`.

## Common C-models (g++)

```bash
G=C:/Users/mine_pc/scoop/apps/gcc/current/bin/g++
FLAGS="-O2 -std=c++17 -Wno-unknown-pragmas -DWITH_PRED_CLASS -I hls/f1_int10/gpp_csim/stubs"
$G $FLAGS -I hls/f1_int10_seq256/vitis hls/f1_int10_seq256/vitis/transformer_top_seq256_raw1024_dsp_margin_argmax.cpp \
   hls/f1_int10_seq256/gpp_csim/batch_logits.cpp -o tmp/csim/batch_raw_argmax.exe
```

## F00 — baseline identity and gates
```bash
python -m tools.research.freeze_baseline --out artifacts/research/F00_20261113
```
Gates (frozen before results): `artifacts/research/F00_20261113/preregistered_protocol.json`.

## A01/A02 — attention precision ablation (validation)
Needs the window cache `D:/fpga_cache/seq256_all` (`python -m tools.generalization_study cache`).
```bash
python -m tools.research.precision_ablation toy --out artifacts/research/A01_20261113
python -m tools.research.precision_ablation selfcheck --out artifacts/research/A02_20261113
OMP_NUM_THREADS=1 python -m tools.research.precision_ablation run --out artifacts/research/A02_20261113 --workers 8
python -m tools.research.precision_ablation summarize --out artifacts/research/A02_20261113
```

## A03/A04 — exact 9-bit softmax divider (operator)
```bash
$G -O2 -std=c++17 -I hls/f1_int10_seq256/vitis -I hls/research/a04_softmax_div \
   hls/research/a04_softmax_div/test_div_q9_exhaustive.cpp -o tmp/a04_exh.exe
tmp/a04_exh.exe
cd tmp && G:/Xilinx/Vitis_HLS/2020.2/bin/vitis_hls.bat -f ../hls/research/a04_softmax_div/run_a04_probe.tcl && cd ..
```

## A05 — integrated exact divider (bit-identical to canonical)
```bash
python hls/research/a04_softmax_div/make_divq9_top.py
powershell -File tools/build_raw_core.ps1 -Variant divq9
python -m tools.research.variant_contracts
python tools/package_seq256_pynq_deploy.py --pkg artifacts/research/A05_20261113/divq9_build \
   --bundle artifacts/deploy/research_divq9_z7020_20261113 --golden hls/f1_int10_seq256/golden_raw1024_argmax \
   --contract artifacts/research/contracts/research_divq9_z7020_20261113.json
python -m tools.raw_core_board_test deploy --bundle artifacts/deploy/research_divq9_z7020_20261113
python -m tools.raw_core_board_test golden --name research_divq9_z7020_20261113
python -m tools.raw_core_board_test fulltest --name research_divq9_z7020_20261113
python -m tools.raw_core_board_test rt --name research_divq9_z7020_20261113
```

## A06 — W10/A10 with unsigned 8-bit probabilities (own reference; not W8/A8)
```bash
python hls/research/a06_p8u/make_a06_top.py
$G $FLAGS -I hls/research/a06_p8u -I hls/f1_int10_seq256/vitis \
   hls/research/a06_p8u/transformer_top_seq256_raw1024_p8u_argmax.cpp hls/f1_int10_seq256/gpp_csim/batch_logits.cpp \
   -o tmp/csim/batch_raw_p8u.exe
python -m tools.check_variant_cmodel --ckpt artifacts/experiments/transformer_f1_seq256_pool4/kaggle_v3/seq256_pool4/scratch/best.pt \
   --scales artifacts/quantized/f_seq256_int10/integer_config.json --exe tmp/csim/batch_raw_p8u.exe \
   --n 600 --conditions clean snr0 --research-bits 10,10,8,1
python -m tools.make_variant_golden --ckpt artifacts/experiments/transformer_f1_seq256_pool4/kaggle_v3/seq256_pool4/scratch/best.pt \
   --scales artifacts/quantized/f_seq256_int10/integer_config.json \
   --dst artifacts/research/A06_20261113/golden_raw1024_p8u --research-bits 10,10,8,1
python -m tools.run_csim_fulltest --exe tmp/csim/batch_raw_p8u.exe \
   --input artifacts/results/fpga_rawinput_testset_20261109/fpga_testset_raw.i32.gz --n 95404 --shards 3 \
   --out artifacts/research/A06_20261113/csim_p8u_fulltest
powershell -File tools/build_raw_core.ps1 -Variant a06
python tools/package_seq256_pynq_deploy.py --pkg artifacts/research/A06_20261113/a06_build \
   --bundle artifacts/deploy/research_a06_z7020_20261113 --golden artifacts/research/A06_20261113/golden_raw1024_p8u \
   --contract artifacts/research/contracts/research_a06_z7020_20261113.json
python -m tools.raw_core_board_test deploy --bundle artifacts/deploy/research_a06_z7020_20261113
python -m tools.raw_core_board_test golden --name research_a06_z7020_20261113
python -m tools.raw_core_board_test fulltest --name research_a06_z7020_20261113 \
   --cmodel artifacts/research/A06_20261113/csim_p8u_fulltest/csim_logits.txt
python -m tools.raw_core_board_test rt --name research_a06_z7020_20261113
```

## B01/B02 — software injection and integrity checks
```bash
python -m tools.research.fault_injection campaign --out artifacts/research/B01_20261113
python -m tools.research.fault_injection weights --out artifacts/research/B01_20261113
python -m tools.research.fault_injection vote --out artifacts/research/B01_20261113
python -m tools.research.fault_injection campaign --seed 20261114 --out artifacts/research/B02_20261113/heldout_campaign
python hls/research/b02_integrity/make_b02_top.py
powershell -File tools/build_raw_core.ps1 -Variant b02
```

## B03 — logical injection overlay (timing needed a second implementation strategy)
```bash
python hls/research/b03_inject/make_b03_top.py
powershell -File tools/build_raw_core.ps1 -Variant b03          # default strategy: WNS -0.437 ns (kept)
G:/Xilinx/Vivado/2020.2/bin/vivado.bat -mode batch -notrace -source tools/vivado/rerun_impl_strategy.tcl \
   -tclargs tmp/vb03/vb03.xpr Performance_ExplorePostRoutePhysOpt
$G $FLAGS -I hls/research/b03_inject -I hls/f1_int10_seq256/vitis \
   hls/research/b03_inject/transformer_top_seq256_raw1024_b03_argmax.cpp hls/research/b03_inject/batch_logits_b03.cpp \
   -o tmp/csim/batch_raw_b03.exe
python hls/research/b03_inject/make_lp64_cmodel.py --variant b03
cd tmp/b03_lp64 && $G $FLAGS -I . transformer_top_b03_lp64.cpp batch_logits_b03.cpp -o ../csim/batch_raw_b03_lp64.exe && cd ../..
python -m tools.research.b03_campaign manifest --out artifacts/research/B03_20261113 --seed 20261115
python -m tools.research.b03_campaign cmodel --out artifacts/research/B03_20261113
python -m tools.research.b03_campaign cmodel --out artifacts/research/B03_20261113 --lp64
python -m tools.raw_core_board_test deploy --bundle artifacts/deploy/research_b03_z7020_20261113
python -m tools.raw_core_board_test golden --name research_b03_z7020_20261113
python -m tools.raw_core_board_test fulltest --name research_b03_z7020_20261113
python -m tools.raw_core_board_test rt --name research_b03_z7020_20261113
python -m tools.research.b03_campaign board --out artifacts/research/B03_20261113 --name research_b03_z7020_20261113
python -m tools.research.b03_campaign analyze --out artifacts/research/B03_20261113 --id B03
```

## B04 / B04i — single-pipeline checks + row checksum (`docs/research/b04_integrity_mechanism.md`)
```bash
python hls/research/b04_integrity_fast/make_b04_top.py
$G $FLAGS -I hls/research/b04_integrity_fast -I hls/f1_int10_seq256/vitis \
   hls/research/b04_integrity_fast/transformer_top_seq256_raw1024_b04_argmax.cpp \
   hls/research/b04_integrity_fast/batch_logits_b04.cpp -o tmp/csim/batch_raw_b04.exe
$G $FLAGS -DB04_INJECT -I hls/research/b04_integrity_fast -I hls/f1_int10_seq256/vitis \
   hls/research/b04_integrity_fast/transformer_top_seq256_raw1024_b04_argmax.cpp \
   hls/research/b04_integrity_fast/batch_logits_b04.cpp -o tmp/csim/batch_raw_b04i.exe
python hls/research/b03_inject/make_lp64_cmodel.py --variant b04i
cd tmp/b04i_lp64 && $G $FLAGS -DB04_INJECT -I . transformer_top_b04i_lp64.cpp batch_logits_b04.cpp -o ../csim/batch_raw_b04i_lp64.exe && cd ../..
python -m tools.run_csim_fulltest --exe tmp/csim/batch_raw_b04.exe \
   --input artifacts/results/fpga_rawinput_testset_20261109/fpga_testset_raw.i32.gz --n 95404 --shards 4 \
   --out artifacts/research/B04_20261114/csim_b04_fulltest
powershell -File tools/build_raw_core.ps1 -Variant b04
powershell -File tools/build_raw_core.ps1 -Variant b04i
python -m tools.research.b03_campaign manifest --out artifacts/research/B04_20261114/heldout --seed 20261116
python -m tools.research.b03_campaign manifest --out artifacts/research/B04_20261114/b03_manifest --seed 20261115
python -m tools.research.b03_campaign manifest --out artifacts/research/B04_20261114/recovery --seed 20261118 --recovery 32
python -m tools.research.b03_campaign cmodel --out artifacts/research/B04_20261114/heldout --exe tmp/csim/batch_raw_b04i.exe
python -m tools.research.b03_campaign cmodel --out artifacts/research/B04_20261114/heldout --lp64 --exe tmp/csim/batch_raw_b04i_lp64.exe
python -m tools.research.b03_campaign board --out artifacts/research/B04_20261114/heldout --name research_b04i_z7020_20261114
python -m tools.research.b03_campaign analyze --out artifacts/research/B04_20261114/heldout --id B04i
```
(The same `cmodel`, `board` and `analyze` commands are run for `b03_manifest` and `recovery`.)
B04 clean validation: `deploy`, `golden`, `fulltest` and `rt` with `--name research_b04_z7020_20261114`.

## T01 v2 — temporal policies (corrected; supersedes T01 short events and gate)
```bash
python -m tools.research.temporal_policies_v2 selfcheck --out artifacts/research/T01v2_20261114
python -m tools.research.temporal_policies_v2 run --out artifacts/research/T01v2_20261114
```

## Q01 v2 — input quality, calibrated confidence, margin (corrected; supersedes Q01 conf/gate)
```bash
python -m tools.export_fpga_testset_raw export --split val --out artifacts/results/fpga_rawinput_valset_20261113
python -m tools.research.q01_quality_v2 --out artifacts/research/Q01v2_20261114
```

## Q02 — quality / decision-status registers in hardware
```bash
python hls/research/q02_quality/make_q02_top.py
$G $FLAGS -I hls/research/q02_quality -I hls/f1_int10_seq256/vitis \
   hls/research/q02_quality/transformer_top_seq256_raw1024_q02_argmax.cpp hls/research/q02_quality/batch_logits_q02.cpp \
   -o tmp/csim/batch_raw_q02.exe
python -m tools.research.q02_check cmodel --out artifacts/research/Q02_20261114
python -m tools.research.q02_check replay --out artifacts/research/Q02_20261114
powershell -File tools/build_raw_core.ps1 -Variant q02
```
Board: see the `Q02` entry of the ledger for the exact replay/clean commands and run directories.

## S01 — interference envelope (board)
```bash
python -m tools.research.s01_interference --name seq256_vote_z7020_20261110 --out artifacts/research/S01_20261113
```

## Z01 — archive and off-repository backup
```bash
python tools/build_artifact_manifest.py
python -m tools.research.z01_archive index
python -m tools.research.z01_archive backup --dest E:/fpga_fault_diagnosis_backup/Z01_20261114 --same-disk
```
