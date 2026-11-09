#!/bin/bash
# usage: bench_run.sh <n> <mode> <tag> [lines]
DEP=/home/xilinx/deployments/seq256_z7020_pynq_dma16_py36_20261008_131446
B=/home/xilinx/bench_20261022
IN=/home/xilinx/fpga_fulltest_20261008/fpga_testset_in.i32.gz
OUT=$B/$3
if pgrep -f '[f]pga_live_runner.py|[f]pga_fulltest_bin_runner.py|[f]pga_stage_bench.py' >/dev/null; then echo "BUSY: another runner active"; exit 3; fi
mkdir -p $OUT
cd $DEP/driver
PYTHONPATH=$DEP/driver python3.6 -u $B/fpga_stage_bench.py $DEP/overlay/design_1_wrapper.bit $IN $1 $2 $OUT $4 &
P=$!
bash $B/temp_guard.sh $P $OUT/guard.log 75 1 &
G=$!
wait $P; RC=$?
wait $G; GRC=$?
echo "BENCH_RC=$RC GUARD_RC=$GRC"
