#!/bin/bash
# CPU baseline on the Zynq ARM Cortex-A9 (runs ON the board, from its own dir).
# Builds the deployed INT10 top as plain C++ twice (-O2, and -O3 with NEON
# tuning), then times it on the RAM demo set: one process pinned to CPU 1
# (latency) and two processes pinned to CPU 0 and CPU 1 (throughput). Every
# workload runs under the independent temperature guard (stop at 75 C).
# Usage: run_cpu_baseline.sh <n_windows> <repeats>
set -u
cd "$(dirname "$0")"
N="${1:-256}"; R="${2:-1}"
SRC=src/transformer_top_seq256_ingress1024_dsp_margin_argmax.cpp
mkdir -p out
uname -a > out/system.txt; g++ --version | head -1 >> out/system.txt
grep -m1 "model name" /proc/cpuinfo >> out/system.txt; nproc >> out/system.txt

guarded() {   # guarded <logname> <command...>
  local log="out/guard_$1.log"; shift
  "$@" & local pid=$!
  ./temp_guard.sh "$pid" "$log" 75 1 &
  wait "$pid"; local rc=$?
  wait
  return $rc
}

for cfg in "O2|-O2" "O3neon|-O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard"; do
  name="${cfg%%|*}"; flags="${cfg#*|}"
  echo "build $name: g++ $flags" >> out/system.txt
  g++ $flags -std=c++17 -Wno-unknown-pragmas -DWITH_PRED_CLASS -I src -I stubs \
      "$SRC" src/cpu_baseline_bench.cpp -o "bench_$name" || exit 3

  guarded "${name}_single" bash -c \
    "taskset -c 1 ./bench_$name set.i32 $N $R > out/${name}_single.txt 2> out/${name}_single.sum"

  t0=$(date +%s.%N)
  guarded "${name}_dual" bash -c \
    "taskset -c 0 ./bench_$name set.i32 $N $R > out/${name}_dual0.txt 2> out/${name}_dual0.sum &
     taskset -c 1 ./bench_$name set.i32 $N $R > out/${name}_dual1.txt 2> out/${name}_dual1.sum &
     wait"
  t1=$(date +%s.%N)
  echo "$name dual_wall_s $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')" >> out/dual_wall.txt
done
echo DONE > out/status.txt
