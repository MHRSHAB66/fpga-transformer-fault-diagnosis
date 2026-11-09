#!/bin/bash
# PREEMPT_RT patch feasibility (PC/WSL only; nothing is built or installed, board untouched)
W=/home/mine_pc/rt_feasibility_20261024
mkdir -p "$W" && cd "$W" || exit 1
echo "workdir: $(pwd)"
if [ ! -f linux-xlnx/Makefile ]; then
  rm -rf linux-xlnx
  timeout 1800 git clone -q --depth 1 --branch xilinx-v2018.3 https://github.com/Xilinx/linux-xlnx.git linux-xlnx || { echo "CLONE FAILED"; exit 2; }
fi
[ -f patch-4.14-rt1.patch.xz ] || curl -s -O https://cdn.kernel.org/pub/linux/kernel/projects/rt/4.14/older/patch-4.14-rt1.patch.xz
sha256sum patch-4.14-rt1.patch.xz
cd linux-xlnx
echo "source commit: $(git log -1 --format='%H %cd')"
head -5 Makefile
xzcat ../patch-4.14-rt1.patch.xz | patch -p1 --dry-run -f > ../dryrun_rt1.txt 2>&1
echo "patch exit status: $?"
echo "files in patch: $(grep -c '^checking file' ../dryrun_rt1.txt)"
echo "FAILED hunk lines: $(grep -c 'FAILED' ../dryrun_rt1.txt)"
echo "hunks applied with offset/fuzz: $(grep -c 'succeeded at\|with fuzz' ../dryrun_rt1.txt)"
echo "files with failed hunks:"
awk '/^checking file/{f=$3} /FAILED/{print f}' ../dryrun_rt1.txt | sort -u
echo "DONE"
