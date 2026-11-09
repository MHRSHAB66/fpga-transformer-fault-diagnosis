#!/bin/bash
# Time-boxed PREEMPT_RT build attempt on the PC (WSL, no sudo). Evidence only:
# nothing is copied to the board. Source: linux-xlnx xilinx-v2018.3 + patch-4.14-rt1,
# config = the board's /proc/config.gz + PREEMPT_RT_FULL.
W=/home/mine_pc/rt_feasibility_20261024
REPO=/mnt/g/aut/project/fpga_fault_diagnosis
cd "$W/debs" || exit 1
apt-get download flex bison m4 libfl2 libfl-dev libssl-dev libelf-dev zlib1g-dev 2>&1 | tail -1
for d in flex_*.deb bison_*.deb m4_*.deb libfl*.deb libssl-dev_*.deb libelf-dev_*.deb zlib1g-dev_*.deb; do [ -f "$d" ] && dpkg -x "$d" ../xroot; done
cd "$W"
X="$W/xroot"
export PATH="$X/usr/bin:$PATH"
export LD_LIBRARY_PATH="$X/usr/lib/x86_64-linux-gnu:$X/usr/x86_64-linux-gnu/arm-linux-gnueabihf/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$X/usr/include" LIBRARY_PATH="$X/usr/lib/x86_64-linux-gnu"
export BISON_PKGDATADIR="$X/usr/share/bison" M4="$X/usr/bin/m4"
for t in flex bison arm-linux-gnueabihf-gcc-9 arm-linux-gnueabihf-as; do printf "%-28s %s\n" "$t" "$(command -v $t || echo MISSING)"; done
echo 'int main(){return 0;}' > t.c && arm-linux-gnueabihf-gcc-9 -c t.c -o t.o && file t.o
# fresh RT tree (the clone stays pristine)
rm -rf build-rt && cp -a linux-xlnx build-rt && cd build-rt
xzcat ../patch-4.14-rt1.patch.xz | patch -p1 -s -f && echo "RT patch applied"
cp "$REPO/artifacts/rollback/board_boot_backup_20261024/proc_config_4.14.0-xilinx.txt" .config
MK="make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- CC=arm-linux-gnueabihf-gcc-9 HOSTCFLAGS=-I$X/usr/include HOSTLDFLAGS=-L$X/usr/lib/x86_64-linux-gnu"
$MK olddefconfig > ../olddefconfig.log 2>&1
./scripts/config --disable PREEMPT --enable PREEMPT_RT_FULL --set-str LOCALVERSION "-rt-xilinx"
$MK olddefconfig >> ../olddefconfig.log 2>&1
grep -E "^CONFIG_PREEMPT|CONFIG_PREEMPT_RT|^CONFIG_HZ=|CONFIG_LOCALVERSION=" .config
START=$(date +%s)
timeout 5400 $MK -j"$(nproc)" zImage modules > ../build.log 2>&1
RC=$?
echo "build rc=$RC, seconds=$(( $(date +%s) - START )), nproc=$(nproc)"
grep -E " error|Error [0-9]" ../build.log | head -15
ls -la arch/arm/boot/zImage 2>/dev/null && sha256sum arch/arm/boot/zImage
strings arch/arm/boot/Image 2>/dev/null | grep -m1 "Linux version"
echo DONE
