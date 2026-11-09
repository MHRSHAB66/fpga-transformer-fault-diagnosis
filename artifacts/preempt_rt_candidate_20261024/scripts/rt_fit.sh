#!/bin/bash
# Inspect the board's FIT image and assemble an RT candidate the same way (PC only).
W=/home/mine_pc/rt_feasibility_20261024
REPO=/mnt/g/aut/project/fpga_fault_diagnosis
OUT=$REPO/artifacts/preempt_rt_candidate_20261024
mkdir -p "$OUT"
cd "$W/debs" && apt-get download u-boot-tools device-tree-compiler libfdt1 2>&1 | tail -1
for d in u-boot-tools_*.deb device-tree-compiler_*.deb libfdt1_*.deb; do dpkg -x "$d" ../xroot; done
X=$W/xroot; export PATH="$X/usr/bin:$PATH" LD_LIBRARY_PATH="$X/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
cd "$W"
cp "$REPO/artifacts/rollback/board_boot_backup_20261024/image.ub" orig_image.ub
echo "== original image.ub (FIT) structure"
dumpimage -l orig_image.ub | tee "$OUT/orig_image_ub_dumpimage.txt"
dtc -I dtb -O dts orig_image.ub 2>/dev/null | grep -E "description|type|arch|os|compression|load|entry|kernel|fdt|ramdisk|default" | head -40 > "$OUT/orig_image_ub_its_fields.txt"
# extract the device tree image (index found from dumpimage listing)
for i in 0 1 2 3; do
  T=$(dumpimage -l orig_image.ub | awk -v n="Image $i" '$0 ~ n {f=1} f && /Type:/ {print $2; exit}')
  [ "$T" = "Flat" ] && dumpimage -T flat_dt -p $i -o orig_system.dtb orig_image.ub && echo "dtb = image $i"
done
ls -la orig_system.dtb && sha256sum orig_system.dtb
dtc -I dtb -O dts orig_system.dtb 2>/dev/null | grep -m3 -E "model|compatible"
echo DONE
