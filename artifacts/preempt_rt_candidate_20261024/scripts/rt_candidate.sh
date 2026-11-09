#!/bin/bash
# Assemble the PREEMPT_RT candidate FIT in the same layout as the board's image.ub
# (uncompressed ARM kernel @0x80000 + the board's own extracted DTB). PC only.
W=/home/mine_pc/rt_feasibility_20261024
REPO=/mnt/g/aut/project/fpga_fault_diagnosis
OUT=$REPO/artifacts/preempt_rt_candidate_20261024
X=$W/xroot; export PATH="$X/usr/bin:$PATH" LD_LIBRARY_PATH="$X/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$X/usr/include" LIBRARY_PATH="$X/usr/lib/x86_64-linux-gnu"
export BISON_PKGDATADIR="$X/usr/share/bison" M4="$X/usr/bin/m4"
cd "$W"
cat > image_rt.its <<'ITS'
/dts-v1/;
/ {
    description = "PREEMPT_RT candidate (linux-xlnx xilinx-v2018.3 + patch-4.14-rt1), board DTB from original image.ub";
    #address-cells = <1>;
    images {
        kernel@0 {
            description = "Linux Kernel 4.14.0-rt1 PREEMPT_RT";
            data = /incbin/("build-rt/arch/arm/boot/zImage");
            type = "kernel"; arch = "arm"; os = "linux"; compression = "none";
            load = <0x80000>; entry = <0x80000>;
            hash@1 { algo = "sha1"; };
        };
        fdt@0 {
            description = "Flattened Device Tree blob (unchanged, from original image.ub)";
            data = /incbin/("orig_system.dtb");
            type = "flat_dt"; arch = "arm"; compression = "none";
            hash@1 { algo = "sha1"; };
        };
    };
    configurations {
        default = "conf@1";
        conf@1 { description = "Boot Linux kernel with FDT blob"; kernel = "kernel@0"; fdt = "fdt@0"; };
    };
};
ITS
mkimage -f image_rt.its image_rt.ub > mkimage.log 2>&1; echo "mkimage rc=$?"
dumpimage -l image_rt.ub | head -30
# modules for the new kernel version into a staging tree (never installed anywhere)
cd build-rt
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- CC=arm-linux-gnueabihf-gcc-9 \
     INSTALL_MOD_PATH="$W/modules_staging" modules_install > ../modules_install.log 2>&1; echo "modules_install rc=$?"
cd "$W"
KV=$(ls modules_staging/lib/modules)
echo "module dir: $KV, modules: $(find modules_staging -name '*.ko' | wc -l)"
rm -f modules_staging/lib/modules/$KV/build modules_staging/lib/modules/$KV/source
tar -C modules_staging -czf modules_rt.tar.gz lib
mkdir -p "$OUT"
cp image_rt.ub modules_rt.tar.gz image_rt.its "$OUT/"
cp build-rt/.config "$OUT/config_4.14.0-rt1-rt-xilinx.txt"
grep -E " warning:" build.log | wc -l > "$OUT/build_warning_count.txt"
grep -E "error" build.log | head -20 > "$OUT/build_errors_head.txt"
tail -30 build.log > "$OUT/build_log_tail.txt"
strings build-rt/arch/arm/boot/Image | grep -m1 "Linux version" > "$OUT/kernel_version_string.txt"
( cd "$OUT" && sha256sum image_rt.ub modules_rt.tar.gz image_rt.its config_4.14.0-rt1-rt-xilinx.txt > SHA256SUMS )
sha256sum patch-4.14-rt1.patch.xz orig_system.dtb > "$OUT/inputs_sha256.txt"
(cd linux-xlnx && echo "linux-xlnx tag xilinx-v2018.3 commit $(git rev-parse HEAD)") >> "$OUT/inputs_sha256.txt"
arm-linux-gnueabihf-gcc-9 --version | head -1 >> "$OUT/inputs_sha256.txt"
cat "$OUT/SHA256SUMS" "$OUT/kernel_version_string.txt" "$OUT/build_warning_count.txt"
echo DONE
