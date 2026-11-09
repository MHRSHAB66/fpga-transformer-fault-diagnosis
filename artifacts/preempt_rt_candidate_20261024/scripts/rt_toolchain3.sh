#!/bin/bash
# No-sudo cross toolchain from Ubuntu 20.04 .debs (apt-get download + dpkg -x).
W=/home/mine_pc/rt_feasibility_20261024
cd "$W" || exit 1
echo "== host tools for a kernel build"
for t in bc flex bison perl make gcc; do printf "%-6s %s\n" "$t" "$(command -v $t || echo MISSING)"; done
ls /usr/include/openssl/opensslv.h 2>/dev/null || echo "openssl headers MISSING (libssl-dev)"
ls /usr/include/gelf.h 2>/dev/null || echo "libelf headers MISSING (libelf-dev; needed only for some options)"
mkdir -p debs xroot && cd debs
apt-get download gcc-9-arm-linux-gnueabihf cpp-9-arm-linux-gnueabihf binutils-arm-linux-gnueabihf \
  libgcc-9-dev-armhf-cross gcc-9-arm-linux-gnueabihf-base gcc-9-cross-base 2>&1 | tail -3
ls *.deb 2>/dev/null
for d in *.deb; do dpkg -x "$d" ../xroot; done
cd ..
G=$(ls xroot/usr/bin/arm-linux-gnueabihf-gcc-9 2>/dev/null)
echo "gcc: $G"
[ -n "$G" ] && PATH="$W/xroot/usr/bin:$PATH" arm-linux-gnueabihf-gcc-9 --version 2>&1 | head -2
echo 'int main(){return 0;}' > t.c
PATH="$W/xroot/usr/bin:$PATH" arm-linux-gnueabihf-gcc-9 -c t.c -o t.o 2>&1 | head -5 && file t.o
echo DONE
