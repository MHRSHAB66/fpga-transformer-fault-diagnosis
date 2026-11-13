#!/usr/bin/env python3
"""B03 board runner (Python 3.6, deployed driver): logical fault injection on the
research B03 overlay. For every manifest row the injection registers are written
while the core is idle, the core is started (ap_start, no auto-restart, so the
registers belong to exactly this window), the raw window is streamed through the
DMA and the class register, logits and integrity_status are read back.

Usage (inside <DEP>/driver with PYTHONPATH there):
  python3.6 -u b03_inject_runner.py <bit> <raw.i32.gz> <manifest.txt> <out.txt>
manifest: one line per event 'window_index site word bit' (site 0 = no injection).
Output: 'R row=<k> idx=<window> site=<s> word=<w> bit=<b> cls=<c> logits=<l0,..> integrity=<v>'.
"""
import gzip
import sys
import time

import numpy as np
from fault_classifier import HardwareBackend, _reg_offset, IP_NAME

AP_CTRL = 0x00


def main():
    bit, gz, manifest, out_path = sys.argv[1:5]
    rows = [tuple(int(v) for v in l.split()) for l in open(manifest) if l.strip()]
    need = sorted(set(r[0] for r in rows))
    want, windows, k = set(need), {}, 0
    with gzip.open(gz, "rb") as f:
        while len(windows) < len(want):
            b = f.read(32768)
            if not b:
                break
            if k in want:
                windows[k] = np.frombuffer(b, dtype="<i4").reshape(1024, 8).copy()
            k += 1
    be = HardwareBackend(bit, auto_restart=False, wait_timeout=10.0)
    regs = dict(be.overlay.ip_dict[IP_NAME].get("registers", {}) or {})
    off = {n: _reg_offset(regs, n) for n in ("inj_site", "inj_word", "inj_bit")}
    if None in off.values() or not be.has_integrity:
        raise SystemExit("overlay has no injection/integrity registers: %s" % off)
    # the constructor issued one ap_start with the registers at 0: consume it with a clean window
    be.infer_codes(windows[need[0]])

    def start(site, word, b):
        t0 = time.time()
        while not (int(be.ip.read(AP_CTRL)) & 0x4):          # wait for ap_idle
            if time.time() - t0 > 2.0:
                raise RuntimeError("core not idle")
        be.ip.write(off["inj_site"], site)
        be.ip.write(off["inj_word"], word)
        be.ip.write(off["inj_bit"], b)
        be.ip.write(AP_CTRL, 0x1)                              # ap_start for this window only

    t_start = time.time()
    with open(out_path, "w") as out:
        for n, (idx, site, word, b) in enumerate(rows):
            start(site, word, b)
            logits = be.infer_codes(windows[idx])
            cls = be.read_hw_class()
            integ = be.read_integrity()
            out.write("R row=%d idx=%d site=%d word=%d bit=%d cls=%d logits=%s integrity=%d\n" % (
                n, idx, site, word, b, cls, ",".join(str(int(v)) for v in logits), integ))
    print("SUMMARY rows=%d seconds=%.1f" % (len(rows), time.time() - t_start))


if __name__ == "__main__":
    main()
