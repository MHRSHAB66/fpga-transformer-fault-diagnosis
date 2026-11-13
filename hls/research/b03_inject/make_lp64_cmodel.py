"""LP64 copy of the B03 desktop C-model sources (B03 board-vs-C-model check).

MinGW g++ on Windows has a 32-bit `long`; Vitis HLS synthesizes `long` as
64 bits. In clean operation every value passed through `long` (e.g.
requant(long acc, ...)) stays far inside 32 bits (projection |acc + b| <= 719,316),
so both agree; a bit-31 flip of the projection accumulator (B03 site 2) can push
acc + b outside 32 bits, where only the Windows C-model wraps. This script writes
a copy of the B03 top, harness and headers with every standalone `long` turned
into `long long` (sources are only read):

    python hls/research/b03_inject/make_lp64_cmodel.py
    cd tmp/b03_lp64 && g++ -O2 -std=c++17 -Wno-unknown-pragmas -DWITH_PRED_CLASS -I . \
        -I ../../hls/f1_int10/gpp_csim/stubs transformer_top_b03_lp64.cpp batch_logits_b03.cpp \
        -o ../csim/batch_raw_b03_lp64.exe
    python -m tools.research.b03_campaign cmodel --out artifacts/research/B03_20261113 --lp64
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DST = REPO / "tmp/b03_lp64"
VITIS = REPO / "hls/f1_int10_seq256/vitis"
SOURCES = {
    "transformer_top_b03_lp64.cpp": REPO / "hls/research/b03_inject/transformer_top_seq256_raw1024_b03_argmax.cpp",
    "batch_logits_b03.cpp": REPO / "hls/research/b03_inject/batch_logits_b03.cpp",
    "weights_int10.h": VITIS / "weights_int10.h",
    "onchip_preprocess.h": VITIS / "onchip_preprocess.h",
    "argmax_first_max.h": VITIS / "argmax_first_max.h",
    "ap_types.h": VITIS / "ap_types.h",
}
LONG = re.compile(r"(?<!long )\blong\b(?! long)")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for name, src in SOURCES.items():
        text = src.read_text()
        (DST / name).write_text(LONG.sub("long long", text))
        print(name, "long -> long long:", len(LONG.findall(text)))


if __name__ == "__main__":
    main()
