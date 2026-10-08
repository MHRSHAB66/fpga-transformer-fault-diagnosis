# Desktop g++ C-simulation of the AXI-Stream synthesizable top

Purpose: functionally C-simulate the **exact** synthesizable tops
(`../vitis/transformer_top.cpp` and `../vitis/transformer_top_opt.cpp`) on a
machine that does **not** have Vitis HLS installed, using only a desktop g++.
This lets us confirm bit-exact golden-vector behaviour and A/B-compare an
optimization here before the real Vitis HLS csim/csynth run on the laptop.

This is NOT synthesis and NOT a replacement for Vitis HLS. It uses the
non-`ap_int` branch of `ap_types.h` (wide native integers), which is the same
arithmetic the int10 reference and golden vectors were generated with. HLS
`#pragma` lines are ignored by g++ (`-Wno-unknown-pragmas`).

`stubs/` provides the two Vitis headers the tops include:
- `hls_stream.h`  — a `std::deque`-backed `hls::stream`.
- `ap_axi_sdata.h` — a plain `ap_axis<...>` struct with data/keep/strb/last.

## Build & run (from repo root)

```bash
# baseline top
g++ -O2 -std=c++17 -Wno-unknown-pragmas \
  -I hls/f1_int10/gpp_csim/stubs -I hls/f1_int10/vitis \
  hls/f1_int10/vitis/transformer_top.cpp hls/f1_int10/vitis/testbench_top.cpp \
  -o tb_top_baseline
./tb_top_baseline hls/f1_int10/golden        # -> AXI top exact vectors: 16/16

# optimized top
g++ -O2 -std=c++17 -Wno-unknown-pragmas \
  -I hls/f1_int10/gpp_csim/stubs -I hls/f1_int10/vitis \
  hls/f1_int10/vitis/transformer_top_opt.cpp hls/f1_int10/vitis/testbench_top.cpp \
  -o tb_top_opt
./tb_top_opt hls/f1_int10/golden             # -> AXI top exact vectors: 16/16
```

The two runs produce byte-identical output on all 16 golden vectors.

## Caveat

Windows g++ uses `long`=32-bit; the exported weights fit in 32-bit and every
accumulator is `long long`/`int64_t`, so results match the int64 reference. The
`ap_int<10>/ap_int<32>` widths used under real `__SYNTHESIS__` were already
validated at 16/16 on the laptop for the baseline; the optimization is a pure
loop reorder with unchanged accumulator widths, so the equivalence carries over
and will be re-confirmed by the laptop `csim_design` step.
