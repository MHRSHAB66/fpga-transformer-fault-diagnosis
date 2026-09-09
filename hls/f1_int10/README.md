# F_1layer @ int10 — HLS / portable-C++ inference core (Stage 7/8)

Full-integer inference core for the **FPGA-winner architecture** `F_1layer`
(1 encoder layer, d_model=32, 4 heads, ffn=64; 9,492 params) quantized to
**int10** (activations, weights and attention all 10-bit).

Why int10: a multi-bit sweep (`kaggle/sweep_multibit/`) showed int10 is the
smallest width that stays near-lossless; the **full-test int10 accuracy is
97.11%** vs float 97.55% (`kaggle/int10_shard/int10_fulltest_combined.json`),
with no class collapse. `ap_int<10>` also narrows storage and arithmetic relative
to int16. The final FPGA board/device is not selected yet; no fit claim is made.
int8 collapses the imbalance class, so it is not used.

This directory lives at `hls/f1_int10/` alongside the older `hls/2layer/`
day14 A_baseline int16 core. The two differ only in three constants
(`LAYERS`, `ACT_BITS`, `ATTN_BITS`) — the arithmetic is identical.

## Files
- `transformer.h` / `transformer.cpp` — bit-exact C++ transcription of
  `src/quantize/integer_model.py` at int10, 1 layer.
- `params_io.cpp` — loads every constant from `data/`.
- `testbench.cpp` — runs the core over `golden/` and checks it reproduces the
  Python reference (class + every logit).
- `verify_export.py` — **compiler-free** check: an independent NumPy
  re-implementation reading the same `data/` + `golden/` files. Does not import
  `integer_model`, so agreement is a genuine second-source proof.
- `data/` — exported weights, folded biases, requant multipliers, residual
  ratios, positional codes, LayerNorm affine, and LUTs (gelu/exp/rsqrt).
- `golden/` — int input codes + expected int logits/classes (16 windows).
- `model_manifest.json` — geometry, bit-widths, `input_scale`, golden summary.

## Regenerate the export
```bash
python -m src.quantize.export_hls_f_int10        # calibrates on 20 train files, writes data/ + golden/
```

## Verify without a compiler (works on this desktop)
```bash
python hls/f1_int10/verify_export.py             # PASS = 16/16 class, 16/16 bit-exact logits
```

## Functional C++ test (needs g++, e.g. on the laptop)
```bash
cd hls/f1_int10 && make test                     # builds ./tb, runs against data/ + golden/
```

## Vitis HLS synthesis (on the laptop with the board)
1. Add `transformer.cpp` + `transformer.h` to a Vitis HLS project, top =
   `transformer_infer`.
2. Swap the typedefs in `transformer.h`: `data_t -> ap_int<10>`,
   `acc_t -> ap_int<32>` (or wider per MAC-count), and represent the requant /
   affine `double` constants as `ap_fixed` or `M0 * 2^-shift` integer multipliers.
3. Add interface pragmas (AXI-Stream for the window in / logits out via one
   `axi_dma`, `s_axilite` for control) and pipeline the inner MAC loops.
4. Synthesize, capture LUT/FF/BRAM/DSP + latency, then drive it from
   `pynq/` (see `pynq/README.md`).

The `data/` constants are the golden source for both the C++ core and the RTL —
do not hand-edit them; regenerate from the Python reference.
