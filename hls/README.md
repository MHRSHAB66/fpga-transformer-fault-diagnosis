# HLS / portable-C++ integer Transformer core

Bit-exact C++ transcription of the verified full-integer Python reference
(`src/quantize/integer_model.py`). It runs anywhere with a C++17 compiler and is
the starting point for a Vitis HLS build.

## What is here

| File | Role |
|------|------|
| `transformer.h` | model geometry, fixed-point constants, `Params` struct, API |
| `transformer.cpp` | the integer core (input proj, attention, FFN, LayerNorm, classifier) |
| `params_io.cpp` | loads the exported constants from `data/` |
| `testbench.cpp` | runs the golden vectors and checks bit-exact match |
| `Makefile` | build/run with plain `g++` |
| `data/` | exported weights, scales, requant multipliers, LUTs (generated) |
| `golden/` | integer golden vectors (generated) |
| `model_manifest.json` | shapes, bit-widths, golden self-check (generated) |

`data/`, `golden/` and `model_manifest.json` are produced by:

```bash
python -m src.quantize.export_hls
```

## Functional verification (no Vitis required)

```bash
cd hls
make test
```

This compiles with `g++` and runs the core over the 16 golden windows, checking
that every predicted class **and every logit** matches the Python reference
exactly.

## Numeric scheme

Every tensor (input, all intermediates, output) is an integer. Only per-operation
*constants* are real numbers:

- **requant multipliers** `M = s_in * s_w / s_out` — applied as `round(acc * M)`;
- **LayerNorm** folded per-channel affine and integer eps;
- **LUT contents** for `exp` (softmax), `erf`-GELU, and reciprocal-sqrt.

In this portable reference those constants are `double`, which is bit-exact with
the Python reference. For a specific board they are converted to fixed-point
(`M0 * 2^-shift`, `ap_fixed` affine, BRAM LUTs) — the deferred optimisation step.

## Path to a Vitis HLS build

1. Replace the typedefs in `transformer.h` (`int64_t` → `ap_int<16>` /
   `ap_int<48>` accumulators; requant constants → `ap_fixed`).
2. Wrap `transformer_infer` with an AXI-Stream / AXI-Lite interface and add
   `#pragma HLS` directives (pipeline, array_partition, BRAM for LUTs/weights).
3. Re-run C-simulation against the same golden vectors, then synthesise and
   check resource/latency for the target device.
