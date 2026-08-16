# Day 14 Notes

## Main focus
Day 14 built the full-integer inference reference and the bit-exact HLS /
portable-C++ core (Stage 6f + Stage 7/8) for the 2-layer baseline at int16.

## What was done
- Added a full-integer Transformer inference reference (Stage 6f): integer
  softmax, LayerNorm and GELU (previously float), input/output integer-only
- Added the HLS / portable-C++ integer Transformer core and the data export
  (Stage 7/8): weights, requant multipliers, LUTs, positional codes, golden
  vectors under `hls/`
- Finalized the integer calibration and refreshed the HLS export
- Verified the HLS core with a real g++ build; fixed an indentation warning

## Current result
A verified full-integer path for the A_baseline (2-layer) model at int16:
integer accuracy ~98.19% vs float ~98.29%, and a C++ core that reproduces the
Python reference bit-for-bit (checked with g++ and a compiler-free NumPy oracle).

## Final note
Day 14 established the integer + HLS methodology on the 2-layer int16 model.
Day 15 then reused this machinery for the smaller F_1layer @ int10 core actually
targeted at the small board.
