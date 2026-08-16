# Day 13 Notes

## Main focus
Day 13 prepared the Stage 7 hardware path on the software side: a pure-NumPy
inference engine and a PYNQ lab demo.

## What was done
- Added a pure-NumPy inference engine (no torch) as a lightweight, portable
  reference for the eventual hardware path
- Added a PYNQ lab demo scaffold (Stage 7 prep)

## Current result
A dependency-light way to run the model's inference outside PyTorch, which is the
basis for the integer reference and the board-side driver built next.

## Final note
Day 13 was Stage 7 prep -- getting inference into a form that maps cleanly toward
FPGA, before committing to the full-integer arithmetic.
