"""
Stage 6 — Quantization and fixed-point preparation.

This package contains software-only tooling that converts the trained
float32 Transformer into a hardware-oriented fixed-point representation and
measures the accuracy impact before any RTL/HLS work begins.

Nothing here changes the model architecture, the data split, or the trained
checkpoint. It only reads the frozen checkpoint and simulates reduced numeric
precision so the accuracy cost of int16/int8 can be reported (CLAUDE.md Stage 6).
"""
