# Day 10 Notes

## Main focus
Day 10 was the frequency/spectrum study for Stage 2 -- deciding, with evidence,
how to treat the 8 channels and any resampling.

## What was done
- Ran a frequency investigation and spectrum analysis to inform the resampling
  decision (do not blindly downsample)
- Added an 8-channel frequency and discriminativity study (which channels carry
  fault-relevant information)
- Merged the Stage 2 frequency-analysis work

## Current result
A justified, documented basis for the channel/resampling choices instead of an
arbitrary one -- the fault-relevant frequency content and per-channel
discriminativity were measured. Artifacts under
`artifacts/results/frequency_investigation/` and
`artifacts/results/channel_analysis/`.

## Final note
Day 10 closed the "is our preprocessing frequency-justified?" question that
Stage 2 required before the pipeline could be called final.
