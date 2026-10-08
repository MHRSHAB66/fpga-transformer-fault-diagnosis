# Experiment design — transformer_f1_seq256_pool4

Isolated SEQ=256 experiment to reach real-time FPGA inference by aggregating the
physical 1024x8 window (50 kHz, 20.48 ms, all 8 MAFAULDA channels PRESERVED) into
256 transformer tokens via non-overlapping 4:1 mean. The approved SEQ=1024 model,
prefix_opt2, checkpoints, int10 exports, golden vectors, and reports are untouched.

## Aggregation (exact definitions)
- Order: apply the approved per-window per-channel z-score normalization FIRST, then
  aggregate 4:1. token[t,c] = mean(x[4t..4t+3, c]).
- Float model: exact arithmetic mean (float64) -> src.data.token_aggregation.aggregate_mean_float.
- Integer/HLS: signed ROUND-HALF-AWAY-FROM-ZERO div-by-4:
    s = a+b+c+d ; q = ((|s| + 2) >> 2) with sign reapplied
  -> src.data.token_aggregation.aggregate_round_div (bit-exact with the HLS block).
  aggregate_round_div function-source SHA-256:
    901b038de25df3707febb31e0e676e7295251c621514434de659f0e76ba894e8
- Input to transformer becomes [B, 256, 8]. factor is configurable; factor=1 is a
  verified no-op so the SEQ=1024 path is unchanged (test_factor_one_is_noop...).

## Reference model (FPGA-selected) — F_1layer, NOT the 2-layer baseline
The model carried to hardware is **F_1layer** (the single-encoder-layer Transformer that
was int10-quantized and synthesized as prefix_opt2). The SEQ=256 experiment retrains this
SAME F_1layer architecture. The two-layer `transformer_baseline` result (test_acc 0.9821)
is a DIFFERENT model and is NOT the reference here — do not compare against it.

## Unchanged model components (F_1layer)
d_model=32, nhead=4, dim_feedforward=64, 1 encoder layer, 4 classes, mean pooling,
fixed sinusoidal positional encoding. All learned params are sequence-independent, so the
only sequence-length-dependent state is the PE buffer (`pos_enc.pe`, shape (1, L, 32)),
which is regenerated for L=256 on warm-start.

## Changed configuration values (seq256 only)
seq_len/INPUT_LENGTH consumed by the model: 1024 -> 256 (via aggregation factor 4).
WINDOW_SIZE/stride/normalization/labels/split/seed: UNCHANGED (raw window stays 1024).

## Preserved invariants
8 channels; four-class map; file-level train/val/test split; RANDOM_SEED=42; window
labeling; 1024-sample physical observation; test-set isolation.

## Reference metrics to beat/compare (F_1layer @ SEQ=1024 — verified)
These are the ONLY correct references for SEQ=256 comparison and every acceptance calc.

Float F_1layer (checkpoint kaggle/f1_ckpt_dataset/best.pt):
- test_acc = 0.9754517630
- macro-F1 = 0.9507051425
- per-class test recall:
    normal         = 0.9897540984
    imbalance      = 0.9807315880
    misalignment   = 0.9770657394
    bearing_fault  = 0.9723980175

Int10 F_1layer (same architecture, int10 quantized):
- test_acc = 0.9711332858
- per-class test recall:
    normal         = 0.9934426230
    imbalance      = 0.9846464399
    misalignment   = 0.9712286802
    bearing_fault  = 0.9658406405

Acceptance (referenced to F_1layer, NOT the 2-layer 0.9821 result):
- seq256 float: test acc >= 0.96, macro-F1 >= 0.95, and NO class recall drop > 3 pp vs
  the float F_1layer per-class recalls above.
- seq256 int10: test acc >= 0.96, macro-F1 >= 0.95, and <= 1 pp drop from seq256 float.

## New artifact paths (seq256, isolated)
- artifacts/experiments/transformer_f1_seq256_pool4/  (this design, manifest)
- artifacts/checkpoints/transformer_f1_seq256_pool4_best.pt   (from Kaggle)
- artifacts/results/transformer_f1_seq256_pool4_*.json        (metrics/history)
- artifacts/quantized/seq256_int10/                            (new int10 export)
- artifacts/golden/seq256_int10/                               (new golden vectors)
- hls/f1_int10/vitis/transformer_top_rt_seq256.cpp            (new HLS top)

## Kaggle workflow (auth confirmed: mehrdadsheikhabbasi)
Mirror kaggle/final_train: private GPU(T4) script kernel, internet off, datasets
= mehrdadsheikhabbasi/fpga-fault-src (code, to be updated with token_aggregation +
seq256 kernel) + vuxuancu/mafaulda-full (raw). Two candidates:

(A) WARM-START from the F_1layer float checkpoint `kaggle/f1_ckpt_dataset/best.pt`
    (NOT the 2-layer baseline checkpoint). Warm-start contract:
    - build the F_1layer model at seq_len=256 (freshly generated sinusoidal PE buffer);
    - load state_dict with strict=False and REPORT every missing and unexpected key;
    - the ONLY permitted difference is the sequence-length-dependent PE buffer
      `pos_enc.pe` (checkpoint (1,1024,32) vs model (1,256,32)); exclude it from the
      load (or overwrite it with the freshly generated 256-length buffer) so the shape
      mismatch cannot corrupt the load;
    - ANY other missing/unexpected/shape-mismatched key aborts the warm-start (it would
      mean the architecture drifted from F_1layer).
(B) FROM-SCRATCH (seed 42), same F_1layer architecture at seq_len=256.

Confirmed warm-start mechanism (matches src/evaluate/evaluate_f1layer.py):
best.pt was trained with TuningTransformer, which is the SAME architecture as
TransformerBaseline with shorter attribute names. Load it by (1) building
TransformerBaseline(d_model=32, nhead=4, num_layers=1, dim_feedforward=64,
dropout=0.1, input_dim=8, num_classes=4, seq_len=256) so its positional buffer is
freshly generated at length 256; (2) remapping checkpoint keys with the existing pure
name-adapter `_remap_key` (pos_enc.->positional_encoding., input_proj.->input_projection.,
encoder.N.attn/attn_norm/ffn/ffn_norm -> encoder_layers.N.self_attention/attention_norm/
feedforward/feedforward_norm) — a rename only, no numeric change; (3) DROPPING the remapped
`positional_encoding.pe` (checkpoint (1,1024,32) vs model (1,256,32)) so the fresh 256-length
buffer is kept; (4) load_state_dict(strict=False) and asserting the ONLY unexpected/missing
entry is that PE buffer — any other mismatch aborts warm-start.

Select by VALIDATION macro-F1 only; evaluate the test set once for the selected candidate.

## Status
Phase 1 aggregation implemented + tested (10/10 token-aggregation tests incl. dtype +
model-forward; full suite 40/40). aggregate_mean_float returns float32 by default (float64
accumulation, cast on output) so the DataLoader/model see float32; factor=1 preserves both
values and dtype. Training, quantization, golden vectors, HLS, and synthesis are MEASURED
steps that require the Kaggle GPU run + downstream pipeline (not yet executed). No result
is reported as measured until produced by a completed run.
