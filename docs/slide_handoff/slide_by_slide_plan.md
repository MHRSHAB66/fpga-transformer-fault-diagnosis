# Slide-by-Slide Update Plan

> Legend: **KEEP** = no changes needed | **UPDATE** = modify content | **REPLACE FIGURE** = swap the image

---

## Slide 1 — Title Slide
**Action: UPDATE subtitle**

- Subtitle line: change `Progress Presentation — Software Phase` → `Progress Presentation — Software Phase Complete`
- Date: keep `June 2026`

Figure: none
Speaker note: "Software/model phase complete. Today presenting final results before moving to FPGA."

---

## Slide 2 — Project Goal & System Pipeline
**Action: ADD one bullet**

All existing content is correct. Add at the end of the bullet list:

> `Software phase complete: 98.22% test accuracy on 95,404 unseen windows`

Figure: replace or supplement with `diagrams/pipeline_diagram.png`
Speaker note: "All software stages are done. Next is quantization and FPGA mapping."

---

## Slide 3 — Dataset: MAFAULDA
**Action: KEEP — all numbers match the final run**

Split numbers (already correct):
- Train: 1,365 files / 333,060 windows
- Val: 195 files / 47,580 windows
- Test: 391 files / 95,404 windows

Optional addition: `Class imbalance ratio: 22x → weighted cross-entropy applied (normal class weight: 10.04×)`

Figure: none needed
Speaker note: "1,951 files, file-level split, no window leakage."

---

## Slide 4 — Stage 2: Preprocessing & Windowing Pipeline
**Action: UPDATE three values**

1. Change `batch_size = 8` → `batch_size = 64`
2. Change `Cold build: ~1,877 s` → `Cold build: ~1,525 s (~25 min)`
3. Change `Per-channel Z-score norm μ=0, σ=1 per file per channel`
   → `Per-window Z-score norm μ=0, σ=1 per window per channel`

All other content (window size 1024, stride 1024, tensor shape [B,1024,8], flags OFF) is correct.

Figure: `diagrams/pipeline_diagram.png` can supplement this slide
Speaker note: "Per-window normalization was the final choice after comparing per-file and global-train alternatives."

---

## Slide 5 — Stage 2: Frequency Analysis & Resampling Decision
**Action: KEEP**

Content is accurate. The 22 kHz finding and decision to keep 50 kHz are confirmed final.

Figure: keep existing frequency/PSD plots already on the slide
Speaker note: "Resampling was intentionally deferred — the 22 kHz resonance is discriminative."

---

## Slide 6 — Stage 2: Channel Discriminativity Study
**Action: KEEP**

All 8 channels used in the final trained model (CHANNEL_SELECTION='all').

Speaker note: "All 8 channels used in the final trained model."

---

## Slide 7 — Stage 4: Lightweight Transformer Architecture
**Action: UPDATE hyperparameters line + ADD checkpoint note**

In the bottom hyperparameters row, fix:
- Change: `epochs=6`
  → `max_epochs=30, early_stopping_patience=6`

Add a note below the table:
> `Best checkpoint: epoch 10 of 16 (early stopped) | size: 380 KB | peak GPU: 0.28 GiB`

All architecture details (d_model=32, nhead=4, layers=2, ffn_dim=64, dropout=0.1, LR=1e-3, Adam, weighted cross-entropy) remain correct.

Figure: `diagrams/architecture_diagram.png`
Speaker note: "This lightweight architecture uses only 0.28 GiB GPU memory — intentionally small for FPGA mapping."

---

## Slide 8 — Stage 4: Training Curves (CRITICAL UPDATE)
**Action: REPLACE FIGURES + UPDATE ALL TEXT**

This slide currently shows OLD 6-epoch results. Replace completely.

**Replace figures:**
- Loss curve: `plots/transformer_loss_curve.png`
- Accuracy curve: `plots/transformer_accuracy_curve.png`

**Replace title:**
- `Stage 4/5: Training Curves (16 Epochs, Early Stopped at Epoch 16)`

**Replace checkpoint line:**
- OLD: `Best checkpoint: epoch 6 (val loss 0.0610, val acc 98.21%)`
- NEW: `Best checkpoint: epoch 10 (val loss 0.0353, val acc 98.68%) | early stopped at epoch 16`

**Replace training time line:**
- OLD: `Total training time: ~2.98 hours on CPU (no GPU acceleration used).`
- NEW: `Total training time: ~4.2 hours on Tesla T4 GPU (~922 s/epoch)`

**Keep:** `Class-weighted loss applied to handle 22× class imbalance.`

**Updated bullets:**
- Training ran for 16 epochs; best validation loss reached at epoch 10 (early stopping patience = 6)
- Validation loss improved from 0.2946 (epoch 1) → 0.0353 (epoch 10), then plateaued
- No overfitting: train and val accuracy converge smoothly to ~98%
- Full MAFAULDA dataset: 333,060 training windows, 47,580 validation windows

Speaker note: "The full MAFAULDA dataset (333k training windows) was used for the first time. Early stopping at epoch 16 confirms convergence."

---

## Slide 9 — Stage 4/5: Test Set — Confusion Matrix & Metrics (CRITICAL UPDATE)
**Action: REPLACE CONFUSION MATRIX + UPDATE METRICS TABLE**

**Replace figure:** `plots/transformer_test_confusion_matrix.png`

**Replace metrics table:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.910 | 0.977 | 0.942 |
| Imbalance | 0.958 | 0.988 | 0.973 |
| Misalignment | 0.979 | 0.988 | 0.983 |
| Bearing Fault | 0.995 | 0.978 | 0.987 |
| Macro avg | 0.960 | 0.983 | 0.971 |

**Replace summary line:**
- OLD: `Test accuracy: 98.21% | Test weighted F1: 0.982`
- NEW: `Test accuracy: 98.22% | Macro F1: 0.971 | Macro F2: 0.978 | Fault recall: 99.75%`

**Keep bullets:**
- "Test set: 391 files (20% of dataset), never seen during training." (correct)
- Update: "Normal class has lowest F1 (0.942) — smallest class (22× imbalance)"
- Keep: "No test-set tuning performed." (correct)

**Add safety bullet:**
- `Fault recall 99.75%: only 236 of 92,964 fault windows missed | False alarms: 57/2,440 normal (2.34%)`

Speaker note: "Macro F2 (0.978) is reported alongside F1 because fault recall matters more than precision in safety applications."

---

## Slide 10 — Stage 5: Per-Class Accuracy (CRITICAL UPDATE)
**Action: REPLACE ALL FIGURES + UPDATE NUMBERS**

**Replace figures:**
- Left panel: `plots/transformer_val_confusion_matrix.png`
- Right panel: `plots/transformer_test_per_class_accuracy.png`

**Replace summary numbers:**
- OLD: `Val accuracy: 98.21% | Val macro F1: 0.977`
  NEW: `Val accuracy: 98.68% (best epoch) | Val macro F2: 0.987`
- OLD: `Test accuracy: 98.21% | Test macro F1: 0.975`
  NEW: `Test accuracy: 98.22% | Test macro F2: 0.978`

**Keep:** "Consistent train/val/test performance → no overfitting, no data leakage." (correct)

Speaker note: "Val and test metrics are within 0.5% — confirms no leakage and stable generalization."

---

## Slide 11 — Completed Work: Stages 0 → 5
**Action: UPDATE two lines**

**Update Stage 4 line:**
- OLD: `Lightweight Transformer (d_model=32, 2 layers, 4 heads) trained to 98.21% accuracy`
- NEW: `Lightweight Transformer (d_model=32, 2 layers, 4 heads) — 98.22% test accuracy, 380 KB checkpoint`

**Update Stage 5 line:**
- OLD: `Clean evaluation script; metrics JSON; confusion matrix; per-class F1 exported`
- NEW: `Full evaluation: 98.22% test acc, macro F2=0.978, fault recall=99.75%; all artifacts saved`

**Add one detail line** (optional):
> `Final run: 1,951 files, 333k windows, Tesla T4, early stopped epoch 16 (best: epoch 10)`

Figure: `diagrams/stage_status_diagram.png`
Speaker note: "Stages 0–5 are complete and reproducible. All artifacts saved."

---

## Slide 12 — Next Step: Stage 6 — Quantization
**Action: KEEP + ADD one line**

Content (tasks 6a–6e) is already accurate. Add at the bottom:

> `Starting point: transformer_best.pt (380 KB, float32, test_acc=98.22%) → target: int8/int16`

Speaker note: "Stage 6 starts immediately. The small checkpoint (380 KB) is promising for quantization."

---

## Optional NEW Slide (add after Slide 10)

**Title:** `Stage 5: Safety Metrics — Fault vs. Normal`

| Metric | Value |
|---|---|
| Fault recall | 99.75% |
| False alarm rate | 2.34% |
| Captured faults | 92,728 / 92,964 |
| Missed faults | 236 |
| False alarms | 57 / 2,440 |

Bullets:
- A missed fault = undetected machine damage → fault recall is the primary safety metric
- 99.75% fault recall on 95,404 unseen test windows
- These results come from the final full-dataset run (not the smoke test)

Figure: none
Speaker note: "For a safety-critical application, missing a fault is worse than a false alarm."
