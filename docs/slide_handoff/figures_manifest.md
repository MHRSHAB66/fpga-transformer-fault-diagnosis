# Figures Manifest

All image files included in this ZIP package.

---

## Training / Evaluation Plots (from `plots/`)

### `plots/transformer_loss_curve.png`
- **What it shows:** Training and validation loss per epoch over 16 epochs. Val loss reaches minimum 0.0353 at epoch 10, then early stopping triggers at epoch 16.
- **Recommended slide:** Slide 8 (Training Curves)
- **Recommended caption:** "Training and validation loss over 16 epochs. Best validation loss (0.0353) at epoch 10; early stopping at epoch 16."

---

### `plots/transformer_accuracy_curve.png`
- **What it shows:** Training and validation accuracy per epoch over 16 epochs. Both reach ~98% by epoch 10.
- **Recommended slide:** Slide 8 (Training Curves)
- **Recommended caption:** "Training and validation accuracy over 16 epochs. Validation accuracy at best checkpoint: 98.68%."

---

### `plots/transformer_val_confusion_matrix.png`
- **What it shows:** 4×4 confusion matrix on the validation set (47,580 windows, 4 classes).
- **Recommended slide:** Slide 10 (Per-Class Accuracy — left panel)
- **Recommended caption:** "Validation confusion matrix. Val accuracy: 98.68%. Classes: 0=Normal, 1=Imbalance, 2=Misalignment, 3=Bearing Fault."

---

### `plots/transformer_test_confusion_matrix.png`
- **What it shows:** 4×4 confusion matrix on the test set (95,404 windows, 4 classes, never seen during training).
- **Recommended slide:** Slide 9 (Test Results) — main figure
- **Recommended caption:** "Test confusion matrix (95,404 windows, 391 unseen files). Test accuracy: 98.22%."

---

### `plots/transformer_val_per_class_accuracy.png`
- **What it shows:** Bar chart of per-class accuracy on validation set. Normal:99.84%, Imbalance:99.50%, Misalignment:99.19%, Bearing:98.13%.
- **Recommended slide:** Slide 10 (Per-Class Accuracy — right panel or secondary figure)
- **Recommended caption:** "Per-class accuracy on validation set. Bearing fault is the hardest class despite being the largest."

---

### `plots/transformer_test_per_class_accuracy.png`
- **What it shows:** Bar chart of per-class accuracy on test set. Normal:97.66%, Imbalance:98.84%, Misalignment:98.76%, Bearing:97.80%.
- **Recommended slide:** Slide 10 (Per-Class Accuracy — right panel)
- **Recommended caption:** "Per-class accuracy on test set (95,404 unseen windows)."

---

## Diagrams (from `diagrams/`)

### `diagrams/pipeline_diagram.png`
- **What it shows:** End-to-end data pipeline: MAFAULDA CSV → file-level split → windowing → per-window Z-score → Transformer → fault class.
- **Recommended slide:** Slide 2 (Project Goal) or Slide 4 (Preprocessing)
- **Recommended caption:** "Final data pipeline. Per-window Z-score normalization applied after windowing. All preprocessing flags OFF."

---

### `diagrams/architecture_diagram.png`
- **What it shows:** Transformer model structure: 8-channel input → Linear projection (8→32) → Positional encoding → 2× Encoder blocks → Mean pooling → Classifier head → 4 output classes.
- **Recommended slide:** Slide 7 (Transformer Architecture) — use as main figure or supplement
- **Recommended caption:** "Lightweight Transformer architecture. Input: [batch, 1024, 8]. Output: 4 fault class logits. Checkpoint: 380 KB."

---

### `diagrams/stage_status_diagram.png`
- **What it shows:** Project stage roadmap (Stages 0–12) with color coding: green = complete (0–5), yellow = next (6), grey = future (7–12).
- **Recommended slide:** Slide 11 (Completed Work) or Slide 12 (Next Steps)
- **Recommended caption:** "Project stage status. Stages 0–5 complete. Stage 6 (Quantization) is next."
