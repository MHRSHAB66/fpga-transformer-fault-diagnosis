"""
Central configuration — the single source of truth for the whole project.

Every runnable stage (data loading, windowing, training, evaluation,
quantization, golden-vector export, and the FPGA-facing constants) reads its
settings from here. Keep constants in this file rather than scattering them
across scripts, and change values here in one place so all stages stay
consistent.

The project root and dataset root are resolved from the environment so no
user-specific absolute paths are hard-coded.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

project_root_env = os.getenv("PROJECT_ROOT")
if not project_root_env:
    raise ValueError("PROJECT_ROOT is not set in .env")

PROJECT_ROOT = Path(project_root_env).resolve()

# =========================
# Paths
# =========================
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mafaulda"      # raw MAFAULDA CSVs
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"       # derived artifacts
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"                 # per-run outputs
DEFAULT_RUN_NAME = "transformer_final_t4"

# =========================
# Reproducibility
# =========================
# Single seed shared by the file-level split and all torch/numpy randomness so
# a clean run reproduces the reported metrics.
RANDOM_SEED = 42

# =========================
# Labels / classes
# =========================
# Four grouped fault classes. This mapping is the taxonomy contract: dataset
# preparation, model output dimension, reports, and hardware constants must all
# agree with it. Do not reorder or renumber without updating every dependent
# stage.
LABEL_MAP = {
    "normal": 0,
    "imbalance": 1,
    "misalignment": 2,
    "bearing_fault": 3,
}
NUM_CLASSES = len(LABEL_MAP)

# =========================
# Input data shape
# =========================
# Verified MAFAULDA column order. Channel roles:
#   col 0     tachometer                     (rotation-speed pulse reference)
#   col 1-3   underhang bearing (b1)         (axial / radial / tangential)
#   col 4-6   overhang bearing  (b2)         (axial / radial / tangential)
#   col 7     microphone                     (acoustic emission)
EXPECTED_COLUMNS = [
    "tachometer",
    "acc_axial_b1",
    "acc_radial_b1",
    "acc_tangential_b1",
    "acc_axial_b2",
    "acc_radial_b2",
    "acc_tangential_b2",
    "microphone",
]
EXPECTED_NUM_COLUMNS = len(EXPECTED_COLUMNS)

# All 8 columns are fed to the model (CHANNEL_SELECTION = "all").
INPUT_CHANNELS = EXPECTED_NUM_COLUMNS

# Fixed-length windowing. Stride == size means non-overlapping windows, and
# files shorter than one full window are dropped.
WINDOW_SIZE = 1024
WINDOW_STRIDE = 1024
MIN_ROWS_PER_FILE = WINDOW_SIZE

# Sequence length seen by the model: one token per time step in a window.
# Consumed by the Transformer as seq_len and by the reproducibility snapshot.
INPUT_LENGTH = WINDOW_SIZE

# =========================
# Signal acquisition
# =========================
MAFAULDA_SAMPLE_RATE = 50000  # Hz — acquisition rate, used by analysis tools

# =========================
# Data quality / preprocessing
# =========================
# Files with more than this fraction of missing values are dropped; the rest
# are filled with the per-column median.
MISSING_RATIO_THRESHOLD = 0.2
MISSING_FILL_METHOD = "median"

# Normalization is per-window z-score (mean 0, std 1 computed independently per
# window). This needs no global statistics, which keeps it FPGA-friendly.
# EPSILON guards against division by zero when a window's std is ~0.
NORMALIZATION_METHOD = "zscore"
NORMALIZATION_SCOPE = "per_window"
EPSILON = 1e-8

# =========================
# Dataset split (file level)
# =========================
# Ratios are applied at file level before windowing so windows from one raw
# file never leak across splits. Must sum to 1.0 (checked below).
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# =========================
# DataLoader
# =========================
BATCH_SIZE = 64
DATALOADER_NUM_WORKERS = 0

# =========================
# Training
# =========================
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6  # epochs without val-loss improvement before stop

# =========================
# Transformer model
# =========================
# This config describes F_1layer — the single primary model, DEPLOYED to the FPGA
# (num_layers=1, d_model=32, nhead=4, dim_feedforward=64, 9,492 params; float test
# 97.55%, int10 97.11%), chosen to fit the EBAZ4205 / Zynq-7010. See DECISION_LOG
# D013/D018. Its checkpoint is kaggle/f1_ckpt_dataset/best.pt and its export lives
# in hls_f_int10/.
# History: an earlier 2-layer variant reached higher software accuracy (test
# 98.22%, checkpoint artifacts/runs/20260615_*). That is a preserved exploration
# result; to evaluate that old checkpoint set TRANSFORMER_NUM_LAYERS=2. The
# project's primary/deployed model is now this 1-layer config.
TRANSFORMER_D_MODEL = 32
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 1
TRANSFORMER_DIM_FEEDFORWARD = 64
TRANSFORMER_DROPOUT = 0.1

# =========================
# Sanity checks
# =========================
if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-8:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0")
