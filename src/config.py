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
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mafaulda"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
DEFAULT_RUN_NAME = "transformer_final_t4"

# =========================
# Reproducibility
# =========================
RANDOM_SEED = 42

# =========================
# Labels
# =========================
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

WINDOW_SIZE = 1024
WINDOW_STRIDE = 1024
MIN_ROWS_PER_FILE = WINDOW_SIZE

# =========================
# Data quality / preprocessing
# =========================
MISSING_RATIO_THRESHOLD = 0.2
MISSING_FILL_METHOD = "median"

NORMALIZATION_METHOD = "zscore"
NORMALIZATION_SCOPE = "per_window"
EPSILON = 1e-8

# =========================
# Dataset split
# =========================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# =========================
# DataLoader / augmentation
# =========================
BATCH_SIZE = 64
DATALOADER_NUM_WORKERS = 0

# =========================
# Shared training
# =========================
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6

# =========================
# MLP baseline
# =========================
INPUT_LENGTH = WINDOW_SIZE
HIDDEN_DIM = 64

# =========================
# Transformer baseline
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

MAFAULDA_SAMPLE_RATE = 50000  # Hz — acquisition rate, used by analysis tools

# =========================
# Channel selection
# =========================
# col 0 = tachometer, col 1-3 = underhang bearing (axial/radial/tangential),
# col 4-6 = overhang bearing (axial/radial/tangential), col 7 = microphone
INPUT_CHANNELS = EXPECTED_NUM_COLUMNS  # 8 — all columns used

# =========================
# Sanity checks
# =========================
if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-8:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0")
