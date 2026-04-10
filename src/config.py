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
DATA_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

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
BATCH_SIZE = 8
DATALOADER_NUM_WORKERS = 4

USE_SMOTE_AUGMENTATION = False
USE_NOISE_INJECTION = False
NOISE_INJECTION_STD = 0.01

# =========================
# Shared training
# =========================
LEARNING_RATE = 1e-3
NUM_EPOCHS = 6   # temporary for fast testing

# =========================
# MLP baseline
# =========================
INPUT_LENGTH = WINDOW_SIZE
INPUT_CHANNELS = EXPECTED_NUM_COLUMNS
HIDDEN_DIM = 64

# =========================
# Transformer baseline
# =========================
TRANSFORMER_D_MODEL = 32
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 64
TRANSFORMER_DROPOUT = 0.1

# =========================
# Sanity checks
# =========================
if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-8:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0")