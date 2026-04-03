from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main dataset
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mafaulda"
# Synthetic dataset
# DATA_RAW_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "raw" / "mafaulda_synthetic"
DATA_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42

LABEL_MAP = {
    "normal": 0,
    "imbalance": 1,
    "misalignment": 2,
    "bearing_fault": 3,
}

WINDOW_SIZE = 1024
WINDOW_STRIDE = 1024

EXPECTED_NUM_COLUMNS = 8

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

MIN_ROWS_PER_FILE = 128

# Drop file if it it has more than 20% missing.
MISSING_RATIO_THRESHOLD = 0.2
MISSING_FILL_METHOD = "median"

NORMALIZATION_METHOD = "zscore"
EPSILON = 1e-8

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

BATCH_SIZE = 8

#MLP baseline
NUM_CLASSES = 4
INPUT_LENGTH = WINDOW_SIZE
INPUT_CHANNELS = EXPECTED_NUM_COLUMNS
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3
HIDDEN_DIM = 64

#TRANSFORMER baseline
TRANSFORMER_D_MODEL = 32
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 64
TRANSFORMER_DROPOUT = 0.1
