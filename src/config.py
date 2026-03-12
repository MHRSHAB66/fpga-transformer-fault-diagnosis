from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_RAW_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "raw" / "mafaulda_synthetic"
DATA_SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42

LABEL_MAP = {
    "normal": 0,
    "imbalance": 1,
    "misalignment": 2,
    "bearing_fault": 3,
}

WINDOW_SIZE = 128
WINDOW_STRIDE = 128

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

MISSING_RATIO_THRESHOLD = 0.2
MISSING_FILL_METHOD = "median"

NORMALIZATION_METHOD = "zscore"
EPSILON = 1e-8