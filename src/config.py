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