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
DATALOADER_NUM_WORKERS = 0

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
# Optional signal processing (Stage 2 experiments)
# All flags default to False/disabled to preserve the existing pipeline.
# Enabling any flag will create new cache entries (old entries remain but
# are no longer used).
# =========================

# Documented acquisition sample rate for MAFAULDA channels (Hz).
# Used as the reference when resampling or filtering is enabled.
MAFAULDA_SAMPLE_RATE = 50000

# Optional FFT-based downsampling of the time axis.
# When enabled, each file's time axis is resampled from MAFAULDA_SAMPLE_RATE
# to RESAMPLE_TARGET_SR before windowing. WINDOW_SIZE stays in samples;
# the time duration covered per window changes proportionally.
# The FPGA input path must use the same sample rate as the software pipeline.
RESAMPLE_ENABLED = False
RESAMPLE_TARGET_SR = 10000  # Hz

# Optional zero-phase Butterworth lowpass filter applied per column.
# Applied before normalization. Useful for suppressing high-frequency noise
# above the fault-relevant band (see spectrum analysis report for cutoff choice).
# Note: filtfilt is non-causal; a hardware equivalent would need a causal filter.
LOWPASS_FILTER_ENABLED = False
LOWPASS_CUTOFF_HZ = 5000.0  # Hz
LOWPASS_FILTER_ORDER = 4

# Optional Wiener per-column denoising applied before normalization.
WIENER_FILTER_ENABLED = False
WIENER_KERNEL_SIZE = 29

# Optional removal of low-energy segments from the time axis.
# Energy is measured as RMS over non-overlapping segments of LOW_ENERGY_SEGMENT_SIZE
# samples on the selected channel. Segments below the threshold are discarded.
# For z-score-normalized vibration signals, a threshold near zero selects
# nearly-constant (low-information) regions.
LOW_ENERGY_FILTER_ENABLED = False
LOW_ENERGY_CHANNEL_IDX = 7      # 7 = microphone column by default
LOW_ENERGY_THRESHOLD_RMS = 0.01
LOW_ENERGY_SEGMENT_SIZE = 512

# =========================
# Channel selection
# =========================
# "all"            — use all 8 columns: tachometer + 6 accel + microphone (default)
# "vibration_only" — use only the 6 accelerometer columns (cols 1-6)
#
# Empirically verified MAFAULDA column order:
#   col 0 = tachometer   (27 ZC/s, skew=+2.5, asymmetric positive pulses)
#   col 1 = acc_axial_b1         (underhang bearing, axial)
#   col 2 = acc_radial_b1        (underhang bearing, radial)
#   col 3 = acc_tangential_b1    (underhang bearing, tangential)
#   col 4 = acc_axial_b2         (overhang bearing, axial)
#   col 5 = acc_radial_b2        (overhang bearing, radial)
#   col 6 = acc_tangential_b2    (overhang bearing, tangential)
#   col 7 = microphone           (39 000+ ZC/s, acoustic)
CHANNEL_SELECTION = "all"   # "all" | "vibration_only"

_VIBRATION_CHANNEL_INDICES = [1, 2, 3, 4, 5, 6]  # accelerometer columns only

if CHANNEL_SELECTION == "vibration_only":
    ACTIVE_CHANNEL_INDICES: list = _VIBRATION_CHANNEL_INDICES
elif CHANNEL_SELECTION == "all":
    ACTIVE_CHANNEL_INDICES = list(range(EXPECTED_NUM_COLUMNS))   # 0-7
else:
    raise ValueError(f"Unknown CHANNEL_SELECTION: {CHANNEL_SELECTION!r}. "
                     "Valid values are 'all' and 'vibration_only'.")

NUM_ACTIVE_CHANNELS = len(ACTIVE_CHANNEL_INDICES)

# =========================
# Sanity checks
# =========================
if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-8:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be 1.0")