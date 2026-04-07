"""
Data package initializer.

Signal preprocessing helpers are kept as lazy exports for backward
compatibility without forcing optional feature-extraction dependencies during
dataloader imports.
"""

_SIGNAL_PREPROCESSING_EXPORTS = {
    "GlobalSignalScaler",
    "compute_kurtosis",
    "compute_rms",
    "compute_skewness",
    "compute_zero_crossing_rate",
    "denoise_wiener",
    "extract_combined_features",
    "extract_mfcc_features",
    "extract_vibration_features",
    "prepare_feature_dataset",
    "preprocess_modalities",
    "preprocess_signal",
    "remove_silence_librosa",
    "resample_fft",
}

__all__ = sorted(_SIGNAL_PREPROCESSING_EXPORTS)


def __getattr__(name):
    if name in _SIGNAL_PREPROCESSING_EXPORTS:
        from src.tools import signal_preprocessing

        return getattr(signal_preprocessing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
