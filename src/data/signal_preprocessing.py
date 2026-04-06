from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.signal import resample, wiener
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "librosa is required for silence removal. Install it with `pip install librosa`."
    ) from exc


def remove_silence_librosa(
    signal: np.ndarray,
    sample_rate: int,
    top_db: float = 25.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Remove low-energy (silent) regions using librosa's frame-energy splitter.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    intervals = librosa.effects.split(
        signal,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    if len(intervals) == 0:
        return np.array([], dtype=np.float32)

    return np.concatenate([signal[start:end] for start, end in intervals], axis=0)


def denoise_wiener(signal: np.ndarray, mysize: int = 29) -> np.ndarray:
    """
    Wiener denoising for 1D vibration/audio signals.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    return np.asarray(wiener(signal, mysize=mysize), dtype=np.float32)


def resample_fft(signal: np.ndarray, original_sr: int = 5000, target_sr: int = 1000) -> np.ndarray:
    """
    FFT-based resampling from original_sr to target_sr.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return signal

    target_len = int(round(signal.shape[0] * target_sr / original_sr))
    target_len = max(1, target_len)
    return np.asarray(resample(signal, target_len), dtype=np.float32)


@dataclass
class GlobalSignalScaler:
    """
    Fits one scaler across both modalities so scaling is consistent.
    """

    method: str = "standard"
    scaler: StandardScaler | MinMaxScaler | None = None

    def fit(self, signals: Iterable[np.ndarray]) -> "GlobalSignalScaler":
        flattened: List[np.ndarray] = [
            np.asarray(sig, dtype=np.float32).reshape(-1, 1) for sig in signals if np.asarray(sig).size > 0
        ]
        if not flattened:
            raise ValueError("No non-empty signals were provided to fit the scaler.")

        values = np.vstack(flattened)

        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'.")

        self.scaler.fit(values)
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler is not fitted. Call fit() first.")

        signal = np.asarray(signal, dtype=np.float32).reshape(-1, 1)
        return self.scaler.transform(signal).reshape(-1).astype(np.float32)


def preprocess_signal(
    signal: np.ndarray,
    sample_rate: int = 5000,
    target_sr: int = 1000,
    top_db: float = 25.0,
    wiener_kernel: int = 29,
) -> np.ndarray:
    """
    Silence removal -> Wiener denoising -> FFT resampling.
    Scaling should be applied later with a fitted GlobalSignalScaler.
    """
    cleaned = remove_silence_librosa(signal, sample_rate=sample_rate, top_db=top_db)
    denoised = denoise_wiener(cleaned, mysize=wiener_kernel)
    return resample_fft(denoised, original_sr=sample_rate, target_sr=target_sr)


def preprocess_modalities(
    audio_signals: Iterable[np.ndarray],
    vibration_signals: Iterable[np.ndarray],
    sample_rate: int = 5000,
    target_sr: int = 1000,
    scale_method: str = "standard",
) -> Dict[str, List[np.ndarray]]:
    """
    End-to-end preprocessing for MAFAULDA audio + vibration streams.
    Uses one fitted scaler on both modalities for consistent normalization.
    """
    audio_pre = [preprocess_signal(sig, sample_rate=sample_rate, target_sr=target_sr) for sig in audio_signals]
    vibration_pre = [
        preprocess_signal(sig, sample_rate=sample_rate, target_sr=target_sr) for sig in vibration_signals
    ]

    scaler = GlobalSignalScaler(method=scale_method).fit(audio_pre + vibration_pre)
    audio_scaled = [scaler.transform(sig) for sig in audio_pre]
    vibration_scaled = [scaler.transform(sig) for sig in vibration_pre]

    return {"audio": audio_scaled, "vibration": vibration_scaled}


def extract_mfcc_features(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extract fixed-length MFCC features by summarizing each coefficient with mean and std.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return np.zeros(n_mfcc * 2, dtype=np.float32)

    effective_n_fft = min(n_fft, signal.size)
    effective_n_fft = max(32, effective_n_fft)
    effective_hop = min(hop_length, max(1, signal.size // 4))

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=effective_n_fft,
        hop_length=effective_hop,
    )

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)


def compute_rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_zero_crossing_rate(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size < 2:
        return 0.0

    signs = np.signbit(signal)
    zero_crossings = np.count_nonzero(signs[1:] != signs[:-1])
    return float(zero_crossings / (signal.size - 1))


def compute_skewness(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size < 3 or np.allclose(signal, signal[0]):
        return 0.0
    value = skew(signal, bias=False)
    return float(np.nan_to_num(value))


def compute_kurtosis(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size < 4 or np.allclose(signal, signal[0]):
        return 0.0
    value = kurtosis(signal, fisher=True, bias=False)
    return float(np.nan_to_num(value))


def extract_vibration_features(signal: np.ndarray) -> np.ndarray:
    """
    Extract basic statistical features from one vibration channel.
    """
    return np.array(
        [
            compute_rms(signal),
            compute_zero_crossing_rate(signal),
            compute_skewness(signal),
            compute_kurtosis(signal),
        ],
        dtype=np.float32,
    )


def extract_combined_features(
    audio_signal: np.ndarray,
    vibration_signals: Sequence[np.ndarray] | np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    Combine audio MFCC features with vibration statistics into one flat feature vector.
    """
    audio_features = extract_mfcc_features(audio_signal, sample_rate=sample_rate, n_mfcc=n_mfcc)

    vibration_array = np.asarray(vibration_signals, dtype=np.float32)
    if vibration_array.ndim == 1:
        vibration_array = vibration_array.reshape(1, -1)

    vibration_features = [
        extract_vibration_features(channel_signal)
        for channel_signal in vibration_array
    ]
    fused_vibration = (
        np.concatenate(vibration_features).astype(np.float32)
        if vibration_features
        else np.zeros(0, dtype=np.float32)
    )

    return np.concatenate([audio_features, fused_vibration]).astype(np.float32)


def prepare_feature_dataset(
    audio_signals: Sequence[np.ndarray],
    vibration_signals: Sequence[Sequence[np.ndarray] | np.ndarray],
    sample_rate: int,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    Build a feature matrix with one row per sample, ready for model input.
    """
    if len(audio_signals) != len(vibration_signals):
        raise ValueError("audio_signals and vibration_signals must have the same number of samples.")

    if len(audio_signals) == 0:
        return np.empty((0, n_mfcc * 2), dtype=np.float32)

    rows = [
        extract_combined_features(
            audio_signal=audio_signal,
            vibration_signals=vibration_signal,
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
        )
        for audio_signal, vibration_signal in zip(audio_signals, vibration_signals)
    ]

    return np.vstack(rows).astype(np.float32)
