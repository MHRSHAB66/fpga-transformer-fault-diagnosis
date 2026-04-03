from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from scipy.signal import resample, wiener
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

