from pathlib import Path
import numpy as np
import pandas as pd

out_root = Path("data/raw/mafaulda_synthetic")
out_root.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
fs = 5000
n_rows = 512
t = np.arange(n_rows) / fs

def make_channels(label, file_idx):
    base_freq = 30 + (file_idx % 3) * 5
    rpm_signal = 1500 + 20 * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0, 2.0, size=n_rows)
    noise = lambda s: rng.normal(0, s, size=n_rows)

    c1 = 0.08 * np.sin(2 * np.pi * base_freq * t) + noise(0.015)
    c2 = 0.06 * np.sin(2 * np.pi * (base_freq + 2) * t + 0.3) + noise(0.015)
    c3 = 0.05 * np.sin(2 * np.pi * (base_freq - 3) * t + 0.7) + noise(0.015)
    c4 = 0.07 * np.sin(2 * np.pi * (base_freq + 1) * t + 0.1) + noise(0.015)
    c5 = 0.05 * np.sin(2 * np.pi * (base_freq + 4) * t + 0.6) + noise(0.015)
    c6 = 0.04 * np.sin(2 * np.pi * (base_freq - 1) * t + 1.0) + noise(0.015)
    mic = 0.1 * np.sin(2 * np.pi * 220 * t) + noise(0.02)

    severity = 1 + (file_idx % 5) / 4.0

    if label == "imbalance":
        low = 0.18 * severity * np.sin(2 * np.pi * base_freq * t)
        c2 += low
        c5 += 0.12 * severity * np.sin(2 * np.pi * base_freq * t + 0.2)
        mic += 0.03 * severity * np.sin(2 * np.pi * base_freq * t)

    elif label == "misalignment":
        harm2 = 0.14 * severity * np.sin(2 * np.pi * 2 * base_freq * t + 0.4)
        harm3 = 0.07 * severity * np.sin(2 * np.pi * 3 * base_freq * t + 0.2)
        c1 += harm2
        c3 += harm3
        c4 += 0.10 * severity * np.sin(2 * np.pi * 2 * base_freq * t + 1.0)
        mic += 0.04 * severity * np.sin(2 * np.pi * 2 * base_freq * t)

    elif label == "bearing_fault":
        impulse_positions = np.arange(60, n_rows, max(80, 150 - 15 * int(severity)))
        impulses = np.zeros(n_rows)
        impulses[impulse_positions] = 0.4 * severity
        kernel = np.exp(-np.linspace(0, 1, 20) * 8)
        bursts = np.convolve(impulses, kernel, mode="same")
        hf = 0.06 * np.sin(2 * np.pi * 420 * t) + 0.04 * np.sin(2 * np.pi * 760 * t)
        c1 += bursts + hf
        c4 += 0.8 * bursts + 0.8 * hf
        c6 += 0.6 * bursts + 0.5 * hf
        mic += 0.07 * severity * np.sin(2 * np.pi * 600 * t) + 0.5 * bursts

    return pd.DataFrame({
        "tachometer": rpm_signal.round(6),
        "acc_axial_b1": c1.round(6),
        "acc_radial_b1": c2.round(6),
        "acc_tangential_b1": c3.round(6),
        "acc_axial_b2": c4.round(6),
        "acc_radial_b2": c5.round(6),
        "acc_tangential_b2": c6.round(6),
        "microphone": mic.round(6),
    })

files_info = []

labels = ["normal", "imbalance", "misalignment", "bearing_fault"]

for label in labels:
    class_dir = out_root / label
    class_dir.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        df = make_channels(label, i)
        file_path = class_dir / f"{label}_{i+1:02d}.csv"
        df.to_csv(file_path, index=False)

        files_info.append({
            "filepath": str(file_path),
            "label": label,
            "rows": n_rows,
            "columns": 8
        })

manifest = pd.DataFrame(files_info)
manifest.to_csv(out_root / "manifest.csv", index=False)

print("Done. Dataset created in:")
print(out_root.resolve())
print()
print(manifest)