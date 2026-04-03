import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.config import NOISE_INJECTION_STD, RANDOM_SEED


def flatten_samples(X):
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(X.shape[0], -1)


def restore_sample_shape(X_flat, sample_shape):
    X_flat = np.asarray(X_flat, dtype=np.float32)
    return X_flat.reshape((-1, *sample_shape))


def apply_noise_injection(X, noise_std=NOISE_INJECTION_STD, random_state=RANDOM_SEED):
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return X

    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


def apply_smote(X, y, random_state=RANDOM_SEED, k_neighbors=5):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if X.size == 0 or y.size == 0:
        return X, y

    sample_shape = X.shape[1:]
    X_flat = flatten_samples(X)

    class_counts = np.bincount(y)
    target_count = class_counts.max()

    rng = np.random.default_rng(random_state)
    synthetic_samples = []
    synthetic_labels = []

    for class_id, class_count in enumerate(class_counts):
        if class_count == 0 or class_count >= target_count:
            continue

        class_samples = X_flat[y == class_id]
        if class_samples.shape[0] < 2:
            continue

        neighbor_count = min(k_neighbors, class_samples.shape[0] - 1)
        neighbors = NearestNeighbors(n_neighbors=neighbor_count + 1)
        neighbors.fit(class_samples)
        neighbor_indices = neighbors.kneighbors(class_samples, return_distance=False)[:, 1:]

        samples_needed = target_count - class_count
        for _ in range(samples_needed):
            base_idx = int(rng.integers(0, class_samples.shape[0]))
            candidate_neighbors = neighbor_indices[base_idx]
            neighbor_idx = int(rng.choice(candidate_neighbors))

            base_sample = class_samples[base_idx]
            neighbor_sample = class_samples[neighbor_idx]
            gap = float(rng.random())
            synthetic = base_sample + gap * (neighbor_sample - base_sample)

            synthetic_samples.append(synthetic.astype(np.float32))
            synthetic_labels.append(class_id)

    if not synthetic_samples:
        return X, y

    X_aug = np.vstack([X_flat, np.vstack(synthetic_samples)])
    y_aug = np.concatenate([y, np.asarray(synthetic_labels, dtype=np.int64)])
    return restore_sample_shape(X_aug, sample_shape), y_aug


def augment_training_data(
    X,
    y,
    use_smote=False,
    use_noise_injection=False,
    noise_std=NOISE_INJECTION_STD,
    random_state=RANDOM_SEED,
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if X.size == 0 or y.size == 0:
        return X, y, {
            "smote_applied": False,
            "noise_injection_applied": False,
            "original_samples": int(len(y)),
            "augmented_samples": int(len(y)),
        }

    original_samples = int(len(y))

    if use_smote:
        X, y = apply_smote(X, y, random_state=random_state)

    if use_noise_injection:
        X_noisy = apply_noise_injection(X, noise_std=noise_std, random_state=random_state)
        X = np.concatenate([X, X_noisy], axis=0)
        y = np.concatenate([y, y.copy()], axis=0)

    augmentation_info = {
        "smote_applied": bool(use_smote),
        "noise_injection_applied": bool(use_noise_injection),
        "noise_std": float(noise_std),
        "original_samples": original_samples,
        "augmented_samples": int(len(y)),
    }

    return X.astype(np.float32), y.astype(np.int64), augmentation_info
