from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS — ZMIEŃ TYLKO TO
# ============================================================

BEFORE_H5 = Path(r"data/for_reconstruction/kosartur/kosartur_from_dicom.h5")
AFTER_H5 = Path(r"data/denoised/kosartur/tv/kosartur_from_dicom_tv.h5")

IMAGE_KEY = "img"

N_IMAGES = 1
START_INDEX = 0

# Jeśli True, każda para before/after ma wspólną skalę jasności.
SAME_CONTRAST_PER_PAIR = True

# Jeśli True, używa percentyli zamiast min/max — zwykle lepiej dla USG.
USE_PERCENTILE_CONTRAST = True
P_LOW = 1
P_HIGH = 99


# ============================================================
# HELPERS
# ============================================================

def load_h5_images(path: Path, image_key: str = "img") -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku: {path}")

    with h5py.File(path, "r") as f:
        if image_key not in f:
            raise KeyError(
                f"Brak datasetu '{image_key}' w {path}. "
                f"Dostępne klucze: {list(f.keys())}"
            )

        img = np.asarray(f[image_key])

    if img.ndim == 2:
        img = img[None, ...]

    if img.ndim == 3 and img.shape[0] < img.shape[-1]:
        # Zakładamy typowy układ (N,H,W). Ten warunek jest tylko asekuracyjny.
        pass

    if img.ndim != 3:
        raise ValueError(f"Oczekiwano shape (N,H,W), dostałem: {img.shape}")

    return img.astype(np.float32)


def contrast_limits(a, b):
    if SAME_CONTRAST_PER_PAIR:
        data = np.concatenate([a.ravel(), b.ravel()])
    else:
        data = a.ravel()

    if USE_PERCENTILE_CONTRAST:
        vmin = np.percentile(data, P_LOW)
        vmax = np.percentile(data, P_HIGH)
    else:
        vmin = np.min(data)
        vmax = np.max(data)

    if vmax <= vmin:
        vmax = vmin + 1e-6

    return float(vmin), float(vmax)


def main():
    before = load_h5_images(BEFORE_H5, IMAGE_KEY)
    after = load_h5_images(AFTER_H5, IMAGE_KEY)

    if before.shape[0] != after.shape[0]:
        print(f"[WARN] Różna liczba klatek: before={before.shape[0]}, after={after.shape[0]}")

    n_total = min(before.shape[0], after.shape[0])
    end = min(START_INDEX + N_IMAGES, n_total)

    indices = list(range(START_INDEX, end))

    if not indices:
        raise RuntimeError("Brak klatek do pokazania.")

    rows = len(indices)
    fig, axes = plt.subplots(rows, 2, figsize=(8, rows * 2.2))

    if rows == 1:
        axes = np.array([axes])

    for row, idx in enumerate(indices):
        b = before[idx]
        a = after[idx]

        vmin, vmax = contrast_limits(b, a)

        axes[row, 0].imshow(b, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Before #{idx}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(a, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"After #{idx}")
        axes[row, 1].axis("off")

    fig.suptitle("Denoising preview: before vs after", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()