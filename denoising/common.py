from pathlib import Path
import shutil

import h5py
import numpy as np

from data_loading.h5_utils import read_h5_meta, guess_keys


LIKELY_POSE_KEYS = [
    "tforms", "poses", "pose", "T", "transform", "transforms",
    "extrinsics", "world_T_probe", "probe_T_world", "sonda_swiat",
]


def find_pose_key(meta, n_frames):
    candidates = [
        k for k, (sh, _) in meta.items()
        if len(sh) == 3 and sh[-2:] == (4, 4)
    ]

    if not candidates:
        raise RuntimeError("Nie znaleziono datasetu pose o shape (N,4,4).")

    if n_frames is not None:
        candidates_n = [
            k for k in candidates
            if int(meta[k][0][0]) == int(n_frames)
        ]
    else:
        candidates_n = candidates

    for pref in LIKELY_POSE_KEYS:
        for k in candidates_n:
            name = Path(k).name.lower()
            if name.endswith(pref.lower()) or pref.lower() in name:
                return k

    if candidates_n:
        return candidates_n[0]

    return candidates[0]


def load_h5_image_and_poses(path, image_key=None):
    path = Path(path)
    meta = read_h5_meta(path)

    if image_key is None:
        image_key, _ = guess_keys(meta)

    if image_key is None:
        raise RuntimeError(f"Nie znaleziono datasetu z obrazami w: {path}")

    with h5py.File(path, "r") as f:
        img = np.asarray(f[image_key])

        sx = float(f[image_key].attrs.get("spacing_x", 1.0))
        sy = float(f[image_key].attrs.get("spacing_y", 1.0))

        if "spacing" in f[image_key].attrs:
            sp = f[image_key].attrs["spacing"]
            if len(sp) >= 2:
                sx = float(sp[0])
                sy = float(sp[1])

        n_guess = img.shape[0] if img.ndim == 3 else None
        pose_key = find_pose_key(meta, n_guess)
        poses = np.asarray(f[pose_key], dtype=np.float32)

    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3 and img.shape[0] != poses.shape[0] and img.shape[-1] == poses.shape[0]:
        img = np.moveaxis(img, -1, 0)

    if img.ndim != 3:
        raise ValueError(f"Obraz musi mieć shape (N,H,W), a ma: {img.shape}")

    if img.shape[0] != poses.shape[0]:
        raise ValueError(
            f"Niezgodna liczba klatek i pose: img={img.shape[0]}, poses={poses.shape[0]}"
        )

    return img.astype(np.float32), poses.astype(np.float32), (sx, sy), image_key


def save_h5_image_and_poses(
    path,
    image,
    poses,
    spacing_xy,
    image_key="img",
    pose_key="poses",
    attrs=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sx, sy = spacing_xy

    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            image_key,
            data=image.astype(np.float32),
            compression="gzip",
        )

        ds.attrs["spacing_x"] = float(sx)
        ds.attrs["spacing_y"] = float(sy)
        ds.attrs["spacing"] = np.array([sx, sy], dtype=np.float32)

        if attrs:
            for k, v in attrs.items():
                try:
                    ds.attrs[k] = v
                except TypeError:
                    ds.attrs[k] = str(v)

        f.create_dataset(
            pose_key,
            data=poses.astype(np.float32),
            compression="gzip",
        )


def normalize_to_float01(img, eps=1e-8):
    img = img.astype(np.float32)
    mn = float(np.nanmin(img))
    mx = float(np.nanmax(img))

    if mx - mn < eps:
        return np.zeros_like(img, dtype=np.float32), mn, mx

    out = (img - mn) / (mx - mn)
    return out.astype(np.float32), mn, mx


def restore_from_float01(img01, mn, mx):
    return (img01.astype(np.float32) * (mx - mn) + mn).astype(np.float32)


def apply_slice_wise(volume_thw, fn):
    out = np.empty_like(volume_thw, dtype=np.float32)

    for i in range(volume_thw.shape[0]):
        out[i] = fn(volume_thw[i].astype(np.float32))

        if (i + 1) % 50 == 0 or i == volume_thw.shape[0] - 1:
            print(f"[INFO] denoised slices: {i + 1}/{volume_thw.shape[0]}")

    return out


def copy_h5_to_for_reconstruction(src_h5, dst_h5):
    src_h5 = Path(src_h5)
    dst_h5 = Path(dst_h5)
    dst_h5.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_h5, dst_h5)