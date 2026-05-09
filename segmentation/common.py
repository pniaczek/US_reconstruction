from pathlib import Path
import json

import h5py
import numpy as np
from PIL import Image

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
        raise RuntimeError("Nie znaleziono datasetu z pose o shape (N,4,4).")

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


def load_model_meta(meta_path):
    meta_path = Path(meta_path)

    if not meta_path.exists():
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_input_size(meta, fallback_size=(256, 256)):
    """
    Zwraca (H, W).
    """
    if not meta:
        return fallback_size

    input_meta = meta.get("input", {})
    size = input_meta.get("image_size", None)

    if size is None:
        size = input_meta.get("example_trace_size", None)

    if size is None:
        return fallback_size

    return int(size[0]), int(size[1])


def normalize_frame_for_unet(frame, mode="auto"):
    """
    frame -> float32 0..1.

    mode:
      "auto"         : jeśli max > 1.5, dziel przez 255
      "minmax"       : normalizacja per frame do 0..1
      "divide_255"   : clip 0..255 i /255
      "none"         : zakłada, że już jest 0..1
    """
    x = frame.astype(np.float32)

    if mode == "minmax":
        mn = float(np.nanmin(x))
        mx = float(np.nanmax(x))
        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mn) / (mx - mn)).astype(np.float32)

    if mode == "divide_255":
        return (np.clip(x, 0.0, 255.0) / 255.0).astype(np.float32)

    if mode == "none":
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    # auto
    mx = float(np.nanmax(x))
    if mx > 1.5:
        return (np.clip(x, 0.0, 255.0) / 255.0).astype(np.float32)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def resize_frame_float(frame01, size_hw):
    h, w = size_hw
    img = Image.fromarray(np.clip(frame01 * 255.0, 0, 255).astype(np.uint8), mode="L")
    img = img.resize((w, h), resample=Image.BILINEAR)
    out = np.asarray(img).astype(np.float32) / 255.0
    return out


def resize_mask_nearest(mask, size_hw):
    h, w = size_hw
    img = Image.fromarray(mask.astype(np.uint8), mode="L")
    img = img.resize((w, h), resample=Image.NEAREST)
    return np.asarray(img).astype(np.uint8)