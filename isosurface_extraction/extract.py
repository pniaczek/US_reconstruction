from pathlib import Path
import json
import re
import numpy as np
from skimage.measure import marching_cubes

from config import (
    GROUND_TRUTH_DIR,
    RECONSTRUCTED_DIR,
    ISO_SOURCE_SUBDIR,
    ISO_PATTERN,
    ISO_LEVEL,
    ISO_THR_RATIO,
    ISO_MIN_VOXELS,
    ISO_STEP_SIZE,
    ISO_SAVE_OBJ,
)


def find_first_array_in_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    preferred_keys = ["volume", "arr_0", "data", "reconstruction", "prob", "probs"]
    for key in preferred_keys:
        if key in data and isinstance(data[key], np.ndarray):
            return np.asarray(data[key]), tuple(data.get("voxel_size", [1.0, 1.0, 1.0]))
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray):
            return np.asarray(arr), tuple(data.get("voxel_size", [1.0, 1.0, 1.0]))
    raise ValueError(f"No ndarray found in: {npz_path}")


def choose_level(volume, level, thr_ratio):
    vmax = float(np.max(volume))
    if vmax <= 0:
        raise ValueError("Volume max <= 0")
    if level is not None:
        return float(level)
    unique_vals = np.unique(volume)
    if unique_vals.size <= 3 and set(np.round(unique_vals, 6)).issubset({0.0, 1.0}):
        return 0.5
    return float(vmax * thr_ratio)


def save_obj(obj_path, verts, faces):
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# OBJ generated from marching cubes\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def extract_surface(volume, level, spacing_xyz, step_size=1):
    sx, sy, sz = spacing_xyz
    verts, faces, normals, values = marching_cubes(
        volume,
        level=level,
        spacing=(sz, sy, sx),
        step_size=step_size,
        allow_degenerate=False,
    )
    verts_xyz = verts[:, [2, 1, 0]]
    normals_xyz = normals[:, [2, 1, 0]]
    return verts_xyz, faces.astype(np.int32), normals_xyz, values.astype(np.float32)


def extract_class_number(name):
    m = re.search(r"class(\d+)", name.lower())
    return int(m.group(1)) if m else None


def get_class_color(name):
    color_map = {
        1: "red", 2: "green", 3: "blue", 4: "yellow", 5: "magenta",
        6: "cyan", 7: "orange", 8: "purple", 9: "lime", 10: "pink",
    }
    class_num = extract_class_number(name)
    return color_map.get(class_num, "white")


def extract_surfaces_in_dir(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(input_dir.glob(ISO_PATTERN))
    if not candidates:
        print(f"[WARN] No candidates found in: {input_dir}")
        return

    summary = []

    for path in candidates:
        volume, spacing_xyz = find_first_array_in_npz(path)
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        positive_voxels = int(np.count_nonzero(volume > 0))
        if positive_voxels < ISO_MIN_VOXELS:
            continue

        try:
            level = choose_level(volume, ISO_LEVEL, ISO_THR_RATIO)
            verts, faces, normals, values = extract_surface(volume, level, spacing_xyz, ISO_STEP_SIZE)
        except Exception as e:
            print(f"[WARN] Failed surface extraction for {path.name}: {e}")
            continue

        if len(verts) == 0 or len(faces) == 0:
            continue

        color = get_class_color(path.stem)
        out_npz = output_dir / f"{path.stem}_surface.npz"
        np.savez_compressed(
            out_npz,
            verts=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float32),
            values=values.astype(np.float32),
            color=np.array([color], dtype=object),
            source_file=np.array([str(path)], dtype=object),
        )

        if ISO_SAVE_OBJ:
            save_obj(output_dir / f"{path.stem}_surface.obj", verts, faces)

        summary.append({
            "source": path.name,
            "surface": out_npz.name,
            "verts": int(len(verts)),
            "faces": int(len(faces)),
            "color": color,
        })

    with open(output_dir / "surface_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Surface extraction finished for: {input_dir}")


def extract_surfaces_for_case(case_name):
    gt_case_dir = GROUND_TRUTH_DIR / case_name
    recon_case_dir = RECONSTRUCTED_DIR / case_name

    gt_seg_dir = gt_case_dir / ISO_SOURCE_SUBDIR
    if not gt_seg_dir.exists():
        gt_seg_dir = gt_case_dir / "seg_classes"
    if gt_seg_dir.exists():
        extract_surfaces_in_dir(gt_seg_dir, gt_case_dir / "surfaces")

    recon_seg_dir = recon_case_dir / ISO_SOURCE_SUBDIR
    if recon_seg_dir.exists():
        extract_surfaces_in_dir(recon_seg_dir, recon_case_dir / "surfaces")


def extract_surfaces_for_cases(case_names=None):
    case_dirs = sorted([p for p in GROUND_TRUTH_DIR.iterdir() if p.is_dir()])
    if case_names is not None:
        case_dirs = [p for p in case_dirs if p.name in case_names]

    if not case_dirs:
        print(f"[WARN] No cases found in: {GROUND_TRUTH_DIR}")
        return

    for case_dir in case_dirs:
        extract_surfaces_for_case(case_dir.name)
