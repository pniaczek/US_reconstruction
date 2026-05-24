from pathlib import Path
import json
import shutil
import re

import cv2
import h5py
import numpy as np
from PIL import Image

from config import (
    ROOT,
    FOR_RECON_DIR,
    SEGMENTATION_DIR,
    UNSUPERVISED_REPO_DIR,
    UNSUPERVISED_DATASET_NAME,
    UNSUPERVISED_CASE_NAME,
    UNSUPERVISED_WANDB_TAG,
    UNSUPERVISED_IMPORT_USE_CRF,
    UNSUPERVISED_IMPORT_MODEL_NAME,
    UNSUPERVISED_IMPORT_TO_FOR_RECONSTRUCTION,
    UNSUPERVISED_IMPORT_TO_SEGMENTATION_DIR,
    UNSUPERVISED_IMPORT_OVERWRITE,
    UNSUPERVISED_OUTPUT_IMPORTS,
)

try:
    from config import UNSUPERVISED_OUTPUT_IMPORT_CASES
except ImportError:
    UNSUPERVISED_OUTPUT_IMPORT_CASES = []

from data_loading.h5_utils import list_h5_files, read_h5_meta, guess_keys


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_case_names(case_names):
    if case_names is None:
        return []

    if isinstance(case_names, str):
        return [case_names]

    return list(case_names)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_name(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "-")
    )


def as_path_or_none(x):
    if x is None:
        return None
    return Path(x)


def make_source_case(base_case=None, denoising_method=None, source_case=None):
    if source_case:
        return str(source_case)

    if not base_case:
        raise ValueError("Either source_case or base_case must be provided.")

    if denoising_method in [None, "", "none", "raw", "original"]:
        return str(base_case)

    return f"{base_case}_denoised_{denoising_method}"


def make_output_case(source_case, segments_num=None, clusters_num=None, use_crf=True, output_case=None):
    if output_case:
        return str(output_case)

    suffix = "crf" if use_crf else "raw"

    if segments_num is not None and clusters_num is not None:
        return sanitize_name(
            f"{source_case}_unsup_seg{int(segments_num)}_clust{int(clusters_num)}_{suffix}"
        )

    return sanitize_name(f"{source_case}_unsup_{suffix}")


def center_crop_info(shape_hw, multiple=14):
    h, w = shape_hw

    crop_h = (h // multiple) * multiple
    crop_w = (w // multiple) * multiple

    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Cannot crop shape {(h, w)} to multiple={multiple}")

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    return top, left, crop_h, crop_w


# ============================================================
# SOURCE H5
# ============================================================

def load_source_h5(case_name):
    case_dir = FOR_RECON_DIR / case_name

    if not case_dir.exists():
        raise FileNotFoundError(f"Missing source case dir: {case_dir}")

    h5_files = list_h5_files(case_dir)

    if not h5_files:
        raise RuntimeError(f"No H5 files found in: {case_dir}")

    if len(h5_files) > 1:
        print("[WARN] More than one H5 found. Using first:")
        for p in h5_files:
            print("  -", p)

    h5_path = Path(h5_files[0])
    meta = read_h5_meta(h5_path)
    image_key, pose_key = guess_keys(meta)

    if image_key is None:
        image_key = "img"

    if pose_key is None:
        pose_key = "poses"

    with h5py.File(h5_path, "r") as f:
        if image_key not in f:
            raise KeyError(
                f"Image key '{image_key}' not found in {h5_path}. "
                f"Available keys: {list(f.keys())}"
            )

        img = np.asarray(f[image_key])

        poses = None
        if pose_key in f:
            poses = np.asarray(f[pose_key])

        attrs = dict(f.attrs)

        spacing_xy = None
        if "spacing_xy" in f:
            spacing_xy = np.asarray(f["spacing_xy"])
        elif "pixel_spacing" in f:
            spacing_xy = np.asarray(f["pixel_spacing"])
        elif "spacing" in f:
            spacing_xy = np.asarray(f["spacing"])

    if img.ndim != 3:
        raise ValueError(f"Expected source image shape (N,H,W), got {img.shape}")

    return {
        "h5_path": h5_path,
        "image": img,
        "poses": poses,
        "spacing_xy": spacing_xy,
        "attrs": attrs,
        "image_key": image_key,
        "pose_key": pose_key,
    }


# ============================================================
# UNSUPERVISED RUN FINDING
# ============================================================

def get_unsupervised_output_base():
    dss_root = UNSUPERVISED_REPO_DIR / "deep-spectral-segmentation"

    return (
        dss_root
        / UNSUPERVISED_DATASET_NAME
        / UNSUPERVISED_CASE_NAME
        / "main"
        / UNSUPERVISED_WANDB_TAG
    )


def has_segmentation_outputs(path):
    path = Path(path)

    direct_segmaps = path / "segmaps"
    direct_crf = path / "crf_segmaps"

    nested_segmaps = path / "semantic_segmentations" / "laplacian" / "segmaps"
    nested_crf = path / "semantic_segmentations" / "laplacian" / "crf_segmaps"

    laplacian_segmaps = path / "semantic_segmentations" / "laplacian"
    if laplacian_segmaps.exists():
        if (laplacian_segmaps / "segmaps").exists() or (laplacian_segmaps / "crf_segmaps").exists():
            return True

    return (
        direct_segmaps.exists()
        or direct_crf.exists()
        or nested_segmaps.exists()
        or nested_crf.exists()
    )


def extract_seg_clust_from_name(path):
    text = str(path)

    seg = None
    clust = None

    m = re.search(r"seg(\d+)", text)
    if m:
        seg = int(m.group(1))

    m = re.search(r"clust(\d+)", text)
    if m:
        clust = int(m.group(1))

    return seg, clust


def score_run_candidate(path, source_case=None, denoising_method=None, segments_num=None, clusters_num=None):
    path = Path(path)
    text = str(path).lower()
    name = path.name.lower()

    score = 0

    if segments_num is not None:
        if f"seg{int(segments_num)}" in text:
            score += 100

    if clusters_num is not None:
        if f"clust{int(clusters_num)}" in text:
            score += 100

    if denoising_method:
        den = str(denoising_method).lower()
        if den in text:
            score += 20

    if source_case:
        src = str(source_case).lower()
        if src in text:
            score += 10

    # Prefer actual run dirs over intermediate dirs.
    if "time" in name:
        score += 5

    return score


def find_matching_unsupervised_run_dir(
    source_case=None,
    denoising_method=None,
    segments_num=None,
    clusters_num=None,
):
    """
    Szuka najlepszego/najnowszego runu pod:
      external/.../deep-spectral-segmentation/DATASET/CASE/main/TAG/

    Nie wymaga ręcznego podawania ścieżki.
    Preferuje katalogi zawierające seg{segments_num} i clust{clusters_num}.
    """
    base = get_unsupervised_output_base()

    if not base.exists():
        raise FileNotFoundError(f"Cannot find unsupervised output base: {base}")

    candidates = []

    for p in base.rglob("*"):
        if not p.is_dir():
            continue

        if has_segmentation_outputs(p):
            candidates.append(p)

    if not candidates:
        raise RuntimeError(f"No unsupervised run dirs found under: {base}")

    scored = []

    for p in candidates:
        score = score_run_candidate(
            path=p,
            source_case=source_case,
            denoising_method=denoising_method,
            segments_num=segments_num,
            clusters_num=clusters_num,
        )

        scored.append((score, p.stat().st_mtime, p))

    scored = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)

    best_score, best_mtime, best_path = scored[0]

    print("============================================================")
    print("[INFO] Unsupervised run search")
    print(f"[INFO] base             : {base}")
    print(f"[INFO] source_case      : {source_case}")
    print(f"[INFO] denoising_method : {denoising_method}")
    print(f"[INFO] segments_num     : {segments_num}")
    print(f"[INFO] clusters_num     : {clusters_num}")
    print(f"[INFO] candidates found : {len(candidates)}")
    print(f"[INFO] selected score   : {best_score}")
    print(f"[INFO] selected run     : {best_path}")

    if len(scored) > 1:
        print("[INFO] top candidates:")
        for s, mt, p in scored[:5]:
            print(f"  score={s:4d} | {p}")

    return best_path


# ============================================================
# SEGMENTATION PNG DIRECTORY
# ============================================================

def get_segmentation_png_dir(run_dir, use_crf=True):
    """
    Obsługuje kilka wariantów:
    1) run_dir = katalog runu:
       .../seg30_clust15_time.../
    2) run_dir = .../semantic_segmentations/laplacian/
    3) run_dir = .../semantic_segmentations/laplacian/segmaps
    4) run_dir = .../semantic_segmentations/laplacian/crf_segmaps
    """
    run_dir = Path(run_dir)

    if run_dir.name in ["segmaps", "crf_segmaps"]:
        if run_dir.exists():
            return run_dir

    if run_dir.name == "laplacian":
        crf_dir = run_dir / "crf_segmaps"
        segmaps_dir = run_dir / "segmaps"

        if use_crf and crf_dir.exists():
            return crf_dir

        if use_crf and not crf_dir.exists():
            print(f"[WARN] CRF segmaps not found, falling back to raw segmaps: {crf_dir}")

        if segmaps_dir.exists():
            return segmaps_dir

    segmaps_dir = run_dir / "semantic_segmentations" / "laplacian" / "segmaps"
    crf_dir = run_dir / "semantic_segmentations" / "laplacian" / "crf_segmaps"

    if use_crf:
        if crf_dir.exists():
            return crf_dir

        print(f"[WARN] CRF segmaps not found, falling back to raw segmaps: {crf_dir}")

    if segmaps_dir.exists():
        return segmaps_dir

    raise FileNotFoundError(
        "No segmentation PNG dir found. Checked:\n"
        f"  {crf_dir}\n"
        f"  {segmaps_dir}\n"
        f"  {run_dir / 'crf_segmaps'}\n"
        f"  {run_dir / 'segmaps'}"
    )


# ============================================================
# READ / ALIGN LABELS
# ============================================================

def read_unsupervised_png_stack(seg_dir, source_shape):
    """
    Czyta PNG z unsupervised i dopasowuje do oryginalnego rozmiaru H5.

    Repo external pracowało na center-cropie do wielokrotności 14.
    Dla danych 697x397 crop to 686x392.
    Tu wracamy do oryginalnego H,W przez padding zerami.
    """
    seg_dir = Path(seg_dir)

    png_files = sorted(seg_dir.glob("*.png"))

    if not png_files:
        raise RuntimeError(f"No PNG files found in: {seg_dir}")

    n, h_orig, w_orig = source_shape
    top, left, crop_h, crop_w = center_crop_info((h_orig, w_orig), multiple=14)

    if len(png_files) != n:
        print(
            "[WARN] Number of segmentation PNGs differs from source frames: "
            f"png={len(png_files)}, source={n}"
        )

    n_out = min(len(png_files), n)

    labels = np.zeros((n_out, h_orig, w_orig), dtype=np.uint8)

    for i in range(n_out):
        seg = np.array(Image.open(png_files[i]))

        if seg.ndim == 3:
            seg = seg[:, :, 0]

        seg = seg.astype(np.uint8)

        if seg.shape != (crop_h, crop_w):
            seg = cv2.resize(
                seg,
                dsize=(crop_w, crop_h),
                interpolation=cv2.INTER_NEAREST,
            )

        labels[i, top:top + crop_h, left:left + crop_w] = seg

    return labels


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_h5_for_reconstruction(output_case, labels, source_data, run_dir, seg_dir):
    out_case_dir = FOR_RECON_DIR / output_case

    if out_case_dir.exists() and UNSUPERVISED_IMPORT_OVERWRITE:
        shutil.rmtree(out_case_dir)

    ensure_dir(out_case_dir)

    out_h5 = out_case_dir / f"{output_case}.h5"

    poses = source_data["poses"]
    spacing_xy = source_data["spacing_xy"]

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("img", data=labels.astype(np.float32), compression="gzip")
        f.create_dataset("labels", data=labels.astype(np.uint8), compression="gzip")

        if poses is not None:
            f.create_dataset("poses", data=poses)

        if spacing_xy is not None:
            f.create_dataset("spacing_xy", data=spacing_xy)

        for k, v in source_data["attrs"].items():
            try:
                f.attrs[k] = v
            except Exception:
                pass

        f.attrs["source_case_h5"] = str(source_data["h5_path"])
        f.attrs["source_unsupervised_run_dir"] = str(run_dir)
        f.attrs["source_unsupervised_seg_dir"] = str(seg_dir)
        f.attrs["content_type"] = "unsupervised_segmentation_labels"

    print(f"[OK] H5 for reconstruction saved: {out_h5}")

    return out_h5


def save_npz_for_segmentation_dir(output_case, labels, source_data, run_dir, seg_dir, model_name):
    out_dir = SEGMENTATION_DIR / output_case / model_name

    if out_dir.exists() and UNSUPERVISED_IMPORT_OVERWRITE:
        shutil.rmtree(out_dir)

    ensure_dir(out_dir)

    out_npz = out_dir / f"{output_case}_labels.npz"

    classes = np.unique(labels).astype(np.uint8)

    np.savez_compressed(
        out_npz,
        labels=labels.astype(np.uint8),
        volume=labels.astype(np.uint8),
        source_h5=str(source_data["h5_path"]),
        source_unsupervised_run_dir=str(run_dir),
        source_unsupervised_seg_dir=str(seg_dir),
        model_name=model_name,
        classes=classes,
    )

    metadata = {
        "output_case": output_case,
        "model_name": model_name,
        "source_h5": str(source_data["h5_path"]),
        "source_unsupervised_run_dir": str(run_dir),
        "source_unsupervised_seg_dir": str(seg_dir),
        "shape": list(labels.shape),
        "classes": classes.astype(int).tolist(),
    }

    with open(out_dir / f"{output_case}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] segmentation NPZ saved: {out_npz}")

    return out_npz


# ============================================================
# IMPORT ONE OUTPUT
# ============================================================

def import_one_unsupervised_output(
    source_case,
    run_dir=None,
    output_case=None,
    use_crf=None,
    model_name=None,
    denoising_method=None,
    segments_num=None,
    clusters_num=None,
):
    source_case = str(source_case)

    if use_crf is None:
        use_crf = UNSUPERVISED_IMPORT_USE_CRF

    if model_name is None:
        model_name = UNSUPERVISED_IMPORT_MODEL_NAME

    if run_dir is None:
        run_dir = find_matching_unsupervised_run_dir(
            source_case=source_case,
            denoising_method=denoising_method,
            segments_num=segments_num,
            clusters_num=clusters_num,
        )
    else:
        run_dir = Path(run_dir)

    if output_case is None:
        output_case = make_output_case(
            source_case=source_case,
            segments_num=segments_num,
            clusters_num=clusters_num,
            use_crf=use_crf,
            output_case=None,
        )

    print("============================================================")
    print("[INFO] Importing unsupervised output")
    print(f"[INFO] source_case      : {source_case}")
    print(f"[INFO] output_case      : {output_case}")
    print(f"[INFO] run_dir          : {run_dir}")
    print(f"[INFO] denoising_method : {denoising_method}")
    print(f"[INFO] segments_num     : {segments_num}")
    print(f"[INFO] clusters_num     : {clusters_num}")
    print(f"[INFO] use_crf          : {use_crf}")
    print(f"[INFO] model_name       : {model_name}")

    source_data = load_source_h5(source_case)
    seg_dir = get_segmentation_png_dir(run_dir, use_crf=use_crf)

    labels = read_unsupervised_png_stack(
        seg_dir=seg_dir,
        source_shape=source_data["image"].shape,
    )

    print(f"[INFO] imported labels shape: {labels.shape}")
    print(f"[INFO] imported classes     : {np.unique(labels).tolist()}")

    outputs = {
        "source_case": source_case,
        "output_case": output_case,
        "run_dir": str(run_dir),
        "seg_dir": str(seg_dir),
        "classes": np.unique(labels).astype(int).tolist(),
    }

    if UNSUPERVISED_IMPORT_TO_FOR_RECONSTRUCTION:
        outputs["h5"] = str(
            save_h5_for_reconstruction(
                output_case=output_case,
                labels=labels,
                source_data=source_data,
                run_dir=run_dir,
                seg_dir=seg_dir,
            )
        )

    if UNSUPERVISED_IMPORT_TO_SEGMENTATION_DIR:
        outputs["npz"] = str(
            save_npz_for_segmentation_dir(
                output_case=output_case,
                labels=labels,
                source_data=source_data,
                run_dir=run_dir,
                seg_dir=seg_dir,
                model_name=model_name,
            )
        )

    return outputs


# ============================================================
# IMPORT CONFIG CASES
# ============================================================

def import_from_case_specs(specs):
    imported = []

    for item in specs:
        base_case = item.get("base_case")
        denoising_method = item.get("denoising_method")
        source_case = item.get("source_case")

        source_case = make_source_case(
            base_case=base_case,
            denoising_method=denoising_method,
            source_case=source_case,
        )

        segments_num = item.get("segments_num")
        clusters_num = item.get("clusters_num")
        use_crf = item.get("use_crf", UNSUPERVISED_IMPORT_USE_CRF)
        model_name = item.get("model_name", UNSUPERVISED_IMPORT_MODEL_NAME)

        output_case = make_output_case(
            source_case=source_case,
            segments_num=segments_num,
            clusters_num=clusters_num,
            use_crf=use_crf,
            output_case=item.get("output_case"),
        )

        result = import_one_unsupervised_output(
            source_case=source_case,
            run_dir=item.get("run_dir"),
            output_case=output_case,
            use_crf=use_crf,
            model_name=model_name,
            denoising_method=denoising_method,
            segments_num=segments_num,
            clusters_num=clusters_num,
        )

        imported.append(result)

    return imported


def import_from_legacy_imports(imports):
    imported = []

    for item in imports:
        result = import_one_unsupervised_output(
            source_case=item["source_case"],
            output_case=item.get("output_case"),
            run_dir=item.get("run_dir"),
            use_crf=item.get("use_crf", UNSUPERVISED_IMPORT_USE_CRF),
            model_name=item.get("model_name", UNSUPERVISED_IMPORT_MODEL_NAME),
            denoising_method=item.get("denoising_method"),
            segments_num=item.get("segments_num"),
            clusters_num=item.get("clusters_num"),
        )

        imported.append(result)

    return imported


def import_unsupervised_outputs(case_names=None, imports=None):
    """
    Główna funkcja dla main.py.

    Priorytet:
    1. UNSUPERVISED_OUTPUT_IMPORT_CASES:
       podajesz base_case + denoising_method + seg/clust.
    2. UNSUPERVISED_OUTPUT_IMPORTS:
       stary tryb z ręcznym run_dir.
    3. case_names:
       fallback: bierze pierwszy case i najnowszy run.
    """
    if UNSUPERVISED_OUTPUT_IMPORT_CASES:
        return import_from_case_specs(UNSUPERVISED_OUTPUT_IMPORT_CASES)

    if imports is None:
        imports = UNSUPERVISED_OUTPUT_IMPORTS

    if imports:
        return import_from_legacy_imports(imports)

    case_names = normalize_case_names(case_names)

    if not case_names:
        raise RuntimeError(
            "No case_names provided and both "
            "UNSUPERVISED_OUTPUT_IMPORT_CASES and UNSUPERVISED_OUTPUT_IMPORTS are empty."
        )

    result = import_one_unsupervised_output(
        source_case=case_names[0],
        run_dir=None,
        output_case=None,
        use_crf=UNSUPERVISED_IMPORT_USE_CRF,
        model_name=UNSUPERVISED_IMPORT_MODEL_NAME,
    )

    return [result]