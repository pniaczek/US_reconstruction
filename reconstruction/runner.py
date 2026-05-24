from pathlib import Path

from config import (
    FOR_RECON_DIR,
    RECONSTRUCTED_DIR,
    RECONSTRUCTION_METHODS,
    RECONSTRUCTION_SAVE_METHOD_SUBDIRS,
    VOXEL_SIZE,
    PIXEL_SPACING,
    BACKGROUND_THRESHOLD,
    CENTER,
    CHUNK_PIXELS,
    IMAGE_KEY,
    N_NEAREST,
    SPLINE_ORDER,
    DISTANCE_EPS,
    DISTANCE_POWER,
    DEVICE,
    VNN_MEAN_N_NEAREST,
    DWR_N_NEAREST,
    DWR_DISTANCE_POWER,
    DWR_DISTANCE_EPS,
    DWR_INTERP,
    DWR_MAX_PLANE_DIST,
)

from data_loading.h5_utils import list_h5_files

from reconstruction.bspline import reconstruct_bspline_for_file
from reconstruction.vnn import reconstruct_voxel_nearest_for_file
from reconstruction.vnn_mean import reconstruct_voxel_nearest_mean_for_file
from reconstruction.vnn_distance_weighted import reconstruct_distance_weighted_for_file


METHODS = {
    "bspline": reconstruct_bspline_for_file,
    "voxel_nearest": reconstruct_voxel_nearest_for_file,
    "voxel_nearest_mean": reconstruct_voxel_nearest_mean_for_file,
    "distance_weighted": reconstruct_distance_weighted_for_file,
}


def normalize_case_names(case_names):
    if case_names is None:
        return None

    if isinstance(case_names, str):
        return [case_names]

    return list(case_names)


def normalize_methods(methods):
    if methods is None:
        return ["bspline"]

    if isinstance(methods, str):
        return [methods]

    return list(methods)


def get_out_dir(case_name, method):
    if RECONSTRUCTION_SAVE_METHOD_SUBDIRS:
        return RECONSTRUCTED_DIR / case_name / method

    return RECONSTRUCTED_DIR / case_name


def reconstruct_file_with_method(path, out_dir, method):
    if method not in METHODS:
        raise ValueError(
            f"Nieznana metoda rekonstrukcji: {method}. "
            f"Dostępne: {list(METHODS.keys())}"
        )

    fn = METHODS[method]

    print("============================================================")
    print(f"[INFO] Reconstruction method: {method}")
    print(f"[INFO] Input file: {path}")
    print(f"[INFO] Output dir: {out_dir}")

    if method == "bspline":
        return fn(
            path=path,
            out_root=out_dir,
            voxel=VOXEL_SIZE,
            pixel_spacing=PIXEL_SPACING,
            background_threshold=BACKGROUND_THRESHOLD,
            center=CENTER,
            chunk_pixels=CHUNK_PIXELS,
            image_key=IMAGE_KEY,
            n_nearest=N_NEAREST,
            spline_order=SPLINE_ORDER,
            distance_eps=DISTANCE_EPS,
            distance_power=DISTANCE_POWER,
            device_str=DEVICE,
        )

    if method == "voxel_nearest":
        return fn(
            path=path,
            out_root=out_dir,
            voxel=VOXEL_SIZE,
            pixel_spacing=PIXEL_SPACING,
            background_threshold=BACKGROUND_THRESHOLD,
            center=CENTER,
            chunk_pixels=CHUNK_PIXELS,
            image_key=IMAGE_KEY,
            device_str=DEVICE,
        )

    if method == "voxel_nearest_mean":
        return fn(
            path=path,
            out_root=out_dir,
            voxel=VOXEL_SIZE,
            pixel_spacing=PIXEL_SPACING,
            background_threshold=BACKGROUND_THRESHOLD,
            center=CENTER,
            chunk_pixels=CHUNK_PIXELS,
            image_key=IMAGE_KEY,
            n_nearest=VNN_MEAN_N_NEAREST,
            device_str=DEVICE,
        )

    if method == "distance_weighted":
        return fn(
            path=path,
            out_root=out_dir,
            voxel=VOXEL_SIZE,
            pixel_spacing=PIXEL_SPACING,
            background_threshold=BACKGROUND_THRESHOLD,
            center=CENTER,
            chunk_pixels=CHUNK_PIXELS,
            image_key=IMAGE_KEY,
            n_nearest=DWR_N_NEAREST,
            distance_power=DWR_DISTANCE_POWER,
            distance_eps=DWR_DISTANCE_EPS,
            interp=DWR_INTERP,
            max_plane_dist=DWR_MAX_PLANE_DIST,
            device_str=DEVICE,
        )


def reconstruct_case(case_name, methods=None):
    methods = normalize_methods(methods)

    case_dir = FOR_RECON_DIR / case_name
    files = list_h5_files(case_dir)

    if not files:
        print(f"[WARN] No H5 files found for case: {case_name}")
        return

    for method in methods:
        out_dir = get_out_dir(case_name, method)
        out_dir.mkdir(parents=True, exist_ok=True)

        for path in files:
            reconstruct_file_with_method(
                path=Path(path),
                out_dir=out_dir,
                method=method,
            )


def reconstruct_cases(case_names=None, methods=None):
    case_names = normalize_case_names(case_names)
    methods = normalize_methods(methods if methods is not None else RECONSTRUCTION_METHODS)

    if case_names is None:
        case_dirs = sorted([p for p in FOR_RECON_DIR.iterdir() if p.is_dir()])
    else:
        case_dirs = [FOR_RECON_DIR / name for name in case_names]

    if not case_dirs:
        print(f"[WARN] No cases found in: {FOR_RECON_DIR}")
        return

    print("[INFO] Reconstruction methods:", methods)

    for case_dir in case_dirs:
        if not case_dir.exists():
            print(f"[WARN] Case dir does not exist: {case_dir}")
            continue

        reconstruct_case(case_dir.name, methods=methods)