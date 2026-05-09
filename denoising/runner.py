from pathlib import Path

from config import (
    FOR_RECON_DIR,
    DENOISED_DIR,
    IMAGE_KEY,
    DENOISING_METHODS,
    DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION,
    DENOISING_OUTPUT_CASE_SUFFIX,
    RDPAD_ITERATIONS,
    RDPAD_TIMESTEP,
    RDPAD_Q0_MODE,
    RDPAD_Q0_PERCENTILE,
    GAUSSIAN_SIGMA,
    MEDIAN_SIZE,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPATIAL,
    TV_WEIGHT,
    TV_MAX_NUM_ITER,
)

from data_loading.h5_utils import list_h5_files

from denoising.common import (
    load_h5_image_and_poses,
    save_h5_image_and_poses,
    apply_slice_wise,
    copy_h5_to_for_reconstruction,
)

from denoising.rdp_ad import denoise_rdpad_2d

from denoising.classical import (
    denoise_gaussian_2d,
    denoise_median_2d,
    denoise_bilateral_2d,
    denoise_tv_2d,
)


def normalize_case_names(case_names):
    if case_names is None:
        return None

    if isinstance(case_names, str):
        return [case_names]

    return list(case_names)


def normalize_methods(methods):
    if methods is None:
        return ["rdpad"]

    if isinstance(methods, str):
        return [methods]

    return list(methods)


def get_denoise_function(method):
    if method == "rdpad":
        return lambda frame: denoise_rdpad_2d(
            frame,
            iterations=RDPAD_ITERATIONS,
            timestep=RDPAD_TIMESTEP,
            q0_mode=RDPAD_Q0_MODE,
            q0_percentile=RDPAD_Q0_PERCENTILE,
        )

    if method == "gaussian":
        return lambda frame: denoise_gaussian_2d(
            frame,
            sigma=GAUSSIAN_SIGMA,
        )

    if method == "median":
        return lambda frame: denoise_median_2d(
            frame,
            size=MEDIAN_SIZE,
        )

    if method == "bilateral":
        return lambda frame: denoise_bilateral_2d(
            frame,
            sigma_color=BILATERAL_SIGMA_COLOR,
            sigma_spatial=BILATERAL_SIGMA_SPATIAL,
        )

    if method == "tv":
        return lambda frame: denoise_tv_2d(
            frame,
            weight=TV_WEIGHT,
            max_num_iter=TV_MAX_NUM_ITER,
        )

    raise ValueError(
        f"Nieznana metoda denoisingu: {method}. "
        f"Dostępne: rdpad, gaussian, median, bilateral, tv"
    )


def denoise_file(path, case_name, method):
    path = Path(path)
    base = path.stem

    print("============================================================")
    print(f"[INFO] Denoising method: {method}")
    print(f"[INFO] Input H5: {path}")

    img, poses, spacing_xy, resolved_image_key = load_h5_image_and_poses(
        path,
        image_key=IMAGE_KEY,
    )

    fn = get_denoise_function(method)

    denoised = apply_slice_wise(
        img,
        fn,
    )

    out_dir = DENOISED_DIR / case_name / method
    out_dir.mkdir(parents=True, exist_ok=True)

    out_h5 = out_dir / f"{base}_{method}.h5"

    attrs = {
        "denoising_method": method,
        "source_h5": str(path),
    }

    save_h5_image_and_poses(
        path=out_h5,
        image=denoised,
        poses=poses,
        spacing_xy=spacing_xy,
        image_key="img",
        pose_key="poses",
        attrs=attrs,
    )

    print(f"[OK] denoised H5: {out_h5}")

    if DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION:
        out_case_name = f"{case_name}{DENOISING_OUTPUT_CASE_SUFFIX}_{method}"
        out_case_dir = FOR_RECON_DIR / out_case_name
        out_case_dir.mkdir(parents=True, exist_ok=True)

        copied_h5 = out_case_dir / f"{base}_{method}.h5"
        copy_h5_to_for_reconstruction(
            src_h5=out_h5,
            dst_h5=copied_h5,
        )

        print(f"[OK] copied to for_reconstruction: {copied_h5}")

    return out_h5


def denoise_case(case_name, methods=None):
    methods = normalize_methods(methods)

    case_dir = FOR_RECON_DIR / case_name

    if not case_dir.exists():
        print(f"[WARN] Case dir does not exist: {case_dir}")
        return

    files = list_h5_files(case_dir)

    if not files:
        print(f"[WARN] No H5 files found for denoising in: {case_dir}")
        return

    for method in methods:
        for path in files:
            denoise_file(
                path=path,
                case_name=case_name,
                method=method,
            )


def denoise_cases(case_names=None, methods=None):
    case_names = normalize_case_names(case_names)
    methods = normalize_methods(methods if methods is not None else DENOISING_METHODS)

    if case_names is None:
        case_dirs = sorted([p for p in FOR_RECON_DIR.iterdir() if p.is_dir()])
    else:
        case_dirs = [FOR_RECON_DIR / name for name in case_names]

    print("[INFO] Denoising methods:", methods)

    for case_dir in case_dirs:
        if not case_dir.exists():
            print(f"[WARN] Case dir does not exist: {case_dir}")
            continue

        denoise_case(
            case_name=case_dir.name,
            methods=methods,
        )