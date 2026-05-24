from pathlib import Path
import shutil

import config

import denoising.runner as denoise_runner
import unsupervised_segmentation.runner as unsup_runner
import data_loading.unsupervised_output_loader as unsup_loader
import reconstruction.runner as recon_runner
import evaluation.no_reference as no_ref_eval


# ============================================================
# USER SETTINGS
# ============================================================

BASE_CASE_NAME = "hamzaoran"

DENOISING_METHODS_TO_RUN = [
    "none",       # bez denoisingu
    "rdpad",
    "gaussian",
    "median",
    "bilateral",
    "tv",
]

# Parametry segmentacji unsupervised
SEGMENTS_NUM = [50]
CLUSTERS_NUM = [15]
N_CLASSES = 15

# CRF
USE_CRF = True

# DINO/eigen
IMAGE_DOWNSAMPLE_FACTOR = 7

# PNG normalization
PNG_NORMALIZATION = "percentile"
P_LOW = 1.0
P_HIGH = 99.0

# Nazwy wyników unsupervised będą np.:
# hamzaoran_unsup_raw_seg50_clust15_crf
# hamzaoran_unsup_rdpad_seg50_clust15_crf
UNSUPERVISED_NAME_PREFIX = f"{BASE_CASE_NAME}_unsup"

# Suffix dla denoisingu:
# hamzaoran_denoised_rdpad
DENOISING_OUTPUT_CASE_SUFFIX = "_denoised"

# Rekonstrukcja wszystkich wariantów unsupervised.
# Uwaga: dla etykiet klas najbezpieczniejszy jest voxel_nearest.
# bspline/distance_weighted mogą tworzyć wartości pośrednie między klasami.
RECONSTRUCTION_METHODS_TO_RUN = [
    "voxel_nearest",
    "voxel_nearest_mean",
    "distance_weighted",
    "bspline",
]

# Dla etykiet/klas lepiej dać większy voxel niż przy intensywności,
# bo 94% nonzero może generować ogromne objętości.
VOXEL_SIZE = (0.6, 0.6, 0.6)

# Dataset H5 po imporcie unsupervised ma:
#   img    = labels float32
#   labels = labels uint8
# Rekonstruktor zwykle czyta IMAGE_KEY.
RECONSTRUCTION_IMAGE_KEY = "labels"

BACKGROUND_THRESHOLD = 0.0

# ============================================================
# BEHAVIOR FLAGS
# ============================================================

# Jeżeli True, denoising będzie liczony od nowa.
# Jeżeli False, użyje istniejącego:
# data/for_reconstruction/<base>_denoised_<method>/
FORCE_RECOMPUTE_DENOISING = False

# Jeżeli True, usuwa external PNG dataset przed unsupervised.
OVERWRITE_UNSUPERVISED_DATASET = True

# Jeżeli True, nadpisuje configi external pipeline.
OVERWRITE_UNSUPERVISED_CONFIGS = True

# Jeżeli True, usuwa poprzednie wyniki external unsupervised dla wariantu.
# Jeżeli False i wynik istnieje, unsupervised nie będzie liczone ponownie.
FORCE_RERUN_UNSUPERVISED = False

# Jeżeli True, ponownie importuje PNG segmentacji unsupervised do H5/NPZ projektu.
FORCE_REIMPORT_UNSUPERVISED_OUTPUT = False

# Jeżeli True, usuwa poprzednie rekonstrukcje dla output_case.
FORCE_RECOMPUTE_RECONSTRUCTION = False

# Jeżeli True, usuwa poprzednią ewaluację no-reference dla output_case.
FORCE_RECOMPUTE_EVALUATION = True


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_case_exists(case_name: str):
    case_dir = config.FOR_RECON_DIR / case_name

    if not case_dir.exists():
        raise FileNotFoundError(f"Brak case w for_reconstruction: {case_dir}")

    h5_files = list(case_dir.glob("*.h5")) + list(case_dir.glob("*.hdf5"))

    if not h5_files:
        raise FileNotFoundError(f"Brak plików H5/HDF5 w: {case_dir}")

    return case_dir


def method_part(method: str):
    return "raw" if method == "none" else method


def denoised_case_name(base_case_name: str, method: str):
    return f"{base_case_name}{DENOISING_OUTPUT_CASE_SUFFIX}_{method}"


def external_dataset_case_name(method: str):
    m = method_part(method)

    return (
        f"{UNSUPERVISED_NAME_PREFIX}_"
        f"{m}_"
        f"seg{SEGMENTS_NUM[0]}_"
        f"clust{CLUSTERS_NUM[0]}"
    )


def external_tag(method: str):
    m = method_part(method)

    return (
        f"{m}_"
        f"seg{SEGMENTS_NUM[0]}_"
        f"clust{CLUSTERS_NUM[0]}"
    )


def imported_output_case_name(method: str):
    suffix = "crf" if USE_CRF else "raw"
    return f"{external_dataset_case_name(method)}_{suffix}"


def print_header(title):
    print("\n")
    print("############################################################")
    print(f"# {title}")
    print("############################################################")


def remove_dir_if_exists(path: Path, label: str):
    if path.exists():
        print("============================================================")
        print(f"[CLEAN] {label}:")
        print(f"[CLEAN] {path}")
        shutil.rmtree(path)


# ============================================================
# PATCH CONFIGS
# ============================================================

def patch_denoising_config():
    config.DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION = True
    config.DENOISING_OUTPUT_CASE_SUFFIX = DENOISING_OUTPUT_CASE_SUFFIX

    denoise_runner.DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION = True
    denoise_runner.DENOISING_OUTPUT_CASE_SUFFIX = DENOISING_OUTPUT_CASE_SUFFIX


def patch_unsupervised_config(dataset_case_name: str, tag: str):
    """
    Patchujemy config oraz unsup_runner, bo runner ma wartości importowane przez:
        from config import ...
    """

    # config.py
    config.UNSUPERVISED_CASE_NAME = dataset_case_name
    config.UNSUPERVISED_WANDB_TAG = tag
    config.UNSUPERVISED_N_CLASSES = N_CLASSES
    config.UNSUPERVISED_SEGMENTS_NUM = SEGMENTS_NUM
    config.UNSUPERVISED_CLUSTERS_NUM = CLUSTERS_NUM
    config.UNSUPERVISED_OVERWRITE_DATASET = OVERWRITE_UNSUPERVISED_DATASET
    config.UNSUPERVISED_OVERWRITE_CONFIGS = OVERWRITE_UNSUPERVISED_CONFIGS
    config.UNSUPERVISED_USE_WANDB_DISABLED = True
    config.UNSUPERVISED_PNG_NORMALIZATION = PNG_NORMALIZATION
    config.UNSUPERVISED_P_LOW = P_LOW
    config.UNSUPERVISED_P_HIGH = P_HIGH

    if hasattr(config, "UNSUPERVISED_IMAGE_DOWNSAMPLE_FACTOR"):
        config.UNSUPERVISED_IMAGE_DOWNSAMPLE_FACTOR = IMAGE_DOWNSAMPLE_FACTOR

    # unsupervised_segmentation.runner
    unsup_runner.UNSUPERVISED_CASE_NAME = dataset_case_name
    unsup_runner.UNSUPERVISED_WANDB_TAG = tag
    unsup_runner.UNSUPERVISED_N_CLASSES = N_CLASSES
    unsup_runner.UNSUPERVISED_SEGMENTS_NUM = SEGMENTS_NUM
    unsup_runner.UNSUPERVISED_CLUSTERS_NUM = CLUSTERS_NUM
    unsup_runner.UNSUPERVISED_OVERWRITE_DATASET = OVERWRITE_UNSUPERVISED_DATASET
    unsup_runner.UNSUPERVISED_OVERWRITE_CONFIGS = OVERWRITE_UNSUPERVISED_CONFIGS
    unsup_runner.UNSUPERVISED_USE_WANDB_DISABLED = True
    unsup_runner.UNSUPERVISED_PNG_NORMALIZATION = PNG_NORMALIZATION
    unsup_runner.UNSUPERVISED_P_LOW = P_LOW
    unsup_runner.UNSUPERVISED_P_HIGH = P_HIGH

    if hasattr(unsup_runner, "UNSUPERVISED_IMAGE_DOWNSAMPLE_FACTOR"):
        unsup_runner.UNSUPERVISED_IMAGE_DOWNSAMPLE_FACTOR = IMAGE_DOWNSAMPLE_FACTOR


def patch_unsupervised_loader_config():
    """
    Patchujemy loader, bo on też może mieć stałe zaciągnięte z config.py.
    """

    config.UNSUPERVISED_IMPORT_USE_CRF = USE_CRF
    config.UNSUPERVISED_IMPORT_MODEL_NAME = "unsupervised_crf" if USE_CRF else "unsupervised_raw"
    config.UNSUPERVISED_IMPORT_TO_FOR_RECONSTRUCTION = True
    config.UNSUPERVISED_IMPORT_TO_SEGMENTATION_DIR = True
    config.UNSUPERVISED_IMPORT_OVERWRITE = FORCE_REIMPORT_UNSUPERVISED_OUTPUT

    unsup_loader.UNSUPERVISED_IMPORT_USE_CRF = USE_CRF
    unsup_loader.UNSUPERVISED_IMPORT_MODEL_NAME = "unsupervised_crf" if USE_CRF else "unsupervised_raw"
    unsup_loader.UNSUPERVISED_IMPORT_TO_FOR_RECONSTRUCTION = True
    unsup_loader.UNSUPERVISED_IMPORT_TO_SEGMENTATION_DIR = True
    unsup_loader.UNSUPERVISED_IMPORT_OVERWRITE = FORCE_REIMPORT_UNSUPERVISED_OUTPUT


def patch_reconstruction_config():
    config.RECONSTRUCTION_METHODS = RECONSTRUCTION_METHODS_TO_RUN
    config.VOXEL_SIZE = VOXEL_SIZE
    config.IMAGE_KEY = RECONSTRUCTION_IMAGE_KEY
    config.BACKGROUND_THRESHOLD = BACKGROUND_THRESHOLD
    config.RECONSTRUCTION_SAVE_METHOD_SUBDIRS = True

    recon_runner.RECONSTRUCTION_METHODS = RECONSTRUCTION_METHODS_TO_RUN
    recon_runner.VOXEL_SIZE = VOXEL_SIZE
    recon_runner.IMAGE_KEY = RECONSTRUCTION_IMAGE_KEY
    recon_runner.BACKGROUND_THRESHOLD = BACKGROUND_THRESHOLD
    recon_runner.RECONSTRUCTION_SAVE_METHOD_SUBDIRS = True


def patch_evaluation_config():
    config.EVAL_RECON_METHODS = RECONSTRUCTION_METHODS_TO_RUN
    config.EVAL_SEGMENTATION_MODEL_NAME = "unsupervised_crf" if USE_CRF else "unsupervised_raw"
    config.EVAL_NO_REFERENCE_USE_MASK = True
    config.EVAL_OBJECT_THRESHOLD = 0.0
    config.EVAL_BACKGROUND_THRESHOLD = 0.0

    no_ref_eval.EVAL_RECON_METHODS = RECONSTRUCTION_METHODS_TO_RUN
    no_ref_eval.EVAL_SEGMENTATION_MODEL_NAME = "unsupervised_crf" if USE_CRF else "unsupervised_raw"
    no_ref_eval.EVAL_NO_REFERENCE_USE_MASK = True
    no_ref_eval.EVAL_OBJECT_THRESHOLD = 0.0
    no_ref_eval.EVAL_BACKGROUND_THRESHOLD = 0.0


# ============================================================
# DENOISING
# ============================================================

def run_denoising_if_needed(base_case_name: str, method: str):
    if method == "none":
        ensure_case_exists(base_case_name)
        return base_case_name

    out_case_name = denoised_case_name(base_case_name, method)
    out_case_dir = config.FOR_RECON_DIR / out_case_name

    if out_case_dir.exists() and not FORCE_RECOMPUTE_DENOISING:
        print("============================================================")
        print("[SKIP] Denoised case already exists:")
        print(f"[SKIP] {out_case_dir}")
        ensure_case_exists(out_case_name)
        return out_case_name

    if out_case_dir.exists() and FORCE_RECOMPUTE_DENOISING:
        remove_dir_if_exists(out_case_dir, "Removing previous denoised case")

    print("============================================================")
    print(f"[RUN] Denoising: {method}")
    print(f"[RUN] Base case : {base_case_name}")
    print(f"[RUN] Output    : {out_case_name}")

    patch_denoising_config()

    denoise_runner.denoise_case(
        case_name=base_case_name,
        methods=[method],
    )

    ensure_case_exists(out_case_name)

    return out_case_name


# ============================================================
# UNSUPERVISED OUTPUT DISCOVERY
# ============================================================

def external_dataset_dir(dataset_case_name: str):
    return (
        config.UNSUPERVISED_REPO_DIR
        / "data"
        / config.UNSUPERVISED_DATASET_NAME
        / dataset_case_name
    )


def external_output_base_dir(dataset_case_name: str, tag: str):
    return (
        config.UNSUPERVISED_REPO_DIR
        / "deep-spectral-segmentation"
        / config.UNSUPERVISED_DATASET_NAME
        / dataset_case_name
        / "main"
        / tag
    )


def has_raw_or_crf_segmaps(run_dir: Path):
    raw_dir = run_dir / "semantic_segmentations" / "laplacian" / "segmaps"
    crf_dir = run_dir / "semantic_segmentations" / "laplacian" / "crf_segmaps"

    if USE_CRF and crf_dir.exists() and list(crf_dir.glob("*.png")):
        return True

    if raw_dir.exists() and list(raw_dir.glob("*.png")):
        return True

    return False


def find_existing_unsupervised_run(dataset_case_name: str, tag: str):
    """
    Szuka istniejącego runu external pipeline.
    Jeśli znajdzie segmaps/crf_segmaps, zwraca najnowszy katalog runu.
    """

    base = external_output_base_dir(dataset_case_name, tag)

    if not base.exists():
        return None

    candidates = []

    for p in base.rglob("*"):
        if p.is_dir() and has_raw_or_crf_segmaps(p):
            candidates.append(p)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)

    return candidates[0]


def clean_external_dataset(dataset_case_name: str):
    remove_dir_if_exists(
        external_dataset_dir(dataset_case_name),
        "Removing external PNG dataset",
    )


def clean_external_outputs(dataset_case_name: str, tag: str):
    remove_dir_if_exists(
        external_output_base_dir(dataset_case_name, tag),
        "Removing unsupervised outputs",
    )


# ============================================================
# UNSUPERVISED SEGMENTATION
# ============================================================

def run_unsupervised_if_needed(source_case_name: str, method: str):
    dataset_case_name = external_dataset_case_name(method)
    tag = external_tag(method)

    patch_unsupervised_config(
        dataset_case_name=dataset_case_name,
        tag=tag,
    )

    existing_run = find_existing_unsupervised_run(
        dataset_case_name=dataset_case_name,
        tag=tag,
    )

    if existing_run is not None and not FORCE_RERUN_UNSUPERVISED:
        print("============================================================")
        print("[SKIP] Existing unsupervised segmentation found:")
        print(f"[SKIP] run_dir: {existing_run}")
        return existing_run

    if FORCE_RERUN_UNSUPERVISED:
        clean_external_outputs(dataset_case_name, tag)

    if OVERWRITE_UNSUPERVISED_DATASET:
        clean_external_dataset(dataset_case_name)

    print("============================================================")
    print("[RUN] Unsupervised segmentation")
    print(f"[RUN] source_case_name        : {source_case_name}")
    print(f"[RUN] dataset_case_name      : {dataset_case_name}")
    print(f"[RUN] tag                    : {tag}")
    print(f"[RUN] SEGMENTS_NUM           : {SEGMENTS_NUM}")
    print(f"[RUN] CLUSTERS_NUM           : {CLUSTERS_NUM}")
    print(f"[RUN] USE_CRF                : {USE_CRF}")
    print(f"[RUN] IMAGE_DOWNSAMPLE_FACTOR: {IMAGE_DOWNSAMPLE_FACTOR}")
    print("============================================================")

    unsup_runner.run_unsupervised_segmentation(
        case_names=[source_case_name],
    )

    run_dir = find_existing_unsupervised_run(
        dataset_case_name=dataset_case_name,
        tag=tag,
    )

    if run_dir is None:
        raise RuntimeError(
            "Unsupervised pipeline finished, but no valid segmaps/crf_segmaps were found under:\n"
            f"{external_output_base_dir(dataset_case_name, tag)}"
        )

    return run_dir


# ============================================================
# IMPORT UNSUPERVISED OUTPUT
# ============================================================

def imported_case_is_ready(output_case: str):
    h5_dir = config.FOR_RECON_DIR / output_case
    seg_dir = config.SEGMENTATION_DIR / output_case

    h5_files = list(h5_dir.glob("*.h5")) + list(h5_dir.glob("*.hdf5")) if h5_dir.exists() else []
    npz_files = list(seg_dir.rglob("*.npz")) if seg_dir.exists() else []

    return bool(h5_files) and bool(npz_files)


def import_unsupervised_if_needed(source_case_name: str, method: str, run_dir: Path):
    output_case = imported_output_case_name(method)
    model_name = "unsupervised_crf" if USE_CRF else "unsupervised_raw"

    patch_unsupervised_loader_config()

    if imported_case_is_ready(output_case) and not FORCE_REIMPORT_UNSUPERVISED_OUTPUT:
        print("============================================================")
        print("[SKIP] Imported unsupervised output already exists:")
        print(f"[SKIP] output_case: {output_case}")
        return output_case

    print("============================================================")
    print("[RUN] Import unsupervised output")
    print(f"[RUN] source_case: {source_case_name}")
    print(f"[RUN] output_case: {output_case}")
    print(f"[RUN] run_dir    : {run_dir}")
    print(f"[RUN] use_crf    : {USE_CRF}")
    print("============================================================")

    result = unsup_loader.import_one_unsupervised_output(
        source_case=source_case_name,
        run_dir=run_dir,
        output_case=output_case,
        use_crf=USE_CRF,
        model_name=model_name,
    )

    if "output_case" in result:
        return result["output_case"]

    return output_case


# ============================================================
# RECONSTRUCTION
# ============================================================

def reconstruction_method_has_output(case_name: str, method: str):
    method_dir = config.RECONSTRUCTED_DIR / case_name / method

    if not method_dir.exists():
        return False

    return bool(list(method_dir.glob("*.npz"))) or bool(list(method_dir.glob("*.vti")))


def all_reconstructions_exist(case_name: str):
    return all(
        reconstruction_method_has_output(case_name, method)
        for method in RECONSTRUCTION_METHODS_TO_RUN
    )


def clean_reconstruction_outputs(case_name: str):
    remove_dir_if_exists(
        config.RECONSTRUCTED_DIR / case_name,
        "Removing previous reconstruction outputs",
    )


def run_reconstruction_if_needed(output_case: str):
    patch_reconstruction_config()

    if FORCE_RECOMPUTE_RECONSTRUCTION:
        clean_reconstruction_outputs(output_case)

    if all_reconstructions_exist(output_case) and not FORCE_RECOMPUTE_RECONSTRUCTION:
        print("============================================================")
        print("[SKIP] All reconstruction outputs already exist:")
        print(f"[SKIP] case: {output_case}")
        print(f"[SKIP] methods: {RECONSTRUCTION_METHODS_TO_RUN}")
        return

    print("============================================================")
    print("[RUN] Reconstruction")
    print(f"[RUN] case   : {output_case}")
    print(f"[RUN] methods: {RECONSTRUCTION_METHODS_TO_RUN}")
    print(f"[RUN] key    : {RECONSTRUCTION_IMAGE_KEY}")
    print(f"[RUN] voxel  : {VOXEL_SIZE}")
    print("============================================================")

    recon_runner.reconstruct_cases(
        case_names=[output_case],
    )


# ============================================================
# EVALUATION
# ============================================================

def clean_evaluation_outputs(case_name: str):
    remove_dir_if_exists(
        config.EVALUATION_DIR / case_name / "no_reference",
        "Removing previous no-reference evaluation outputs",
    )


def run_evaluation(output_case: str):
    patch_evaluation_config()

    if FORCE_RECOMPUTE_EVALUATION:
        clean_evaluation_outputs(output_case)

    print("============================================================")
    print("[RUN] No-reference evaluation")
    print(f"[RUN] case   : {output_case}")
    print(f"[RUN] methods: {RECONSTRUCTION_METHODS_TO_RUN}")
    print("============================================================")

    no_ref_eval.evaluate_no_reference_cases(
        case_names=[output_case],
    )


# ============================================================
# ONE VARIANT
# ============================================================

def run_one_variant(base_case_name: str, method: str):
    print_header(f"VARIANT: {method}")

    source_case_name = run_denoising_if_needed(
        base_case_name=base_case_name,
        method=method,
    )

    run_dir = run_unsupervised_if_needed(
        source_case_name=source_case_name,
        method=method,
    )

    output_case = import_unsupervised_if_needed(
        source_case_name=source_case_name,
        method=method,
        run_dir=run_dir,
    )

    run_reconstruction_if_needed(
        output_case=output_case,
    )

    run_evaluation(
        output_case=output_case,
    )

    print("============================================================")
    print("[OK] Variant finished")
    print(f"[OK] method      : {method}")
    print(f"[OK] source_case : {source_case_name}")
    print(f"[OK] output_case : {output_case}")
    print(f"[OK] run_dir     : {run_dir}")
    print("============================================================")

    return {
        "method": method,
        "source_case": source_case_name,
        "output_case": output_case,
        "run_dir": str(run_dir),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_case_exists(BASE_CASE_NAME)

    print("============================================================")
    print("[INFO] Running all denoising -> unsupervised -> reconstruction -> evaluation variants")
    print(f"[INFO] Base case      : {BASE_CASE_NAME}")
    print(f"[INFO] Methods        : {DENOISING_METHODS_TO_RUN}")
    print(f"[INFO] Segments       : {SEGMENTS_NUM}")
    print(f"[INFO] Clusters       : {CLUSTERS_NUM}")
    print(f"[INFO] Recon methods  : {RECONSTRUCTION_METHODS_TO_RUN}")
    print(f"[INFO] USE_CRF        : {USE_CRF}")
    print("============================================================")

    results = []

    for method in DENOISING_METHODS_TO_RUN:
        result = run_one_variant(
            base_case_name=BASE_CASE_NAME,
            method=method,
        )
        results.append(result)

    print("\n")
    print("============================================================")
    print("[OK] All variants finished")
    print("============================================================")

    for item in results:
        print(
            f"[RESULT] method={item['method']} | "
            f"source_case={item['source_case']} | "
            f"output_case={item['output_case']} | "
            f"run_dir={item['run_dir']}"
        )


if __name__ == "__main__":
    main()