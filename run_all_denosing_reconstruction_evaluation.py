from pathlib import Path
import shutil

import config

import denoising.runner as denoise_runner
import reconstruction.runner as recon_runner

from evaluation.reconstruction_with_gt import evaluate_reconstruction_cases_with_gt
from evaluation.no_reference import evaluate_no_reference_cases


# ============================================================
# USER SETTINGS
# ============================================================

# Jeśli None, skrypt automatycznie znajdzie bazowe case'y w data/for_reconstruction/.
# Jeżeli chcesz ręcznie:
# BASE_CASE_NAMES = ["hamzaoran", "kosartur", "friebemichael"]
BASE_CASE_NAMES = None

DENOISING_METHODS_TO_RUN = [
    "none",       # bez denoisingu
    "rdpad",
    "gaussian",
    "median",
    "bilateral",
    "tv",
]

RECONSTRUCTION_METHODS_TO_RUN = [
    "bspline",
    "voxel_nearest",
    "voxel_nearest_mean",
    "distance_weighted",
]

# Dla zwykłych danych intensywnościowych zwykle może być mniejszy voxel niż dla klas.
VOXEL_SIZE = (0.15, 0.15, 0.15)

# Dla zwykłych danych intensywnościowych zostaw None, wtedy rekonstruktor sam wybierze img.
RECONSTRUCTION_IMAGE_KEY = None

BACKGROUND_THRESHOLD = 0.0

# Jeśli True, denoising liczony od nowa.
FORCE_RECOMPUTE_DENOISING = False

# Jeśli True, usuwa stare rekonstrukcje danego case'a i liczy od nowa.
FORCE_RECOMPUTE_RECONSTRUCTION = False

# Jeśli True, usuwa stare foldery ewaluacji i liczy od nowa.
FORCE_RECOMPUTE_EVALUATION = True

# Czy liczyć metryki względem GT.
# Działa tylko, jeśli masz odpowiednie dane w data/ground_truth/<base_case>/.
RUN_EVAL_RECONSTRUCTION_WITH_GT = True

# Czy liczyć metryki bez referencji.
RUN_EVAL_NO_REFERENCE = True

# Suffix dla denoisingu:
# hamzaoran_denoised_rdpad
DENOISING_OUTPUT_CASE_SUFFIX = "_denoised"


# ============================================================
# HELPERS
# ============================================================

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


def has_h5(case_dir: Path):
    if not case_dir.exists():
        return False

    h5_files = list(case_dir.glob("*.h5")) + list(case_dir.glob("*.hdf5"))
    return len(h5_files) > 0


def ensure_case_exists(case_name: str):
    case_dir = config.FOR_RECON_DIR / case_name

    if not case_dir.exists():
        raise FileNotFoundError(f"Brak case w for_reconstruction: {case_dir}")

    if not has_h5(case_dir):
        raise FileNotFoundError(f"Brak plików H5/HDF5 w: {case_dir}")

    return case_dir


def is_generated_case_name(name: str):
    """
    Odfiltrowuje case'y wynikowe, żeby przy BASE_CASE_NAMES=None
    nie brać ponownie denoised/seg/unsup jako bazowych.
    """
    bad_fragments = [
        "_denoised_",
        "_seg_",
        "_unsup_",
        "_seg",
        "_crf",
    ]

    return any(fragment in name for fragment in bad_fragments)


def discover_base_cases():
    case_dirs = sorted([p for p in config.FOR_RECON_DIR.iterdir() if p.is_dir()])

    base_cases = []

    for case_dir in case_dirs:
        name = case_dir.name

        if is_generated_case_name(name):
            continue

        if not has_h5(case_dir):
            continue

        base_cases.append(name)

    return base_cases


def normalize_base_case_names(case_names):
    if case_names is None:
        return discover_base_cases()

    if isinstance(case_names, str):
        return [case_names]

    return list(case_names)


def denoised_case_name(base_case_name: str, method: str):
    return f"{base_case_name}{DENOISING_OUTPUT_CASE_SUFFIX}_{method}"


def patch_denoising_config():
    """
    Denoising ma pracować na zwykłym obrazie intensywnościowym,
    a nie na labels z unsupervised segmentation.
    """
    config.DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION = True
    config.DENOISING_OUTPUT_CASE_SUFFIX = DENOISING_OUTPUT_CASE_SUFFIX

    # Kluczowe: zwykłe H5 mają obraz pod img, nie labels.
    config.IMAGE_KEY = None

    denoise_runner.DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION = True
    denoise_runner.DENOISING_OUTPUT_CASE_SUFFIX = DENOISING_OUTPUT_CASE_SUFFIX

    # denoising.runner zrobił wcześniej: from config import IMAGE_KEY,
    # więc trzeba spatchować też zmienną w module runnera.
    denoise_runner.IMAGE_KEY = None


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
    config.EVAL_NO_REFERENCE_USE_MASK = True
    config.EVAL_OBJECT_THRESHOLD = 0.0
    config.EVAL_BACKGROUND_THRESHOLD = 0.0


# ============================================================
# DENOISING
# ============================================================

def run_denoising_if_needed(base_case_name: str, method: str):
    if method == "none":
        ensure_case_exists(base_case_name)
        return base_case_name

    output_case = denoised_case_name(base_case_name, method)
    output_dir = config.FOR_RECON_DIR / output_case

    if output_dir.exists() and has_h5(output_dir) and not FORCE_RECOMPUTE_DENOISING:
        print("============================================================")
        print("[SKIP] Denoised case already exists:")
        print(f"[SKIP] {output_dir}")
        return output_case

    if output_dir.exists() and FORCE_RECOMPUTE_DENOISING:
        remove_dir_if_exists(output_dir, "Removing previous denoised case")

    print("============================================================")
    print("[RUN] Denoising")
    print(f"[RUN] base case : {base_case_name}")
    print(f"[RUN] method    : {method}")
    print(f"[RUN] output    : {output_case}")
    print("============================================================")

    patch_denoising_config()

    denoise_runner.denoise_case(
        case_name=base_case_name,
        methods=[method],
    )

    ensure_case_exists(output_case)

    return output_case


# ============================================================
# RECONSTRUCTION
# ============================================================

def reconstruction_method_has_output(case_name: str, method: str):
    method_dir = config.RECONSTRUCTED_DIR / case_name / method

    if not method_dir.exists():
        return False

    has_npz = bool(list(method_dir.glob("*.npz")))
    has_vti = bool(list(method_dir.glob("*.vti")))

    return has_npz or has_vti


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


def run_reconstruction_if_needed(case_name: str):
    patch_reconstruction_config()

    if FORCE_RECOMPUTE_RECONSTRUCTION:
        clean_reconstruction_outputs(case_name)

    if all_reconstructions_exist(case_name) and not FORCE_RECOMPUTE_RECONSTRUCTION:
        print("============================================================")
        print("[SKIP] All reconstruction outputs already exist:")
        print(f"[SKIP] case   : {case_name}")
        print(f"[SKIP] methods: {RECONSTRUCTION_METHODS_TO_RUN}")
        return

    print("============================================================")
    print("[RUN] Reconstruction")
    print(f"[RUN] case   : {case_name}")
    print(f"[RUN] methods: {RECONSTRUCTION_METHODS_TO_RUN}")
    print(f"[RUN] key    : {RECONSTRUCTION_IMAGE_KEY}")
    print(f"[RUN] voxel  : {VOXEL_SIZE}")
    print("============================================================")

    recon_runner.reconstruct_cases(
        case_names=[case_name],
    )


# ============================================================
# EVALUATION
# ============================================================

def clean_evaluation_outputs(case_name: str):
    remove_dir_if_exists(
        config.EVALUATION_DIR / case_name,
        "Removing previous evaluation outputs",
    )


def run_evaluation(case_name: str):
    patch_evaluation_config()

    if FORCE_RECOMPUTE_EVALUATION:
        clean_evaluation_outputs(case_name)

    if RUN_EVAL_RECONSTRUCTION_WITH_GT:
        print("============================================================")
        print("[RUN] Reconstruction-with-GT evaluation")
        print(f"[RUN] case   : {case_name}")
        print(f"[RUN] methods: {RECONSTRUCTION_METHODS_TO_RUN}")
        print("============================================================")

        try:
            evaluate_reconstruction_cases_with_gt(
                case_names=[case_name],
            )
        except Exception as e:
            print(f"[WARN] Reconstruction-with-GT evaluation failed for {case_name}: {e}")

    if RUN_EVAL_NO_REFERENCE:
        print("============================================================")
        print("[RUN] No-reference evaluation")
        print(f"[RUN] case   : {case_name}")
        print(f"[RUN] methods: {RECONSTRUCTION_METHODS_TO_RUN}")
        print("============================================================")

        try:
            evaluate_no_reference_cases(
                case_names=[case_name],
            )
        except Exception as e:
            print(f"[WARN] No-reference evaluation failed for {case_name}: {e}")


# ============================================================
# ONE VARIANT
# ============================================================

def run_one_variant(base_case_name: str, method: str):
    print_header(f"CASE: {base_case_name} | METHOD: {method}")

    case_for_reconstruction = run_denoising_if_needed(
        base_case_name=base_case_name,
        method=method,
    )

    run_reconstruction_if_needed(
        case_name=case_for_reconstruction,
    )

    run_evaluation(
        case_name=case_for_reconstruction,
    )

    print("============================================================")
    print("[OK] Variant finished")
    print(f"[OK] base case     : {base_case_name}")
    print(f"[OK] method        : {method}")
    print(f"[OK] output case   : {case_for_reconstruction}")
    print("============================================================")

    return {
        "base_case": base_case_name,
        "method": method,
        "output_case": case_for_reconstruction,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    base_case_names = normalize_base_case_names(BASE_CASE_NAMES)

    if not base_case_names:
        raise RuntimeError(
            "Nie znaleziono żadnych bazowych case'ów w data/for_reconstruction/."
        )

    print("============================================================")
    print("[INFO] Running all raw/denoised reconstruction + evaluation variants")
    print(f"[INFO] Base cases     : {base_case_names}")
    print(f"[INFO] Denoising      : {DENOISING_METHODS_TO_RUN}")
    print(f"[INFO] Recon methods  : {RECONSTRUCTION_METHODS_TO_RUN}")
    print(f"[INFO] Voxel size     : {VOXEL_SIZE}")
    print(f"[INFO] Image key      : {RECONSTRUCTION_IMAGE_KEY}")
    print("============================================================")

    results = []

    for base_case_name in base_case_names:
        ensure_case_exists(base_case_name)

        for method in DENOISING_METHODS_TO_RUN:
            result = run_one_variant(
                base_case_name=base_case_name,
                method=method,
            )
            results.append(result)

    print("\n")
    print("============================================================")
    print("[OK] All variants finished")
    print("============================================================")

    for item in results:
        print(
            f"[RESULT] base_case={item['base_case']} | "
            f"method={item['method']} | "
            f"output_case={item['output_case']}"
        )


if __name__ == "__main__":
    main()