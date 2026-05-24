import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from config import (
    CASE_NAMES,

    RUN_DATA_LOADING,
    RUN_DENOISING,
    RUN_SEGMENTATION,
    RUN_UNSUPERVISED_SEGMENTATION,
    RUN_UNSUPERVISED_OUTPUT_LOADING,
    RUN_RECONSTRUCTION,
    RUN_RECONSTRUCT_INTENSITY,
    RUN_ISOSURFACE_EXTRACTION,
    RUN_NAPARI_VIS,
    RUN_SIMPLE_VIS,
    RUN_EVALUATION,

    RUN_EVAL_RECONSTRUCTION_WITH_GT,
    RUN_EVAL_SEGMENTATION,
    RUN_EVAL_NO_REFERENCE,

    VIS_CASE,
    RECONSTRUCT_SEGMENTATION_OUTPUT,
    DENOISING_METHODS,
)

from data_loading.prepare_dicom import prepare_cases
from data_loading.unsupervised_output_loader import import_unsupervised_outputs

from denoising.runner import denoise_cases
from segmentation.runner import segment_cases
from unsupervised_segmentation.runner import run_unsupervised_segmentation

from reconstruction.runner import reconstruct_cases
from isosurface_extraction.extract import extract_surfaces_for_cases

from visualization.napari_viewer import visualize_case_napari
from visualization.simple_viewer import visualize_case_simple

from evaluation.reconstruction_with_gt import evaluate_reconstruction_cases_with_gt
from evaluation.segmentation_metrics import evaluate_segmentation_cases
from evaluation.no_reference import evaluate_no_reference_cases


def normalize_case_names(case_names):
    if case_names is None:
        return []

    if isinstance(case_names, str):
        return [case_names]

    return list(case_names)


def get_first_case(case_names):
    case_names = normalize_case_names(case_names)

    if not case_names:
        return None

    return case_names[0]


def build_denoised_case_names(base_case_names):
    out = []

    for case_name in normalize_case_names(base_case_names):
        for method in DENOISING_METHODS:
            out.append(f"{case_name}_denoised_{method}")

    return out


def print_case_group(title, case_names):
    print("============================================================")
    print(title)

    case_names = normalize_case_names(case_names)

    if not case_names:
        print("  [empty]")
        return

    for name in case_names:
        print(f"  - {name}")


def main():
    base_case_names = normalize_case_names(CASE_NAMES)

    if not base_case_names:
        raise RuntimeError("CASE_NAMES is empty.")

    # ------------------------------------------------------------
    # 0. DATA LOADING
    # ------------------------------------------------------------

    if RUN_DATA_LOADING:
        prepare_cases(case_names=base_case_names)

    # ------------------------------------------------------------
    # 1. DENOISING
    # ------------------------------------------------------------

    intensity_case_names = list(base_case_names)

    if RUN_DENOISING:
        denoise_cases(case_names=base_case_names)
        intensity_case_names = build_denoised_case_names(base_case_names)

    print_case_group(
        "[INFO] Intensity/input cases for downstream steps:",
        intensity_case_names,
    )

    # ------------------------------------------------------------
    # 2. SUPERVISED SEGMENTATION
    # ------------------------------------------------------------

    supervised_segmentation_cases = []

    if RUN_SEGMENTATION:
        supervised_segmentation_cases = segment_cases(
            case_names=intensity_case_names,
        )

        if supervised_segmentation_cases is None:
            supervised_segmentation_cases = []

    print_case_group(
        "[INFO] Supervised segmentation output cases:",
        supervised_segmentation_cases,
    )

    # ------------------------------------------------------------
    # 3. UNSUPERVISED SEGMENTATION - EXTERNAL PIPELINE
    # ------------------------------------------------------------

    if RUN_UNSUPERVISED_SEGMENTATION:
        run_unsupervised_segmentation(case_names=intensity_case_names)

    # ------------------------------------------------------------
    # 4. IMPORT UNSUPERVISED OUTPUT DO FORMATU PROJEKTU
    # ------------------------------------------------------------

    unsupervised_segmentation_cases = []

    if RUN_UNSUPERVISED_OUTPUT_LOADING:
        imported_unsup = import_unsupervised_outputs(
            case_names=intensity_case_names,
        )

        if imported_unsup is None:
            imported_unsup = []

        unsupervised_segmentation_cases = [
            item["output_case"]
            for item in imported_unsup
            if "output_case" in item
        ]

    print_case_group(
        "[INFO] Unsupervised imported output cases:",
        unsupervised_segmentation_cases,
    )

    # ------------------------------------------------------------
    # 5. RECONSTRUCTION
    # ------------------------------------------------------------

    intensity_reconstruction_cases = []

    if RUN_RECONSTRUCT_INTENSITY:
        intensity_reconstruction_cases = list(intensity_case_names)

    supervised_label_reconstruction_cases = []

    if (
        RUN_SEGMENTATION
        and RECONSTRUCT_SEGMENTATION_OUTPUT
        and supervised_segmentation_cases
    ):
        supervised_label_reconstruction_cases = list(supervised_segmentation_cases)

    unsupervised_label_reconstruction_cases = list(unsupervised_segmentation_cases)

    label_reconstruction_cases = (
        supervised_label_reconstruction_cases
        + unsupervised_label_reconstruction_cases
    )

    print_case_group(
        "[INFO] Intensity reconstruction cases:",
        intensity_reconstruction_cases,
    )
    print_case_group(
        "[INFO] Supervised label reconstruction cases:",
        supervised_label_reconstruction_cases,
    )
    print_case_group(
        "[INFO] Unsupervised label reconstruction cases:",
        unsupervised_label_reconstruction_cases,
    )

    if RUN_RECONSTRUCTION:
        if intensity_reconstruction_cases:
            reconstruct_cases(case_names=intensity_reconstruction_cases)

        if label_reconstruction_cases:
            reconstruct_cases(case_names=label_reconstruction_cases)

        if not intensity_reconstruction_cases and not label_reconstruction_cases:
            print("[WARN] RUN_RECONSTRUCTION=True, but no reconstruction cases found.")

    # ------------------------------------------------------------
    # 6. ISOSURFACE EXTRACTION
    # ------------------------------------------------------------

    if RUN_ISOSURFACE_EXTRACTION:
        if label_reconstruction_cases:
            extract_surfaces_for_cases(case_names=label_reconstruction_cases)
        else:
            print("[WARN] RUN_ISOSURFACE_EXTRACTION=True, but no label cases found.")

    # ------------------------------------------------------------
    # 7. EVALUATION
    # ------------------------------------------------------------

    if RUN_EVALUATION:
        if RUN_EVAL_RECONSTRUCTION_WITH_GT:
            if intensity_reconstruction_cases:
                evaluate_reconstruction_cases_with_gt(
                    case_names=intensity_reconstruction_cases,
                )
            else:
                print(
                    "[INFO] Skipping reconstruction-with-GT evaluation: "
                    "no intensity reconstruction cases."
                )

        if RUN_EVAL_SEGMENTATION:
            if supervised_label_reconstruction_cases:
                evaluate_segmentation_cases(
                    case_names=supervised_label_reconstruction_cases,
                )
            else:
                print(
                    "[INFO] Skipping segmentation metrics: "
                    "no supervised segmentation cases."
                )

        if RUN_EVAL_NO_REFERENCE:
            no_reference_cases = (
                intensity_reconstruction_cases
                + supervised_label_reconstruction_cases
                + unsupervised_label_reconstruction_cases
            )

            if no_reference_cases:
                evaluate_no_reference_cases(
                    case_names=no_reference_cases,
                )
            else:
                print("[WARN] No cases for no-reference evaluation.")

    # ------------------------------------------------------------
    # 8. VISUALIZATION
    # ------------------------------------------------------------

    case_for_vis = VIS_CASE

    if case_for_vis is None:
        if unsupervised_label_reconstruction_cases:
            case_for_vis = unsupervised_label_reconstruction_cases[0]
        elif supervised_label_reconstruction_cases:
            case_for_vis = supervised_label_reconstruction_cases[0]
        elif intensity_reconstruction_cases:
            case_for_vis = intensity_reconstruction_cases[0]
        else:
            case_for_vis = get_first_case(base_case_names)

    if RUN_NAPARI_VIS:
        visualize_case_napari(case_name=case_for_vis)

    if RUN_SIMPLE_VIS:
        visualize_case_simple(case_name=case_for_vis)


if __name__ == "__main__":
    main()