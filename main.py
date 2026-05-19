import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from config import (
    CASE_NAMES,
    RUN_DATA_LOADING,
    RUN_DENOISING,
    RUN_SEGMENTATION,
    RUN_RECONSTRUCTION,
    RUN_ISOSURFACE_EXTRACTION,
    RUN_NAPARI_VIS,
    RUN_SIMPLE_VIS,
    VIS_CASE,
    RECONSTRUCT_SEGMENTATION_OUTPUT,
    RUN_EVALUATION,
    DENOISING_METHODS,
)

from data_loading.prepare_dicom import prepare_cases
from denoising.runner import denoise_cases
from segmentation.runner import segment_cases
from reconstruction.runner import reconstruct_cases
from isosurface_extraction.extract import extract_surfaces_for_cases
from visualization.napari_viewer import visualize_case_napari
from visualization.simple_viewer import visualize_case_simple
from evaluation.reconstruction_with_gt import evaluate_reconstruction_cases_with_gt
from evaluation.segmentation_metrics import evaluate_segmentation_cases
from evaluation.no_reference import evaluate_no_reference_cases


def get_first_case(case_names):
    if case_names is None:
        return None

    if isinstance(case_names, str):
        return case_names

    if isinstance(case_names, (list, tuple)) and len(case_names) > 0:
        return case_names[0]

    return None


def main():
    segmentation_output_cases = []

    if RUN_DATA_LOADING:
        prepare_cases(case_names=CASE_NAMES)

    # ------------------------------------------------------------
    # 1. DENOISING
    # ------------------------------------------------------------

    denoised_case_names = CASE_NAMES

    if RUN_DENOISING:
        denoise_cases(case_names=CASE_NAMES)

        denoised_case_names = []
        for case_name in CASE_NAMES:
            for method in DENOISING_METHODS:
                denoised_case_names.append(f"{case_name}_denoised_{method}")

    # ------------------------------------------------------------
    # 2. SEGMENTATION
    # ------------------------------------------------------------

    if RUN_SEGMENTATION:
        segmentation_output_cases = segment_cases(case_names=denoised_case_names)

    # ------------------------------------------------------------
    # 3. RECONSTRUCTION
    #    Robimy DWIE rekonstrukcje:
    #    A) pełną intensywność po denoisingu
    #    B) rekonstrukcję klas po segmentacji
    # ------------------------------------------------------------

    intensity_reconstruction_case_names = denoised_case_names
    segmentation_reconstruction_case_names = []

    if (
        RUN_SEGMENTATION
        and RECONSTRUCT_SEGMENTATION_OUTPUT
        and segmentation_output_cases
    ):
        segmentation_reconstruction_case_names = segmentation_output_cases

    if RUN_RECONSTRUCTION:
        # A. Rekonstrukcja pełnej intensywności:
        reconstruct_cases(case_names=intensity_reconstruction_case_names)

        # B. Rekonstrukcja klas/masek po segmentacji:
        if segmentation_reconstruction_case_names:
            reconstruct_cases(case_names=segmentation_reconstruction_case_names)

    # ------------------------------------------------------------
    # 4. ISOSURFACE
    # ------------------------------------------------------------

    if RUN_ISOSURFACE_EXTRACTION:
        extract_surfaces_for_cases(case_names=segmentation_reconstruction_case_names)

    # ------------------------------------------------------------
    # 5. EVALUATION
    #    A) reconstruction-with-GT tylko dla pełnej intensywności
    #    B) segmentation metrics tylko dla rekonstrukcji klas
    #    C) no-reference najlepiej dla obu typów rekonstrukcji
    # ------------------------------------------------------------

    if RUN_EVALUATION:
        evaluate_reconstruction_cases_with_gt(
            case_names=intensity_reconstruction_case_names
        )

        if segmentation_reconstruction_case_names:
            evaluate_segmentation_cases(
                case_names=segmentation_reconstruction_case_names
            )

        evaluate_no_reference_cases(
            case_names=intensity_reconstruction_case_names + segmentation_reconstruction_case_names
        )

    # ------------------------------------------------------------
    # 6. VISUALIZATION
    # ------------------------------------------------------------

    case_for_vis = VIS_CASE

    if case_for_vis is None:
        if segmentation_reconstruction_case_names:
            case_for_vis = segmentation_reconstruction_case_names[0]
        else:
            case_for_vis = get_first_case(intensity_reconstruction_case_names)

    if RUN_NAPARI_VIS:
        visualize_case_napari(case_name=case_for_vis)

    if RUN_SIMPLE_VIS:
        visualize_case_simple(case_name=case_for_vis)


if __name__ == "__main__":
    main()