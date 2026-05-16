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
)

from data_loading.prepare_dicom import prepare_cases
from denoising.runner import denoise_cases
from segmentation.runner import segment_cases
from reconstruction.runner import reconstruct_cases
from isosurface_extraction.extract import extract_surfaces_for_cases
from visualization.napari_viewer import visualize_case_napari
from visualization.simple_viewer import visualize_case_simple


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

    if RUN_DENOISING:
        denoise_cases(case_names=CASE_NAMES)

    if RUN_SEGMENTATION:
        segmentation_output_cases = segment_cases(case_names=CASE_NAMES)

    reconstruction_case_names = CASE_NAMES

    if (
        RUN_SEGMENTATION
        and RECONSTRUCT_SEGMENTATION_OUTPUT
        and segmentation_output_cases
    ):
        reconstruction_case_names = segmentation_output_cases

    if RUN_RECONSTRUCTION:
        reconstruct_cases(case_names=reconstruction_case_names)

    if RUN_ISOSURFACE_EXTRACTION:
        extract_surfaces_for_cases(case_names=reconstruction_case_names)

    case_for_vis = VIS_CASE

    if case_for_vis is None:
        if (
            RUN_SEGMENTATION
            and RECONSTRUCT_SEGMENTATION_OUTPUT
            and segmentation_output_cases
        ):
            case_for_vis = segmentation_output_cases[0]
        else:
            case_for_vis = get_first_case(CASE_NAMES)

    if RUN_NAPARI_VIS:
        visualize_case_napari(case_name=case_for_vis)

    if RUN_SIMPLE_VIS:
        visualize_case_simple(case_name=case_for_vis)


if __name__ == "__main__":
    main()