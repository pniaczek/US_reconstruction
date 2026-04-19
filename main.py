import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from config import (
    CASE_NAMES,
    RUN_DATA_LOADING,
    RUN_RECONSTRUCTION,
    RUN_ISOSURFACE_EXTRACTION,
    RUN_NAPARI_VIS,
    RUN_SIMPLE_VIS,
    VIS_CASE,
)
from data_loading.prepare_dicom import prepare_cases
from reconstruction.bspline import reconstruct_cases
from isosurface_extraction.extract import extract_surfaces_for_cases
from visualization.napari_viewer import visualize_case_napari
from visualization.simple_viewer import visualize_case_simple


def main():
    if RUN_DATA_LOADING:
        prepare_cases(case_names=CASE_NAMES)

    if RUN_RECONSTRUCTION:
        reconstruct_cases(case_names=CASE_NAMES)

    if RUN_ISOSURFACE_EXTRACTION:
        extract_surfaces_for_cases(case_names=CASE_NAMES)

    case_for_vis = VIS_CASE
    if case_for_vis is None and CASE_NAMES:
        case_for_vis = CASE_NAMES[0]

    if RUN_NAPARI_VIS:
        visualize_case_napari(case_name=case_for_vis)

    if RUN_SIMPLE_VIS:
        visualize_case_simple(case_name=case_for_vis)


if __name__ == "__main__":
    main()
