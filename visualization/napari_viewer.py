from pathlib import Path
import numpy as np
import napari

from config import (
    RECONSTRUCTED_DIR,
    GROUND_TRUTH_DIR,
    SHOW_RECONSTRUCTION,
    SHOW_GROUND_TRUTH_VOLUME,
    SHOW_SEGMENTATION_LABELS,
    SHOW_SEGMENTATION_CLASSES,
    SHOW_SURFACES,
    NAPARI_NDISPLAY,
)
from visualization.common import load_volume, find_case_name, find_first_existing


def add_surface_npz(viewer, path):
    data = np.load(path, allow_pickle=True)
    verts = data["verts"]
    faces = data["faces"]
    values = data["values"] if "values" in data else np.ones((verts.shape[0],), dtype=np.float32)
    viewer.add_surface((verts, faces, values), name=Path(path).stem, opacity=0.6, shading="smooth")


def visualize_case_napari(case_name=None):
    case_name = find_case_name(case_name)
    recon_dir = RECONSTRUCTED_DIR / case_name
    gt_dir = GROUND_TRUTH_DIR / case_name

    viewer = napari.Viewer(ndisplay=NAPARI_NDISPLAY)

    if SHOW_RECONSTRUCTION:
        recon_path = find_first_existing([recon_dir / f"{case_name}_from_dicom.npz", *sorted(recon_dir.glob("*.npz")), *sorted(recon_dir.glob("*.vti"))])
        if recon_path is not None:
            vol, voxel = load_volume(recon_path)
            viewer.add_image(vol, name="reconstruction", scale=voxel, colormap="gray", rendering="mip")

    if SHOW_GROUND_TRUTH_VOLUME:
        gt_vol_path = gt_dir / f"{case_name}_volume.npz"
        if gt_vol_path.exists():
            vol, voxel = load_volume(gt_vol_path)
            viewer.add_image(vol, name="ground_truth_volume", scale=voxel, colormap="green", rendering="mip", opacity=0.35)

    if SHOW_SEGMENTATION_LABELS:
        seg_path = gt_dir / f"{case_name}_segmentation_labels.npz"
        if seg_path.exists():
            vol, voxel = load_volume(seg_path)
            viewer.add_labels(vol.astype(np.int32), name="segmentation_labels", scale=voxel, opacity=0.45)

    if SHOW_SEGMENTATION_CLASSES:
        seg_classes_dir = gt_dir / "seg_classes"
        if seg_classes_dir.exists():
            for path in sorted(seg_classes_dir.glob("*.npz")):
                vol, voxel = load_volume(path)
                viewer.add_image(vol, name=path.stem, scale=voxel, rendering="iso", opacity=0.45)

    if SHOW_SURFACES:
        for surface_dir in [gt_dir / "surfaces", recon_dir / "surfaces"]:
            if surface_dir.exists():
                for path in sorted(surface_dir.glob("*_surface.npz")):
                    add_surface_npz(viewer, path)

    napari.run()
