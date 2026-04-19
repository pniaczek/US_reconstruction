from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from config import RECONSTRUCTED_DIR, GROUND_TRUTH_DIR


def load_volume(path):
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "volume" in data:
            vol = data["volume"]
        elif "final_volume" in data:
            vol = data["final_volume"]
        elif "appearance_volume" in data:
            vol = data["appearance_volume"]
        elif "mask_volume" in data:
            vol = data["mask_volume"]
        else:
            vol = data[data.files[0]]

        voxel = tuple(data.get("voxel_size", data.get("voxel_size_mm", [1.0, 1.0, 1.0])))
        return vol.astype(np.float32), tuple(float(x) for x in voxel)

    if ext == ".vti":
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        img = reader.GetOutput()
        dims = img.GetDimensions()
        spacing = img.GetSpacing()
        arr = img.GetPointData().GetScalars()
        np_data = vtk_to_numpy(arr)
        vol = np_data.reshape(dims[2], dims[1], dims[0])
        voxel = (spacing[2], spacing[1], spacing[0])
        return vol.astype(np.float32), tuple(float(x) for x in voxel)

    raise ValueError(f"Unsupported extension: {ext}")


def find_case_name(case_name=None):
    if case_name is not None:
        return case_name

    recon_cases = sorted([p.name for p in RECONSTRUCTED_DIR.iterdir() if p.is_dir()])
    gt_cases = sorted([p.name for p in GROUND_TRUTH_DIR.iterdir() if p.is_dir()])
    cases = sorted(set(recon_cases + gt_cases))
    if not cases:
        raise RuntimeError("No cases found for visualization")
    return cases[0]


def find_first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None
