# VNN USG Repository

Minimal, modular repository for:
- preparing DICOM cases,
- reconstructing 3D ultrasound volumes,
- extracting isosurfaces,
- visualizing reconstructed data and ground truth.

## Folder structure

```text
vnn_usg_repo/
├── config.py
├── main.py
├── data/
│   ├── raw/
│   │   └── CASE_NAME/
│   │       ├── *Sweeps*.dcm
│   │       ├── *Volume*.dcm
│   │       └── *Segmentation*.dcm
│   ├── for_reconstruction/
│   │   └── CASE_NAME/
│   │       ├── CASE_NAME_from_dicom.h5
│   │       ├── CASE_NAME_images.npy
│   │       └── CASE_NAME_poses.npy
│   ├── reconstructed/
│   │   └── CASE_NAME/
│   │       ├── CASE_NAME_from_dicom.npz
│   │       ├── CASE_NAME_from_dicom.vti
│   │       └── surfaces/
│   └── ground_truth/
│       └── CASE_NAME/
│           ├── CASE_NAME_volume.npz
│           ├── CASE_NAME_segmentation_labels.npz
│           ├── seg_classes/
│           │   ├── CASE_NAME_class1.npz
│           │   └── CASE_NAME_class2.npz
│           └── surfaces/
├── data_loading/
│   ├── h5_utils.py
│   └── prepare_dicom.py
├── reconstruction/
│   └── bspline.py
├── isosurface_extraction/
│   └── extract.py
├── visualization/
│   ├── common.py
│   ├── napari_viewer.py
│   └── simple_viewer.py
└── utils/
```

## How to use

All control is done from `config.py`.

### 1. Put raw cases here

```text
data/raw/001/
data/raw/002/
...
```

Each case folder should contain your DICOM files.

### 2. Edit `config.py`

Example:

```python
RUN_DATA_LOADING = True
RUN_RECONSTRUCTION = True
RUN_ISOSURFACE_EXTRACTION = True
RUN_NAPARI_VIS = True
RUN_SIMPLE_VIS = False

CASE_NAMES = ["001"]
VIS_CASE = "001"
DEVICE = "cuda"
```

### 3. Run

```bash
python main.py
```

## Notes

- `data_loading/prepare_dicom.py` converts DICOM data into H5 for reconstruction and NPZ for ground truth/visualization.
- `reconstruction/bspline.py` contains your GPU-ready B-spline reconstruction.
- `visualization/napari_viewer.py` shows reconstruction, ground truth, labels, per-class segmentations, and extracted surfaces only if they exist.
- `isosurface_extraction/extract.py` runs marching cubes on NPZ volumes.
- All paths are relative to the repository root.
