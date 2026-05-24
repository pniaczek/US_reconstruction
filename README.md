# VNN USG Repository

Repository for processing 2D ultrasound sweeps and reconstructing 3D ultrasound volumes.

The project supports:

- DICOM case preparation,
- denoising,
- supervised U-Net segmentation,
- unsupervised segmentation with `UnsupervisedSegmentor4Ultrasound`,
- importing unsupervised segmentation outputs into the project format,
- 3D reconstruction,
- isosurface extraction,
- visualization,
- evaluation,
- U-Net training preparation.

All main pipeline behavior is controlled through `config.py` and executed with `main.py`.

---

## Repository structure

```text
US_reconstruction/
├── config.py
├── main.py
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── CASE_NAME/
│   │       ├── *.dcm
│   │       └── ...
│   ├── for_reconstruction/
│   │   └── CASE_NAME/
│   │       ├── CASE_NAME_from_dicom.h5
│   │       ├── CASE_NAME_images.npy
│   │       └── CASE_NAME_poses.npy
│   ├── denoised/
│   │   └── CASE_NAME/
│   │       └── METHOD/
│   │           └── CASE_NAME_METHOD.h5
│   ├── segmentations/
│   │   └── CASE_NAME/
│   │       └── MODEL_NAME/
│   │           ├── CASE_NAME_labels.npz
│   │           └── CASE_NAME_metadata.json
│   ├── reconstructed/
│   │   └── CASE_NAME/
│   │       └── METHOD/
│   │           ├── CASE_NAME.npz
│   │           ├── CASE_NAME.vti
│   │           └── surfaces/
│   ├── ground_truth/
│   │   └── CASE_NAME/
│   │       ├── CASE_NAME_volume.npz
│   │       ├── CASE_NAME_segmentation_labels.npz
│   │       └── seg_classes/
│   ├── evaluation/
│   │   └── CASE_NAME/# VNN USG Repository

Repository for processing 2D ultrasound sweeps and reconstructing 3D ultrasound volumes.

The project supports:

- DICOM case preparation,
- denoising,
- supervised U-Net segmentation,
- unsupervised segmentation with `UnsupervisedSegmentor4Ultrasound`,
- importing unsupervised segmentation outputs into the project format,
- 3D reconstruction,
- isosurface extraction,
- visualization,
- evaluation,
- U-Net training preparation.

All main pipeline behavior is controlled through `config.py` and executed with `main.py`.

---

## Repository structure

```text
US_reconstruction/
├── config.py
├── main.py
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── CASE_NAME/
│   │       ├── *.dcm
│   │       └── ...
│   ├── for_reconstruction/
│   │   └── CASE_NAME/
│   │       ├── CASE_NAME_from_dicom.h5
│   │       ├── CASE_NAME_images.npy
│   │       └── CASE_NAME_poses.npy
│   ├── denoised/
│   │   └── CASE_NAME/
│   │       └── METHOD/
│   │           └── CASE_NAME_METHOD.h5
│   ├── segmentations/
│   │   └── CASE_NAME/
│   │       └── MODEL_NAME/
│   │           ├── CASE_NAME_labels.npz
│   │           └── CASE_NAME_metadata.json
│   ├── reconstructed/
│   │   └── CASE_NAME/
│   │       └── METHOD/
│   │           ├── CASE_NAME.npz
│   │           ├── CASE_NAME.vti
│   │           └── surfaces/
│   ├── ground_truth/
│   │   └── CASE_NAME/
│   │       ├── CASE_NAME_volume.npz
│   │       ├── CASE_NAME_segmentation_labels.npz
│   │       └── seg_classes/
│   ├── evaluation/
│   │   └── CASE_NAME/
│   │       ├── no_reference/
│   │       ├── reconstruction_with_gt/
│   │       └── segmentation/
│   └── segmentation_training/
│       └── DATASET_NAME/
│           ├── images/
│           └── masks/
├── data_loading/
│   ├── h5_utils.py
│   ├── prepare_dicom.py
│   └── unsupervised_output_loader.py
├── denoising/
│   ├── runner.py
│   ├── common.py
│   ├── classical.py
│   └── rdp_ad.py
├── segmentation/
│   ├── runner.py
│   ├── train_unet.py
│   └── ...
├── unsupervised_segmentation/
│   └── runner.py
├── reconstruction/
│   ├── runner.py
│   ├── common.py
│   ├── bspline.py
│   ├── vnn.py
│   ├── vnn_mean.py
│   └── vnn_distance_weighted.py
├── isosurface_extraction/
│   └── extract.py
├── visualization/
│   ├── common.py
│   ├── napari_viewer.py
│   └── simple_viewer.py
├── evaluation/
│   ├── no_reference.py
│   ├── reconstruction_with_gt.py
│   └── segmentation_metrics.py
├── models/
│   └── segmentation/
└── external/
    └── UnsupervisedSegmentor4Ultrasound/
```

---

## Installation

Create a virtual environment:

```bash
cd /path/to/US_reconstruction
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you use a machine without a system C++ compiler, some optional packages such as `SimpleCRF` may fail to build. See the CRF section below.

---

## Data input

### Raw DICOM data

Put each case into a separate folder:

```text
data/raw/CASE_NAME/
├── *.dcm
└── ...
```

Example:

```text
data/raw/kosartur/
├── sweep_001.dcm
├── volume_001.dcm
└── segmentation_001.dcm
```

Then enable DICOM preparation in `config.py`:

```python
RUN_DATA_LOADING = True
CASE_NAMES = ["kosartur"]
```

Run:

```bash
python main.py
```

The prepared case will be written to:

```text
data/for_reconstruction/kosartur/
```

The main H5 file should contain at least:

```text
img      -> ultrasound frames, shape (N, H, W)
poses    -> probe poses, shape (N, 4, 4)
```

---

## Main configuration workflow

Most workflows are controlled from `config.py`.

Typical switches:

```python
RUN_DATA_LOADING = False
RUN_DENOISING = False
RUN_SEGMENTATION = False
RUN_UNSUPERVISED_SEGMENTATION = False
RUN_UNSUPERVISED_OUTPUT_LOADING = False
RUN_RECONSTRUCTION = True
RUN_ISOSURFACE_EXTRACTION = False
RUN_NAPARI_VIS = False
RUN_SIMPLE_VIS = False
RUN_EVALUATION = True
```

Select cases:

```python
CASE_NAMES = ["kosartur"]
```

Run the pipeline:

```bash
python main.py
```

---

## Denoising

Supported denoising methods:

```python
DENOISING_METHODS = [
    "rdpad",
    "gaussian",
    "median",
    "bilateral",
    "tv",
]
```

Enable denoising:

```python
RUN_DENOISING = True
CASE_NAMES = ["kosartur"]
DENOISING_METHODS = ["rdpad"]
DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION = True
```

Output:

```text
data/denoised/kosartur/rdpad/
data/for_reconstruction/kosartur_denoised_rdpad/
```

The output case name is built as:

```text
<base_case>_denoised_<method>
```

Example:

```text
kosartur_denoised_rdpad
```

---

## Supervised segmentation

Supervised segmentation uses a trained TorchScript U-Net model stored in:

```text
models/segmentation/MODEL_NAME/
├── best_model_traced.pt
└── best_model_export_meta.json
```

Configure:

```python
RUN_SEGMENTATION = True

SEGMENTATION_MODEL_NAME = "rdpad"
SEGMENTATION_MODEL_FILE = "best_model_traced.pt"
SEGMENTATION_MODEL_META_FILE = "best_model_export_meta.json"

SEGMENTATION_CLASSES_TO_EXPORT = [1, 2, 3, 4, 5]
SEGMENTATION_WRITE_TO_FOR_RECONSTRUCTION = True
```

Output:

```text
data/segmentations/CASE_NAME_seg_MODEL_NAME/MODEL_NAME/
data/for_reconstruction/CASE_NAME_seg_MODEL_NAME/
```

Example:

```text
data/for_reconstruction/kosartur_denoised_rdpad_seg_rdpad/
```

---

## Training a U-Net segmentation model

Training data should be placed under:

```text
data/segmentation_training/DATASET_NAME/
├── images/
│   ├── image_000001.png
│   ├── image_000002.png
│   └── ...
└── masks/
    ├── image_000001.png
    ├── image_000002.png
    └── ...
```

The image and mask filenames should match.

Example:

```text
data/segmentation_training/rdpad_dataset/
├── images/
│   ├── frame_000000.png
│   └── frame_000001.png
└── masks/
    ├── frame_000000.png
    └── frame_000001.png
```

Run training directly from the `segmentation` module, not through `main.py`:

```bash
python segmentation/train_unet.py
```

The trained model should be saved under:

```text
models/segmentation/MODEL_NAME/
```

Expected output:

```text
models/segmentation/MODEL_NAME/
├── best_model.pt
├── best_model_traced.pt
├── best_model_export_meta.json
└── training_log.csv
```

After training, set in `config.py`:

```python
SEGMENTATION_MODEL_NAME = "MODEL_NAME"
SEGMENTATION_MODEL_FILE = "best_model_traced.pt"
SEGMENTATION_MODEL_META_FILE = "best_model_export_meta.json"
```

---

## Unsupervised segmentation

The project integrates the external repository:

```text
external/UnsupervisedSegmentor4Ultrasound/
```

The external repository must contain:

```text
external/UnsupervisedSegmentor4Ultrasound/
├── deep-spectral-segmentation/
└── dinov2_with_attention_extraction/
```

If the repository is already copied manually, keep:

```python
UNSUPERVISED_ALLOW_GIT_CLONE = False
```

### Running unsupervised segmentation

Configure:

```python
RUN_UNSUPERVISED_SEGMENTATION = True

CASE_NAMES = ["kosartur_denoised_rdpad"]

UNSUPERVISED_CASE_NAME = "kosartur_unsupervised"
UNSUPERVISED_DATASET_NAME = "DATASET1"

UNSUPERVISED_SEGMENTS_NUM = [30]
UNSUPERVISED_CLUSTERS_NUM = [15]
UNSUPERVISED_N_CLASSES = 15

UNSUPERVISED_IMAGE_DOWNSAMPLE_FACTOR = 7
```

Run:

```bash
python main.py
```

The external pipeline creates output under:

```text
external/UnsupervisedSegmentor4Ultrasound/
└── deep-spectral-segmentation/
    └── DATASET1/
        └── kosartur_unsupervised/
            └── main/
                └── test1/
                    └── seg30_clust15_time.../
```

Important output folders:

```text
semantic_segmentations/laplacian/segmaps/
semantic_segmentations/laplacian/crf_segmaps/
plots/segmaps/
plots/crf_segmaps/
```

---

## CRF post-processing

CRF is optional but useful for sharpening segmentation boundaries.

Enable it in the generated external pipeline config:

```python
pipeline_steps.crf_segm = True
```

In this project, use:

```python
UNSUPERVISED_IMPORT_USE_CRF = True
```

CRF output is expected at:

```text
semantic_segmentations/laplacian/crf_segmaps/
```

If CRF output does not exist, the loader can fall back to raw `segmaps`.

### Installing SimpleCRF

The external repository expects:

```python
import denseCRF
```

This is provided by `SimpleCRF`.

Try:

```bash
pip install SimpleCRF
```

If installation fails with:

```text
error: command 'x86_64-linux-gnu-g++' failed: No such file or directory
```

then the system has no C++ compiler available.

If you do not have `sudo`, use Miniforge/Conda to provide compilers:

```bash
conda create -n usg-compiler -c conda-forge -y gcc_linux-64 gxx_linux-64 make cmake python=3.12
conda activate usg-compiler
```

Then activate your project venv and set compilers:

```bash
cd /path/to/US_reconstruction
source venv/bin/activate

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

If `Python.h` is missing during compilation, add:

```bash
export CPPFLAGS="-I$CONDA_PREFIX/include/python3.12"
export CFLAGS="-I$CONDA_PREFIX/include/python3.12"
export CXXFLAGS="-I$CONDA_PREFIX/include/python3.12"
```

Then install:

```bash
python -m pip install --no-cache-dir --no-build-isolation SimpleCRF
```

Check:

```bash
python - <<'PY'
import denseCRF
print("denseCRF OK")
PY
```

If `SimpleCRF` still fails because of NumPy API incompatibility, use a compatible NumPy version or disable CRF and use raw `segmaps`.

---

## Importing unsupervised outputs

After the external unsupervised pipeline has finished, import its PNG segmentations into the project format.

Enable:

```python
RUN_UNSUPERVISED_SEGMENTATION = False
RUN_UNSUPERVISED_OUTPUT_LOADING = True
```

Use the automatic import mode:

```python
UNSUPERVISED_OUTPUT_IMPORT_CASES = [
    {
        "base_case": "kosartur",
        "denoising_method": "rdpad",
        "segments_num": 30,
        "clusters_num": 15,
        "use_crf": True,
        "model_name": "unsupervised_crf",
        "output_case": None,
    },
]
```

This means:

```text
base_case = kosartur
denoising_method = rdpad
source_case = kosartur_denoised_rdpad
output_case = kosartur_denoised_rdpad_unsup_seg30_clust15_crf
```

The loader automatically finds the matching external unsupervised run.

Imported outputs are saved to:

```text
data/for_reconstruction/kosartur_denoised_rdpad_unsup_seg30_clust15_crf/
data/segmentations/kosartur_denoised_rdpad_unsup_seg30_clust15_crf/unsupervised_crf/
```

The imported H5 contains:

```text
img      -> labels as float32, used by reconstruction
labels   -> labels as uint8
poses    -> copied source poses
```

---

## Reconstruction

Supported reconstruction methods:

```python
RECONSTRUCTION_METHODS = [
    "bspline",
    "voxel_nearest",
    "voxel_nearest_mean",
    "distance_weighted",
]
```

For label maps or segmentation classes, prefer:

```python
RECONSTRUCTION_METHODS = ["voxel_nearest"]
```

because interpolation-based methods such as B-spline can create invalid intermediate class values.

For intensity reconstruction, `bspline` or `distance_weighted` can be used.

Example for unsupervised labels:

```python
RUN_RECONSTRUCTION = True
RUN_RECONSTRUCT_INTENSITY = False

RECONSTRUCTION_METHODS = ["voxel_nearest"]
IMAGE_KEY = None
VOXEL_SIZE = (0.2, 0.2, 0.2)
```

Run:

```bash
python main.py
```

Output:

```text
data/reconstructed/CASE_NAME/voxel_nearest/
├── CASE_NAME.npz
└── CASE_NAME.vti
```

---

## Isosurface extraction

Enable:

```python
RUN_ISOSURFACE_EXTRACTION = True
```

Configure:

```python
ISO_SOURCE_SUBDIR = "segmentations"
ISO_PATTERN = "*.npz"
ISO_LEVEL = None
ISO_THR_RATIO = 0.10
ISO_MIN_VOXELS = 20
ISO_STEP_SIZE = 1
ISO_SAVE_OBJ = True
```

Run:

```bash
python main.py
```

Surfaces are saved under:

```text
data/reconstructed/CASE_NAME/METHOD/surfaces/
```

---

## Visualization

Napari visualization:

```python
RUN_NAPARI_VIS = True
VIS_CASE = "kosartur_denoised_rdpad_unsup_seg30_clust15_crf"
```

Run:

```bash
python main.py
```

The viewer displays available data only:

- reconstructed volumes,
- ground truth volumes,
- segmentation labels,
- per-class segmentations,
- extracted surfaces.

Napari may not work correctly on a remote Jupyter server without GUI forwarding. For interactive visualization, run it locally or with a configured graphical environment.

Simple visualization:

```python
RUN_SIMPLE_VIS = True
```

---

## Evaluation

Evaluation is controlled by:

```python
RUN_EVALUATION = True

RUN_EVAL_RECONSTRUCTION_WITH_GT = False
RUN_EVAL_SEGMENTATION = False
RUN_EVAL_NO_REFERENCE = True
```

### No-reference evaluation

No-reference metrics do not require ground truth.

They include metrics such as:

- CNR,
- gCNR,
- boundary sharpness,
- noise standard deviation,
- total variation,
- entropy,
- gradient energy.

Configure:

```python
RUN_EVAL_NO_REFERENCE = True

EVAL_RECON_METHODS = RECONSTRUCTION_METHODS
EVAL_OBJECT_THRESHOLD = 0.0
EVAL_BACKGROUND_THRESHOLD = 0.0
EVAL_HIST_BINS = 128
EVAL_NO_REFERENCE_USE_MASK = True
```

Output:

```text
data/evaluation/CASE_NAME/no_reference/
├── *_no_reference_metrics.json
├── *_hist_object_background.png
├── *_middle_slice_quality.png
└── summary_no_reference.csv
```

### Reconstruction evaluation with ground truth

Enable only if ground truth is available:

```python
RUN_EVAL_RECONSTRUCTION_WITH_GT = True
```

Ground truth should be placed under:

```text
data/ground_truth/CASE_NAME/
├── CASE_NAME_volume.npz
└── CASE_NAME_segmentation_labels.npz
```

If needed, map output cases to GT cases:

```python
EVAL_GT_CASE_NAME_MAP = {
    "kosartur_denoised_rdpad": "kosartur",
}
```

For no-reference-only workflows, this map can stay empty:

```python
EVAL_GT_CASE_NAME_MAP = {}
```

### Segmentation metrics

Use segmentation metrics only when class labels have stable semantics, for example supervised U-Net output.

Do not use standard segmentation metrics directly for unsupervised clusters unless you first map cluster IDs to anatomical labels.

---

## Typical workflows

### 1. Prepare DICOM case

```python
RUN_DATA_LOADING = True
RUN_DENOISING = False
RUN_RECONSTRUCTION = False
RUN_EVALUATION = False

CASE_NAMES = ["kosartur"]
```

```bash
python main.py
```

---

### 2. Denoise a case

```python
RUN_DATA_LOADING = False
RUN_DENOISING = True
RUN_RECONSTRUCTION = False

CASE_NAMES = ["kosartur"]
DENOISING_METHODS = ["rdpad"]
```

```bash
python main.py
```

Output:

```text
data/for_reconstruction/kosartur_denoised_rdpad/
```

---

### 3. Run unsupervised segmentation

```python
RUN_DENOISING = False
RUN_UNSUPERVISED_SEGMENTATION = True
RUN_UNSUPERVISED_OUTPUT_LOADING = False
RUN_RECONSTRUCTION = False

CASE_NAMES = ["kosartur_denoised_rdpad"]

UNSUPERVISED_SEGMENTS_NUM = [30]
UNSUPERVISED_CLUSTERS_NUM = [15]
UNSUPERVISED_N_CLASSES = 15
```

```bash
python main.py
```

---

### 4. Import already computed unsupervised output

```python
RUN_UNSUPERVISED_SEGMENTATION = False
RUN_UNSUPERVISED_OUTPUT_LOADING = True
RUN_RECONSTRUCTION = False

UNSUPERVISED_OUTPUT_IMPORT_CASES = [
    {
        "base_case": "kosartur",
        "denoising_method": "rdpad",
        "segments_num": 30,
        "clusters_num": 15,
        "use_crf": True,
        "model_name": "unsupervised_crf",
        "output_case": None,
    },
]
```

```bash
python main.py
```

---

### 5. Reconstruct imported unsupervised labels

```python
RUN_UNSUPERVISED_OUTPUT_LOADING = True
RUN_RECONSTRUCTION = True
RUN_RECONSTRUCT_INTENSITY = False
RUN_EVALUATION = False

RECONSTRUCTION_METHODS = ["voxel_nearest"]
```

```bash
python main.py
```

---

### 6. Run no-reference evaluation

```python
RUN_RECONSTRUCTION = False
RUN_EVALUATION = True

RUN_EVAL_RECONSTRUCTION_WITH_GT = False
RUN_EVAL_SEGMENTATION = False
RUN_EVAL_NO_REFERENCE = True
```

```bash
python main.py
```

---

## Notes

- Use `voxel_nearest` for segmentation labels and class volumes.
- Use `bspline`, `voxel_nearest_mean`, or `distance_weighted` mainly for intensity volumes.
- CRF is optional. If `SimpleCRF` is not installed, disable CRF or import raw `segmaps`.
- Do not commit API keys, W&B keys, model checkpoints, large datasets, or generated reconstruction outputs unless intentionally versioned with Git LFS.
- All paths are relative to the repository root.

│   │       ├── no_reference/
│   │       ├── reconstruction_with_gt/
│   │       └── segmentation/
│   └── segmentation_training/
│       └── DATASET_NAME/
│           ├── images/
│           └── masks/
├── data_loading/
│   ├── h5_utils.py
│   ├── prepare_dicom.py
│   └── unsupervised_output_loader.py
├── denoising/
│   ├── runner.py
│   ├── common.py
│   ├── classical.py
│   └── rdp_ad.py
├── segmentation/
│   ├── runner.py
│   ├── train_unet.py
│   └── ...
├── unsupervised_segmentation/
│   └── runner.py
├── reconstruction/
│   ├── runner.py
│   ├── common.py
│   ├── bspline.py
│   ├── vnn.py
│   ├── vnn_mean.py
│   └── vnn_distance_weighted.py
├── isosurface_extraction/
│   └── extract.py
├── visualization/
│   ├── common.py
│   ├── napari_viewer.py
│   └── simple_viewer.py
├── evaluation/
│   ├── no_reference.py
│   ├── reconstruction_with_gt.py
│   └── segmentation_metrics.py
├── models/
│   └── segmentation/
└── external/
    └── UnsupervisedSegmentor4Ultrasound/