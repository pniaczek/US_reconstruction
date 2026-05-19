from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
FOR_RECON_DIR = DATA_DIR / "for_reconstruction"
DENOISED_DIR = DATA_DIR / "denoised"
SEGMENTATION_DIR = DATA_DIR / "segmentations"
RECONSTRUCTED_DIR = DATA_DIR / "reconstructed"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

SEGMENTATION_MODELS_DIR = ROOT / "models" / "segmentation"




# ============================================================
# MAIN SWITCHES
# ============================================================

RUN_DATA_LOADING = True
RUN_DENOISING = False
RUN_SEGMENTATION = False
RUN_RECONSTRUCTION = False
RUN_ISOSURFACE_EXTRACTION = False

RUN_NAPARI_VIS = False
RUN_SIMPLE_VIS = False

RUN_EVALUATION = False


# ============================================================
# CASE SELECTION
# ============================================================

# To musi być case WYNIKOWY po segmentacji Gaussian:
#   data/reconstructed/kosartur_denoised_gaussian_seg_gaussian/
#
# Nie ustawiaj tutaj "kosartur_denoised_gaussian", bo to jest case wejściowy
# przed segmentacją.

CASE_NAMES = ["rybakszymon_3"]


# ============================================================
# DATA LOADING
# ============================================================

VERBOSE = True
SAVE_IMAGES_NPY = True
SAVE_POSES_NPY = True
CONVERT_IMAGE_TO_FLOAT32 = True

IMAGE_DATASET_NAME = "img"
POSE_DATASET_NAME = "poses"



# ============================================================
# DENOISING
# ============================================================

DENOISING_METHODS = [
    "rdpad",
]

DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION = True
DENOISING_OUTPUT_CASE_SUFFIX = "_denoised"

RDPAD_ITERATIONS = 100
RDPAD_TIMESTEP = 0.20
RDPAD_Q0_MODE = "median"
RDPAD_Q0_PERCENTILE = 25.0

# Gaussian
GAUSSIAN_SIGMA = 1.0

# Median
MEDIAN_SIZE = 3

# Bilateral
BILATERAL_SIGMA_COLOR = 0.05
BILATERAL_SIGMA_SPATIAL = 3.0

# Non-local means
NLM_H_FACTOR = 0.8
NLM_PATCH_SIZE = 5
NLM_PATCH_DISTANCE = 6

# Total variation
TV_WEIGHT = 0.08
TV_MAX_NUM_ITER = 200


# ============================================================
# SUPERVISED SEGMENTATION
# ============================================================

# ============================================================
# SUPERVISED SEGMENTATION
# ============================================================

SEGMENTATION_MODEL_NAME = "rdpad"

SEGMENTATION_MODEL_FILE = "best_model_traced.pt"
SEGMENTATION_MODEL_META_FILE = "best_model_export_meta.json"

SEGMENTATION_IMAGE_KEY = None
SEGMENTATION_DEVICE = None
SEGMENTATION_NORMALIZATION = "auto"

SEGMENTATION_CLASSES_TO_EXPORT = [1, 2, 3, 4, 5]
SEGMENTATION_SKIP_BACKGROUND = True

SEGMENTATION_OUTPUT_CASE_SUFFIX = "_seg"
SEGMENTATION_WRITE_TO_FOR_RECONSTRUCTION = True
SEGMENTATION_SAVE_FRAME_MASKS = True

SEGMENTATION_MASKED_IMAGE_VALUE = "masked_intensity"

RECONSTRUCT_SEGMENTATION_OUTPUT = True


# ============================================================
# RECONSTRUCTION - GENERAL
# ============================================================

RECONSTRUCTION_METHODS = [
    "bspline",
]

RECONSTRUCTION_SAVE_METHOD_SUBDIRS = True

VOXEL_SIZE = (0.15, 0.15, 0.15)
PIXEL_SPACING = None

BACKGROUND_THRESHOLD = 0.0

CENTER = True
CHUNK_PIXELS = None
IMAGE_KEY = None

DEVICE = "cpu"


# ============================================================
# RECONSTRUCTION - B-SPLINE
# ============================================================

N_NEAREST = 4
SPLINE_ORDER = 3
DISTANCE_EPS = 1e-6
DISTANCE_POWER = 1.0


# ============================================================
# RECONSTRUCTION - VOXEL NEAREST
# ============================================================

VNN_N_NEAREST = 1


# ============================================================
# RECONSTRUCTION - VOXEL NEAREST MEAN
# ============================================================

VNN_MEAN_N_NEAREST = 4


# ============================================================
# RECONSTRUCTION - DISTANCE WEIGHTED
# ============================================================

DWR_N_NEAREST = 4
DWR_DISTANCE_POWER = 2.0
DWR_DISTANCE_EPS = 1e-6
DWR_INTERP = "bilinear"
DWR_MAX_PLANE_DIST = None


# ============================================================
# ISOSURFACE EXTRACTION
# ============================================================

ISO_SOURCE_SUBDIR = "segmentations"
ISO_PATTERN = "*.npz"
ISO_LEVEL = None
ISO_THR_RATIO = 0.10
ISO_MIN_VOXELS = 20
ISO_STEP_SIZE = 1
ISO_SAVE_OBJ = True


# ============================================================
# VISUALIZATION
# ============================================================

# Jawnie wskazujemy case wynikowy.
VIS_CASE = "kosartur_denoised_rdpad_seg_rdpad"

SHOW_RECONSTRUCTION = True

# Te elementy mogą nie istnieć dla case'a wynikowego po segmentacji.
# Nie szkodzi, viewer powinien je pominąć.
SHOW_GROUND_TRUTH_VOLUME = True
SHOW_SEGMENTATION_LABELS = True
SHOW_SEGMENTATION_CLASSES = True

# Jeśli nie wygenerowałeś surfaces, może zostać True — viewer pominie brak folderu.
SHOW_SURFACES = True

NAPARI_NDISPLAY = 3


# ============================================================
# EVALUATION
# ============================================================

RUN_EVALUATION = True

RUN_EVAL_RECONSTRUCTION_WITH_GT = True
RUN_EVAL_SEGMENTATION = True
RUN_EVAL_NO_REFERENCE = True

EVALUATION_DIR = DATA_DIR / "evaluation"

EVAL_RECON_METHODS = RECONSTRUCTION_METHODS

EVAL_GT_CASE_NAME_MAP = {
    "kosartur_denoised_rdpad": "kosartur",
    "kosartur_denoised_rdpad_seg_rdpad": "kosartur",

    "kosartur_denoised_gaussian": "kosartur",
    "kosartur_denoised_gaussian_seg_gaussian": "kosartur",

    "kosartur_denoised_median": "kosartur",
    "kosartur_denoised_median_seg_median": "kosartur",

    "kosartur_denoised_bilateral": "kosartur",
    "kosartur_denoised_bilateral_seg_bilateral": "kosartur",

    "kosartur_denoised_tv": "kosartur",
    "kosartur_denoised_tv_seg_tv": "kosartur",
}

EVAL_CASE_NAMES = [
    "kosartur_denoised_rdpad",
    "kosartur_denoised_rdpad_seg_rdpad",
]

EVAL_SEGMENTATION_MODEL_NAME = "rdpad"

EVAL_SEGMENTATION_CLASSES = [1, 2, 3, 4, 5]

EVAL_EPS = 1e-8
EVAL_HIST_BINS = 128
EVAL_SURFACE_DICE_TOLERANCE_MM = 1.0

EVAL_OBJECT_THRESHOLD = 0.0
EVAL_BACKGROUND_THRESHOLD = 0.0

EVAL_SSIM_AXIS = 0
EVAL_MS_SSIM_ENABLED = True

EVAL_NO_REFERENCE_USE_MASK = True