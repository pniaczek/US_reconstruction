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
EVALUATION_DIR = DATA_DIR / "evaluation"

SEGMENTATION_MODELS_DIR = ROOT / "models" / "segmentation"

UNSUPERVISED_REPO_DIR = ROOT / "external" / "UnsupervisedSegmentor4Ultrasound"


# ============================================================
# MAIN SWITCHES
# ============================================================

RUN_DATA_LOADING = False
RUN_DENOISING = False
RUN_SEGMENTATION = False

# External unsupervised pipeline.
# False = nie licz jeszcze raz segmentacji unsupervised.
RUN_UNSUPERVISED_SEGMENTATION = False

# Import już policzonych wyników unsupervised do formatu projektu.
RUN_UNSUPERVISED_OUTPUT_LOADING = True

# Rekonstrukcja etykiet/klastrów po imporcie.
RUN_RECONSTRUCTION = True

# Czy rekonstruować też pełną intensywność source_case.
# Dla testu unsupervised zostaw False.
RUN_RECONSTRUCT_INTENSITY = False

RUN_ISOSURFACE_EXTRACTION = False

RUN_NAPARI_VIS = False
RUN_SIMPLE_VIS = False

RUN_EVALUATION = True


# ============================================================
# EVALUATION SWITCHES
# ============================================================

RUN_EVAL_RECONSTRUCTION_WITH_GT = False
RUN_EVAL_SEGMENTATION = False
RUN_EVAL_NO_REFERENCE = True


# ============================================================
# CASE SELECTION
# ============================================================

# Bazowe case'y wejściowe.
# Jeśli RUN_DENOISING=True, main zrobi:
#   <case>_denoised_<method>
#
# Dla importu gotowego unsupervised ten CASE_NAMES może być bazowy,
# ale faktyczny import kontroluje UNSUPERVISED_OUTPUT_IMPORT_CASES.
CASE_NAMES = [
    "kosartur",
]


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
    # "gaussian",
    # "median",
    # "bilateral",
    # "tv",
]

DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION = True
DENOISING_OUTPUT_CASE_SUFFIX = "_denoised"

RDPAD_ITERATIONS = 100
RDPAD_TIMESTEP = 0.20
RDPAD_Q0_MODE = "median"
RDPAD_Q0_PERCENTILE = 25.0

GAUSSIAN_SIGMA = 1.0

MEDIAN_SIZE = 3

BILATERAL_SIGMA_COLOR = 0.05
BILATERAL_SIGMA_SPATIAL = 3.0

NLM_H_FACTOR = 0.8
NLM_PATCH_SIZE = 5
NLM_PATCH_DISTANCE = 6

TV_WEIGHT = 0.08
TV_MAX_NUM_ITER = 200


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
# UNSUPERVISED SEGMENTATION - EXTERNAL PIPELINE
# ============================================================

UNSUPERVISED_REPO_URL = "https://github.com/alexaatm/UnsupervisedSegmentor4Ultrasound.git"
UNSUPERVISED_DINOV2_REPO_URL = "https://github.com/3cology/dinov2_with_attention_extraction.git"

UNSUPERVISED_ALLOW_GIT_CLONE = False

UNSUPERVISED_DATASET_NAME = "DATASET1"
UNSUPERVISED_CASE_NAME = "kosartur_unsupervised"

UNSUPERVISED_IMAGE_KEY = None

UNSUPERVISED_FRAME_START = 0
UNSUPERVISED_FRAME_END = None
UNSUPERVISED_FRAME_STEP = 1

UNSUPERVISED_PNG_NORMALIZATION = "percentile"
UNSUPERVISED_P_LOW = 1.0
UNSUPERVISED_P_HIGH = 99.0

UNSUPERVISED_N_CLASSES = 15

UNSUPERVISED_SEGMENTS_NUM = [30]
UNSUPERVISED_CLUSTERS_NUM = [15]

UNSUPERVISED_WANDB_PROJECT = "kosartur_unsupervised"
UNSUPERVISED_WANDB_ENTITY = "mpniak-agh"
UNSUPERVISED_WANDB_MODE = "disabled"
UNSUPERVISED_USE_WANDB_DISABLED = True
UNSUPERVISED_WANDB_KEY = ""
UNSUPERVISED_WANDB_TAG = "test1"

UNSUPERVISED_PYTHON_EXECUTABLE = "python"

UNSUPERVISED_OVERWRITE_CONFIGS = True
UNSUPERVISED_OVERWRITE_DATASET = True

UNSUPERVISED_NORM = "none"

UNSUPERVISED_INV = False
UNSUPERVISED_GAUSS_BLUR = False
UNSUPERVISED_GAUSS_TETA = 1.0
UNSUPERVISED_HIST_EQ = False

UNSUPERVISED_BATCH_SIZE = 1


# ============================================================
# UNSUPERVISED SEGMENTATION - SPECTRAL / MEMORY
# ============================================================

UNSUPERVISED_IMAGE_DOWNSAMPLE_FACTOR = 7

UNSUPERVISED_C_DINO = 1.0
UNSUPERVISED_C_SSD_KNN = 0.0
UNSUPERVISED_C_VAR_KNN = 0.0
UNSUPERVISED_C_POS_KNN = 0.0
UNSUPERVISED_C_SSD = 0.0
UNSUPERVISED_C_NCC = 0.0
UNSUPERVISED_C_LNCC = 0.0
UNSUPERVISED_C_SSIM = 0.0
UNSUPERVISED_C_MI = 0.0
UNSUPERVISED_C_SAM = 0.0


# ============================================================
# UNSUPERVISED OUTPUT IMPORT
# ============================================================

UNSUPERVISED_IMPORT_USE_CRF = True
UNSUPERVISED_IMPORT_MODEL_NAME = "unsupervised_crf"

UNSUPERVISED_IMPORT_TO_FOR_RECONSTRUCTION = True
UNSUPERVISED_IMPORT_TO_SEGMENTATION_DIR = True
UNSUPERVISED_IMPORT_OVERWRITE = True

# Nowy sposób importu:
# Nie podajesz pełnej ścieżki.
# Loader sam znajdzie najnowszy run pasujący do seg/clust.
#
# base_case + denoising_method:
#   "kosartur" + "rdpad" -> source_case = "kosartur_denoised_rdpad"
#
# output_case zostaw None, wtedy powstanie automatycznie:
#   kosartur_denoised_rdpad_unsup_seg30_clust15_crf
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

# Stary tryb z ręcznym run_dir.
# Zostaw pusty. Loader najpierw sprawdza UNSUPERVISED_OUTPUT_IMPORT_CASES.
UNSUPERVISED_OUTPUT_IMPORTS = []


# ============================================================
# RECONSTRUCTION - GENERAL
# ============================================================

# Dla etykiet / klas najlepiej voxel_nearest.
# bspline interpoluje wartości klas i może tworzyć wartości pośrednie.
RECONSTRUCTION_METHODS = [
    "voxel_nearest",
]

RECONSTRUCTION_SAVE_METHOD_SUBDIRS = True

VOXEL_SIZE = (0.2, 0.2, 0.2)
PIXEL_SPACING = None

BACKGROUND_THRESHOLD = 0.0

CENTER = True
CHUNK_PIXELS = None

# Loader zapisuje etykiety jako:
#   img    -> float32 label volume
#   labels -> uint8 label volume
#
# Rekonstruktor zwykle czyta IMAGE_KEY.
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

VIS_CASE = None

SHOW_RECONSTRUCTION = True
SHOW_GROUND_TRUTH_VOLUME = True
SHOW_SEGMENTATION_LABELS = True
SHOW_SEGMENTATION_CLASSES = True
SHOW_SURFACES = True

NAPARI_NDISPLAY = 3


# ============================================================
# EVALUATION
# ============================================================

EVAL_RECON_METHODS = RECONSTRUCTION_METHODS

EVAL_CASE_NAMES = [
    "kosartur_denoised_rdpad_unsup_seg30_clust15_crf",
]

EVAL_GT_CASE_NAME_MAP = {
    "kosartur": "kosartur",

    "kosartur_denoised_rdpad": "kosartur",
    "kosartur_denoised_gaussian": "kosartur",
    "kosartur_denoised_median": "kosartur",
    "kosartur_denoised_bilateral": "kosartur",
    "kosartur_denoised_tv": "kosartur",

    "kosartur_denoised_rdpad_seg_rdpad": "kosartur",
    "kosartur_denoised_gaussian_seg_gaussian": "kosartur",
    "kosartur_denoised_median_seg_median": "kosartur",
    "kosartur_denoised_bilateral_seg_bilateral": "kosartur",
    "kosartur_denoised_tv_seg_tv": "kosartur",

    "kosartur_denoised_rdpad_unsup_seg30_clust15_crf": "kosartur",
}

# Dla unsupervised nie używamy klasycznych metryk segmentacji bez mapowania klas.
EVAL_SEGMENTATION_MODEL_NAME = "unsupervised_crf"

EVAL_SEGMENTATION_CLASSES = [1, 2, 3, 4, 5]

EVAL_EPS = 1e-8
EVAL_HIST_BINS = 128
EVAL_SURFACE_DICE_TOLERANCE_MM = 1.0

EVAL_OBJECT_THRESHOLD = 0.0
EVAL_BACKGROUND_THRESHOLD = 0.0

EVAL_SSIM_AXIS = 0
EVAL_MS_SSIM_ENABLED = True

EVAL_NO_REFERENCE_USE_MASK = True