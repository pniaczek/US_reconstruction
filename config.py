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

# External unsupervised segmentation repo
UNSUPERVISED_REPO_DIR = ROOT / "external" / "UnsupervisedSegmentor4Ultrasound"


# ============================================================
# MAIN SWITCHES
# ============================================================

RUN_DATA_LOADING = False
RUN_DENOISING = False

# Supervised U-Net segmentation using your trained TorchScript models.
RUN_SEGMENTATION = False

# External unsupervised segmentation pipeline.
RUN_UNSUPERVISED_SEGMENTATION = True

RUN_RECONSTRUCTION = False
RUN_ISOSURFACE_EXTRACTION = False
RUN_NAPARI_VIS = False
RUN_SIMPLE_VIS = False


# ============================================================
# CASE SELECTION
# ============================================================

# None -> wszystkie case'y z odpowiedniego folderu
# ["kosartur"] -> tylko case kosartur
#
# Typowe case'y:
#   kosartur
#   kosartur_denoised_rdpad
#   kosartur_denoised_gaussian
#   kosartur_denoised_tv
#   kosartur_denoised_gaussian_seg_gaussian
#
# Dla unsupervised segmentation ustaw case, który istnieje w:
#   data/for_reconstruction/<CASE_NAME>/
#
# Przykład:
#   data/for_reconstruction/kosartur_denoised_gaussian/
CASE_NAMES = ["kosartur_denoised_gaussian"]


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

# Dostępne metody:
#   "rdpad"
#   "gaussian"
#   "median"
#   "bilateral"
#   "nlm"
#   "tv"
DENOISING_METHODS = [
    # "rdpad",
    "gaussian",
    # "median",
    # "bilateral",
    # "nlm",
    # "tv",
]

DENOISING_WRITE_BACK_TO_FOR_RECONSTRUCTION = True
DENOISING_OUTPUT_CASE_SUFFIX = "_denoised"

# RDPAD
RDPAD_ITERATIONS = 100
RDPAD_TIMESTEP = 0.20
RDPAD_Q0_MODE = "median"      # "median", "mean", "percentile"
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

# Nazwa podfolderu z modelem:
#   models/segmentation/<SEGMENTATION_MODEL_NAME>/
#
# Dostępne przykładowo:
#   "original"
#   "rdpad"
#   "gaussian"
#   "median"
#   "bilateral"
#   "tv"
SEGMENTATION_MODEL_NAME = "gaussian"

SEGMENTATION_MODEL_FILE = "best_model_traced.pt"
SEGMENTATION_MODEL_META_FILE = "best_model_export_meta.json"

# None -> auto-detect dataset image key z H5
SEGMENTATION_IMAGE_KEY = None

# None -> auto: cuda jeśli dostępna, inaczej cpu
# albo ręcznie:
#   "cuda"
#   "cpu"
SEGMENTATION_DEVICE = None

# "auto", "divide_255", "minmax", "none"
SEGMENTATION_NORMALIZATION = "auto"

# None -> eksportuj wszystkie znalezione klasy
# albo np. [1, 2, 3, 4, 5]
SEGMENTATION_CLASSES_TO_EXPORT = [1, 2, 3, 4, 5]

SEGMENTATION_SKIP_BACKGROUND = True

# Wynikowy case:
#   <case>_seg_<SEGMENTATION_MODEL_NAME>
SEGMENTATION_OUTPUT_CASE_SUFFIX = "_seg"

SEGMENTATION_WRITE_TO_FOR_RECONSTRUCTION = True
SEGMENTATION_SAVE_FRAME_MASKS = True

# "masked_intensity" -> zapisuje img * maska
# "binary"           -> zapisuje maskę 0/1
SEGMENTATION_MASKED_IMAGE_VALUE = "masked_intensity"

# Jeśli True, main po supervised segmentacji automatycznie odpali rekonstrukcję
# na case'ach wyprodukowanych przez segmentację.
RECONSTRUCT_SEGMENTATION_OUTPUT = True


# ============================================================
# UNSUPERVISED SEGMENTATION
# ============================================================

# Repozytoria zewnętrzne
UNSUPERVISED_REPO_URL = "https://github.com/alexaatm/UnsupervisedSegmentor4Ultrasound.git"
UNSUPERVISED_DINOV2_REPO_URL = "https://github.com/3cology/dinov2_with_attention_extraction.git"

# Gdzie w zewnętrznym repo ma zostać zapisany dataset:
#
# external/UnsupervisedSegmentor4Ultrasound/data/<UNSUPERVISED_DATASET_NAME>/<UNSUPERVISED_CASE_NAME>/
#
# Przykład:
#   external/UnsupervisedSegmentor4Ultrasound/data/DATASET1/kosartur_unsupervised/
UNSUPERVISED_DATASET_NAME = "DATASET1"

# Nazwa dataset-case w external repo.
# Może być niezależna od CASE_NAMES.
UNSUPERVISED_CASE_NAME = "kosartur_unsupervised"

# None -> auto-detect image key z H5
UNSUPERVISED_IMAGE_KEY = None

# Eksport klatek z H5 do PNG
UNSUPERVISED_FRAME_START = 0
UNSUPERVISED_FRAME_END = None
UNSUPERVISED_FRAME_STEP = 1

# Normalizacja PNG:
#   "percentile" - zwykle najlepsze dla USG
#   "minmax"
#   "raw"
UNSUPERVISED_PNG_NORMALIZATION = "percentile"
UNSUPERVISED_P_LOW = 1.0
UNSUPERVISED_P_HIGH = 99.0

# Dataset config
UNSUPERVISED_N_CLASSES = 6

# Sweep config
UNSUPERVISED_SEGMENTS_NUM = [15]
UNSUPERVISED_CLUSTERS_NUM = [6, 9, 12, 15]

# W&B config generowany automatycznie
UNSUPERVISED_WANDB_PROJECT = "kosartur_unsupervised"
UNSUPERVISED_WANDB_ENTITY = "mpniak-agh"
UNSUPERVISED_WANDB_MODE = "offline"
UNSUPERVISED_WANDB_KEY = "0000000000000000000000000000000000000000"
UNSUPERVISED_WANDB_TAG = "test1"

# Run behavior
UNSUPERVISED_USE_WANDB_DISABLED = True
UNSUPERVISED_PYTHON_EXECUTABLE = "python"

# Przy każdym uruchomieniu nadpisuj dataset PNG i configi.
UNSUPERVISED_OVERWRITE_DATASET = True
UNSUPERVISED_OVERWRITE_CONFIGS = True


# ============================================================
# RECONSTRUCTION - GENERAL
# ============================================================

# Dostępne metody:
#   "bspline"
#   "voxel_nearest"
#   "voxel_nearest_mean"
#   "distance_weighted"
RECONSTRUCTION_METHODS = [
    "bspline",
    # "voxel_nearest",
    # "voxel_nearest_mean",
    # "distance_weighted",
]

RECONSTRUCTION_SAVE_METHOD_SUBDIRS = True

# Rozmiar voxela wynikowego w mm: (x, y, z)
VOXEL_SIZE = (0.15, 0.15, 0.15)

# None -> spacing z H5
# albo ręcznie, np. (0.15, 0.15)
PIXEL_SPACING = None

# Dla danych po cropowaniu/maskowaniu/segmentacji zwykle:
#   BACKGROUND_THRESHOLD = 0.0
#
# Dla pełnych danych bez maskowania można użyć:
#   BACKGROUND_THRESHOLD = None
BACKGROUND_THRESHOLD = 0.0

CENTER = True
CHUNK_PIXELS = None
IMAGE_KEY = None

# "cuda" albo "cpu"
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
DWR_INTERP = "bilinear"  # "nearest" albo "bilinear"
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

# None -> weź pierwszy case z CASE_NAMES albo case wyjściowy z segmentacji.
# Nie ustawiaj VIS_CASE = "".
VIS_CASE = None

SHOW_RECONSTRUCTION = True
SHOW_GROUND_TRUTH_VOLUME = True
SHOW_SEGMENTATION_LABELS = True
SHOW_SEGMENTATION_CLASSES = True
SHOW_SURFACES = True

NAPARI_NDISPLAY = 3