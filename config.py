from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
FOR_RECON_DIR = DATA_DIR / "for_reconstruction"
RECONSTRUCTED_DIR = DATA_DIR / "reconstructed"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"


# ============================================================
# MAIN SWITCHES
# ============================================================

RUN_DATA_LOADING = False
RUN_RECONSTRUCTION = True
RUN_ISOSURFACE_EXTRACTION = False
RUN_NAPARI_VIS = True
RUN_SIMPLE_VIS = False


# ============================================================
# CASE SELECTION
# ============================================================

# None -> wszystkie case'y z odpowiedniego folderu
# ["kosartur"] -> tylko case kosartur
CASE_NAMES = ["kosartur"]


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
# RECONSTRUCTION - GENERAL
# ============================================================

# Dostępne metody:
#   "bspline"
#   "voxel_nearest"
#   "voxel_nearest_mean"
#   "distance_weighted"

RECONSTRUCTION_METHODS = [
    # "bspline",
    "voxel_nearest",
    # "voxel_nearest_mean",
    # "distance_weighted",
]

RECONSTRUCTION_SAVE_METHOD_SUBDIRS = True

# Rozmiar voxela wynikowego w mm: (x, y, z)
VOXEL_SIZE = (0.15, 0.15, 0.15)

# None -> spacing z H5
# albo ręcznie, np. (0.15, 0.15)
PIXEL_SPACING = None
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

# None -> weź pierwszy case z CASE_NAMES.
VIS_CASE = None

SHOW_RECONSTRUCTION = True
SHOW_GROUND_TRUTH_VOLUME = True
SHOW_SEGMENTATION_LABELS = True
SHOW_SEGMENTATION_CLASSES = True
SHOW_SURFACES = True

NAPARI_NDISPLAY = 3