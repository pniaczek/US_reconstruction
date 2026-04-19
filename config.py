from pathlib import Path

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
RUN_RECONSTRUCTION = False
RUN_ISOSURFACE_EXTRACTION = False
RUN_NAPARI_VIS = False
RUN_SIMPLE_VIS = False

# ============================================================
# CASE SELECTION
# ============================================================
# None -> all cases found in the expected folder for a given stage
CASE_NAMES = [""]

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
# RECONSTRUCTION (B-SPLINE)
# ============================================================
VOXEL_SIZE = (0.15, 0.15, 0.15)  # (x, y, z) in mm
PIXEL_SPACING = None
BACKGROUND_THRESHOLD = None
CENTER = True
CHUNK_PIXELS = None
IMAGE_KEY = None
N_NEAREST = 4
SPLINE_ORDER = 3
DISTANCE_EPS = 1e-6
DISTANCE_POWER = 1.0
DEVICE = "cpu"  # "cuda" or "cpu"

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
VIS_CASE = ""
SHOW_RECONSTRUCTION = True
SHOW_GROUND_TRUTH_VOLUME = True
SHOW_SEGMENTATION_LABELS = True
SHOW_SEGMENTATION_CLASSES = True
SHOW_SURFACES = True
NAPARI_NDISPLAY = 3
