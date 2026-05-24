from pathlib import Path
import glob
import numpy as np
import h5py
import pydicom
from pydicom.tag import Tag

from config import (
    RAW_DIR,
    FOR_RECON_DIR,
    GROUND_TRUTH_DIR,
    VERBOSE,
    SAVE_IMAGES_NPY,
    SAVE_POSES_NPY,
    CONVERT_IMAGE_TO_FLOAT32,
    IMAGE_DATASET_NAME,
    POSE_DATASET_NAME,
)


def log(msg):
    if VERBOSE:
        print(msg)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_single_file(root, patterns):
    root = Path(root)
    for pat in patterns:
        files = sorted(glob.glob(str(root / pat)))
        if files:
            return Path(files[0])
    return None


def describe_dataset(ds, label):
    log("--------------------------------------------------")
    log(f"[INFO] {label}")
    log(f"[INFO] SOPClassUID : {getattr(ds, 'SOPClassUID', 'N/A')}")
    log(f"[INFO] Modality    : {getattr(ds, 'Modality', 'N/A')}")
    log(f"[INFO] Manufacturer: {getattr(ds, 'Manufacturer', 'N/A')}")
    log(f"[INFO] Rows/Cols   : {getattr(ds, 'Rows', 'N/A')} / {getattr(ds, 'Columns', 'N/A')}")
    log(f"[INFO] Frames      : {getattr(ds, 'NumberOfFrames', 'N/A')}")
    log("--------------------------------------------------")


def to_numpy_pixel_array(ds):
    return np.asarray(ds.pixel_array)


def ensure_thw(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        return arr[..., 0]
    raise RuntimeError(f"Unsupported sweep shape: {arr.shape}")


def ensure_zyx(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        return arr[..., 0]
    raise RuntimeError(f"Unsupported volume shape: {arr.shape}")


def extract_spacing_xy(ds):
    sx, sy = 1.0, 1.0
    if hasattr(ds, "PixelSpacing"):
        try:
            ps = [float(x) for x in ds.PixelSpacing]
            if len(ps) >= 2:
                sy, sx = ps[0], ps[1]
                return float(sx), float(sy)
        except Exception:
            pass

    for attr_name in ["ImagerPixelSpacing", "NominalScannedPixelSpacing"]:
        if hasattr(ds, attr_name):
            try:
                ps = [float(x) for x in getattr(ds, attr_name)]
                if len(ps) >= 2:
                    sy, sx = ps[0], ps[1]
                    return float(sx), float(sy)
            except Exception:
                pass

    return float(sx), float(sy)


def extract_spacing_zyx(ds):
    sx, sy = extract_spacing_xy(ds)
    sz = 1.0
    if hasattr(ds, "SliceThickness"):
        try:
            sz = float(ds.SliceThickness)
        except Exception:
            pass
    return float(sz), float(sy), float(sx)


def decode_od_to_float64(raw_value):
    if isinstance(raw_value, bytes) and len(raw_value) % 8 == 0:
        return np.frombuffer(raw_value, dtype=np.float64)
    return None


def extract_poses_from_private_tag_0051_1039(ds, n_frames):
    tag = Tag(0x0051, 0x1039)
    if tag not in ds:
        log("[WARN] Missing private tag (0051,1039).")
        return None

    arr = decode_od_to_float64(ds[tag].value)
    if arr is None or arr.size % 16 != 0:
        log("[WARN] Cannot decode poses from (0051,1039).")
        return None

    mats = arr.reshape(-1, 4, 4).astype(np.float32)
    if mats.shape[0] != n_frames:
        log(f"[WARN] Number of poses != number of frames: {mats.shape[0]} vs {n_frames}")
        return None
    return mats


def save_h5(out_h5, img_thw, Ts, sx, sy):
    with h5py.File(out_h5, "w") as f:
        ds = f.create_dataset(
            IMAGE_DATASET_NAME,
            data=img_thw.astype(np.float32) if CONVERT_IMAGE_TO_FLOAT32 else img_thw,
            compression="gzip",
        )
        ds.attrs["spacing_x"] = float(sx)
        ds.attrs["spacing_y"] = float(sy)
        ds.attrs["spacing"] = np.array([sx, sy], dtype=np.float32)
        f.create_dataset(POSE_DATASET_NAME, data=Ts.astype(np.float32), compression="gzip")


def save_npz_volume(out_npz, vol_zyx, voxel_size_zyx, origin_zyx=None):
    if origin_zyx is None:
        origin_zyx = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    np.savez_compressed(
        out_npz,
        volume=vol_zyx.astype(np.float32),
        voxel_size=np.asarray(voxel_size_zyx, dtype=np.float32),
        origin=np.asarray(origin_zyx, dtype=np.float32),
    )


def save_segmentation_classes(seg_zyx, voxel_size_zyx, out_dir, case_name):
    ensure_dir(out_dir)
    labels = np.unique(seg_zyx)
    labels = labels[labels != 0]
    for lab in labels:
        vol = (seg_zyx == lab).astype(np.float32)
        save_npz_volume(Path(out_dir) / f"{case_name}_class{int(lab)}.npz", vol, voxel_size_zyx)


def prepare_case(case_dir):
    case_dir = Path(case_dir)
    case_name = case_dir.name

    sweeps_dcm = find_single_file(case_dir, ["*Sweeps*.dcm", "*Sweep*.dcm", "*sweeps*.dcm", "*sweep*.dcm"])
    volume_dcm = find_single_file(case_dir, ["*Volume*.dcm", "*volume*.dcm"])
    seg_dcm = find_single_file(case_dir, ["*Segmentation*.dcm", "*segmentation*.dcm", "*Segment*.dcm", "*segment*.dcm"])

    if sweeps_dcm is None:
        print(f"[WARN] {case_name}: no sweep DICOM found")
        return

    recon_case_dir = FOR_RECON_DIR / case_name
    gt_case_dir = GROUND_TRUTH_DIR / case_name
    seg_classes_dir = gt_case_dir / "seg_classes"

    ensure_dir(recon_case_dir)
    ensure_dir(gt_case_dir)
    ensure_dir(seg_classes_dir)

    out_h5 = recon_case_dir / f"{case_name}_from_dicom.h5"
    out_img_npy = recon_case_dir / f"{case_name}_images.npy"
    out_pose_npy = recon_case_dir / f"{case_name}_poses.npy"

    out_volume_npz = gt_case_dir / f"{case_name}_volume.npz"
    out_seg_labels_npz = gt_case_dir / f"{case_name}_segmentation_labels.npz"

    ds_sweep = pydicom.dcmread(str(sweeps_dcm))
    describe_dataset(ds_sweep, f"SWEEPS: {case_name}")

    img_thw = ensure_thw(to_numpy_pixel_array(ds_sweep))
    sx, sy = extract_spacing_xy(ds_sweep)
    Ts = extract_poses_from_private_tag_0051_1039(ds_sweep, img_thw.shape[0])
    if Ts is None:
        raise RuntimeError(f"{case_name}: failed to extract poses from sweep DICOM")

    save_h5(out_h5, img_thw, Ts, sx, sy)

    if SAVE_IMAGES_NPY:
        np.save(out_img_npy, img_thw)
    if SAVE_POSES_NPY:
        np.save(out_pose_npy, Ts)

    if volume_dcm is not None:
        ds_vol = pydicom.dcmread(str(volume_dcm))
        vol_zyx = ensure_zyx(to_numpy_pixel_array(ds_vol))
        voxel_size_zyx = extract_spacing_zyx(ds_vol)
        save_npz_volume(out_volume_npz, vol_zyx, voxel_size_zyx)

    if seg_dcm is not None:
        ds_seg = pydicom.dcmread(str(seg_dcm))
        seg_zyx = ensure_zyx(to_numpy_pixel_array(ds_seg))
        if np.issubdtype(seg_zyx.dtype, np.floating):
            seg_zyx = np.rint(seg_zyx)
        seg_zyx = seg_zyx.astype(np.uint16)
        voxel_size_zyx = extract_spacing_zyx(ds_seg)
        save_npz_volume(out_seg_labels_npz, seg_zyx, voxel_size_zyx)
        save_segmentation_classes(seg_zyx, voxel_size_zyx, seg_classes_dir, case_name)

    print(f"[OK] Prepared case: {case_name}")


def prepare_cases(case_names=None):
    case_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])
    if case_names is not None:
        case_dirs = [p for p in case_dirs if p.name in case_names]

    if not case_dirs:
        print(f"[WARN] No cases found in: {RAW_DIR}")
        return

    for case_dir in case_dirs:
        prepare_case(case_dir)
