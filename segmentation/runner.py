from pathlib import Path
import json

import numpy as np
import torch

from config import (
    FOR_RECON_DIR,
    SEGMENTATION_DIR,
    SEGMENTATION_MODELS_DIR,
    SEGMENTATION_MODEL_NAME,
    SEGMENTATION_MODEL_FILE,
    SEGMENTATION_MODEL_META_FILE,
    SEGMENTATION_IMAGE_KEY,
    SEGMENTATION_DEVICE,
    SEGMENTATION_NORMALIZATION,
    SEGMENTATION_CLASSES_TO_EXPORT,
    SEGMENTATION_SKIP_BACKGROUND,
    SEGMENTATION_OUTPUT_CASE_SUFFIX,
    SEGMENTATION_WRITE_TO_FOR_RECONSTRUCTION,
    SEGMENTATION_SAVE_FRAME_MASKS,
    SEGMENTATION_MASKED_IMAGE_VALUE,
)

from data_loading.h5_utils import list_h5_files

from segmentation.common import (
    load_h5_image_and_poses,
    save_h5_image_and_poses,
    load_model_meta,
    get_model_input_size,
    normalize_frame_for_unet,
    resize_frame_float,
    resize_mask_nearest,
)


def normalize_case_names(case_names):
    if case_names is None:
        return None

    if isinstance(case_names, str):
        return [case_names]

    return list(case_names)


def get_model_paths():
    model_dir = SEGMENTATION_MODELS_DIR / SEGMENTATION_MODEL_NAME
    model_path = model_dir / SEGMENTATION_MODEL_FILE
    meta_path = model_dir / SEGMENTATION_MODEL_META_FILE

    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu segmentacji: {model_path}")

    return model_path, meta_path


def load_segmentation_model():
    model_path, meta_path = get_model_paths()

    device = torch.device(
        SEGMENTATION_DEVICE if SEGMENTATION_DEVICE is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    print(f"[INFO] Loading segmentation model: {model_path}")
    print(f"[INFO] Segmentation device: {device}")

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    meta = load_model_meta(meta_path)

    return model, meta, device, model_path, meta_path


@torch.no_grad()
def predict_masks_for_sequence(
    img_thw,
    model,
    model_input_size,
    device,
):
    """
    img_thw: (T,H,W)
    return: masks_thw uint8 (T,H,W), logits optional not saved
    """
    t, h, w = img_thw.shape

    pred_masks = np.zeros((t, h, w), dtype=np.uint8)

    for i in range(t):
        frame = img_thw[i]

        frame01 = normalize_frame_for_unet(
            frame,
            mode=SEGMENTATION_NORMALIZATION,
        )

        frame_resized = resize_frame_float(
            frame01,
            model_input_size,
        )

        x = torch.from_numpy(frame_resized[None, None, :, :]).to(
            device=device,
            dtype=torch.float32,
        )

        logits = model(x)
        pred_small = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        pred_full = resize_mask_nearest(
            pred_small,
            size_hw=(h, w),
        )

        pred_masks[i] = pred_full

        if (i + 1) % 50 == 0 or i == t - 1:
            print(f"[INFO] segmented frames: {i + 1}/{t}")

    return pred_masks


def get_classes_to_export(pred_masks):
    unique_classes = sorted([int(x) for x in np.unique(pred_masks)])

    if SEGMENTATION_CLASSES_TO_EXPORT is None:
        classes = unique_classes
    else:
        classes = [int(c) for c in SEGMENTATION_CLASSES_TO_EXPORT]

    if SEGMENTATION_SKIP_BACKGROUND:
        classes = [c for c in classes if c != 0]

    return classes


def save_segmentation_outputs(
    case_name,
    source_h5,
    base,
    img,
    poses,
    spacing_xy,
    pred_masks,
    classes_to_export,
    model_path,
    meta_path,
):
    """
    Zapisuje:
      1) data/segmentations/<case>/<model_name>/<base>_labels.npz
      2) data/segmentations/<case>/<model_name>/classes/<base>_classX_masks.npz
      3) data/for_reconstruction/<case>_seg_<model_name>/<base>_classX.h5
    """
    seg_case_dir = SEGMENTATION_DIR / case_name / SEGMENTATION_MODEL_NAME
    seg_classes_dir = seg_case_dir / "classes"
    seg_case_dir.mkdir(parents=True, exist_ok=True)
    seg_classes_dir.mkdir(parents=True, exist_ok=True)

    labels_npz = seg_case_dir / f"{base}_labels.npz"

    np.savez_compressed(
        labels_npz,
        labels=pred_masks.astype(np.uint8),
        source_h5=np.array(str(source_h5)),
        model_name=np.array(SEGMENTATION_MODEL_NAME),
        model_path=np.array(str(model_path)),
    )

    print(f"[OK] saved labels: {labels_npz}")

    if SEGMENTATION_SAVE_FRAME_MASKS:
        for cls in classes_to_export:
            cls_mask = (pred_masks == cls).astype(np.uint8)
            cls_npz = seg_classes_dir / f"{base}_class{cls}_masks.npz"

            np.savez_compressed(
                cls_npz,
                masks=cls_mask,
                class_id=np.int32(cls),
                source_h5=np.array(str(source_h5)),
            )

            print(f"[OK] saved class masks: {cls_npz}")

    output_case_name = f"{case_name}{SEGMENTATION_OUTPUT_CASE_SUFFIX}_{SEGMENTATION_MODEL_NAME}"
    output_case_dir = FOR_RECON_DIR / output_case_name

    if SEGMENTATION_WRITE_TO_FOR_RECONSTRUCTION:
        output_case_dir.mkdir(parents=True, exist_ok=True)

        for cls in classes_to_export:
            cls_mask = (pred_masks == cls).astype(np.float32)

            if SEGMENTATION_MASKED_IMAGE_VALUE == "binary":
                segmented_img = cls_mask
            else:
                segmented_img = img.astype(np.float32) * cls_mask

            out_h5 = output_case_dir / f"{base}_class{cls}.h5"

            attrs = {
                "segmentation_model_name": SEGMENTATION_MODEL_NAME,
                "segmentation_model_path": str(model_path),
                "segmentation_meta_path": str(meta_path),
                "source_h5": str(source_h5),
                "class_id": int(cls),
                "masked_image_value": str(SEGMENTATION_MASKED_IMAGE_VALUE),
            }

            save_h5_image_and_poses(
                path=out_h5,
                image=segmented_img,
                poses=poses,
                spacing_xy=spacing_xy,
                image_key="img",
                pose_key="poses",
                attrs=attrs,
            )

            print(f"[OK] saved H5 for reconstruction: {out_h5}")

    summary = {
        "case_name": case_name,
        "output_case_name": output_case_name,
        "source_h5": str(source_h5),
        "labels_npz": str(labels_npz),
        "classes_to_export": classes_to_export,
        "model_name": SEGMENTATION_MODEL_NAME,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "write_to_for_reconstruction": bool(SEGMENTATION_WRITE_TO_FOR_RECONSTRUCTION),
        "output_case_dir": str(output_case_dir),
    }

    with open(seg_case_dir / f"{base}_segmentation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return output_case_name


def segment_file(path, case_name, model, meta, device, model_path, meta_path):
    path = Path(path)
    base = path.stem

    print("============================================================")
    print(f"[INFO] Segmenting file: {path}")
    print(f"[INFO] Case: {case_name}")
    print(f"[INFO] Model: {SEGMENTATION_MODEL_NAME}")

    img, poses, spacing_xy, resolved_image_key = load_h5_image_and_poses(
        path,
        image_key=SEGMENTATION_IMAGE_KEY,
    )

    model_input_size = get_model_input_size(
        meta,
        fallback_size=(256, 256),
    )

    print(f"[INFO] image shape: {img.shape}")
    print(f"[INFO] model input size: {model_input_size}")

    pred_masks = predict_masks_for_sequence(
        img_thw=img,
        model=model,
        model_input_size=model_input_size,
        device=device,
    )

    classes_to_export = get_classes_to_export(pred_masks)

    print(f"[INFO] predicted classes: {sorted([int(x) for x in np.unique(pred_masks)])}")
    print(f"[INFO] classes to export: {classes_to_export}")

    output_case_name = save_segmentation_outputs(
        case_name=case_name,
        source_h5=path,
        base=base,
        img=img,
        poses=poses,
        spacing_xy=spacing_xy,
        pred_masks=pred_masks,
        classes_to_export=classes_to_export,
        model_path=model_path,
        meta_path=meta_path,
    )

    return output_case_name


def segment_case(case_name, model, meta, device, model_path, meta_path):
    case_dir = FOR_RECON_DIR / case_name

    if not case_dir.exists():
        print(f"[WARN] Case dir does not exist: {case_dir}")
        return []

    files = list_h5_files(case_dir)

    if not files:
        print(f"[WARN] No H5 files found for segmentation in: {case_dir}")
        return []

    output_cases = []

    for path in files:
        output_case_name = segment_file(
            path=path,
            case_name=case_name,
            model=model,
            meta=meta,
            device=device,
            model_path=model_path,
            meta_path=meta_path,
        )
        output_cases.append(output_case_name)

    return sorted(list(set(output_cases)))


def segment_cases(case_names=None):
    case_names = normalize_case_names(case_names)

    model, meta, device, model_path, meta_path = load_segmentation_model()

    if case_names is None:
        case_dirs = sorted([p for p in FOR_RECON_DIR.iterdir() if p.is_dir()])
    else:
        case_dirs = [FOR_RECON_DIR / name for name in case_names]

    all_output_cases = []

    for case_dir in case_dirs:
        if not case_dir.exists():
            print(f"[WARN] Case dir does not exist: {case_dir}")
            continue

        output_cases = segment_case(
            case_name=case_dir.name,
            model=model,
            meta=meta,
            device=device,
            model_path=model_path,
            meta_path=meta_path,
        )

        all_output_cases.extend(output_cases)

    all_output_cases = sorted(list(set(all_output_cases)))

    print("============================================================")
    print(f"[INFO] Segmentation output cases: {all_output_cases}")

    return all_output_cases