from pathlib import Path

import numpy as np
import napari

from config import (
    RECONSTRUCTED_DIR,
    GROUND_TRUTH_DIR,
    SHOW_RECONSTRUCTION,
    SHOW_GROUND_TRUTH_VOLUME,
    SHOW_SEGMENTATION_LABELS,
    SHOW_SEGMENTATION_CLASSES,
    SHOW_SURFACES,
    NAPARI_NDISPLAY,
)

from visualization.common import (
    load_volume,
    find_case_name,
)


def find_reconstruction_files(recon_dir: Path):
    """
    Szuka rekonstrukcji zarówno w starym, jak i nowym układzie.

    Stary układ:
        data/reconstructed/<case>/*.npz
        data/reconstructed/<case>/*.vti

    Nowy układ:
        data/reconstructed/<case>/<method>/*.npz
        data/reconstructed/<case>/<method>/*.vti
    """
    files = []

    if not recon_dir.exists():
        return files

    # Stary układ: pliki bezpośrednio w folderze case'a.
    files.extend(sorted(recon_dir.glob("*.npz")))
    files.extend(sorted(recon_dir.glob("*.vti")))

    # Nowy układ: podfoldery metod.
    method_dirs = sorted([p for p in recon_dir.iterdir() if p.is_dir()])

    for method_dir in method_dirs:
        files.extend(sorted(method_dir.glob("*.npz")))
        files.extend(sorted(method_dir.glob("*.vti")))

    # Preferuj NPZ względem VTI, żeby nie dublować tej samej rekonstrukcji.
    # Jeżeli istnieje base.npz i base.vti, zostawiamy tylko base.npz.
    selected = {}
    for path in files:
        key = str(path.with_suffix(""))
        if key not in selected:
            selected[key] = path
        elif selected[key].suffix.lower() != ".npz" and path.suffix.lower() == ".npz":
            selected[key] = path

    return list(selected.values())


def layer_name_from_reconstruction_path(path: Path, recon_dir: Path):
    """
    Czytelna nazwa warstwy:
      bspline/kosartur_from_dicom
      voxel_nearest/kosartur_from_dicom
      reconstruction/kosartur_from_dicom
    """
    try:
        rel = path.relative_to(recon_dir)
    except ValueError:
        return path.stem

    parts = rel.parts

    if len(parts) >= 2:
        method = parts[0]
        return f"{method}/{path.stem}"

    return f"reconstruction/{path.stem}"


def add_reconstruction_layers(viewer, recon_dir: Path):
    recon_files = find_reconstruction_files(recon_dir)

    if not recon_files:
        print(f"[WARN] No reconstruction files found in: {recon_dir}")
        return

    print(f"[INFO] Found reconstruction files: {len(recon_files)}")

    for path in recon_files:
        print(f"[INFO] loading reconstruction: {path}")

        try:
            vol, voxel = load_volume(path)
        except Exception as e:
            print(f"[WARN] Could not load reconstruction {path}: {e}")
            continue

        layer_name = layer_name_from_reconstruction_path(path, recon_dir)

        viewer.add_image(
            vol.astype(np.float32),
            name=layer_name,
            scale=voxel,
            colormap="gray",
            rendering="mip",
            opacity=1.0,
        )


def add_surface_npz(viewer, path: Path):
    data = np.load(path, allow_pickle=True)

    verts = data["verts"]
    faces = data["faces"]

    if "values" in data:
        values = data["values"]
    else:
        values = np.ones((verts.shape[0],), dtype=np.float32)

    viewer.add_surface(
        (verts, faces, values),
        name=Path(path).stem,
        opacity=0.6,
        shading="smooth",
    )


def add_surface_layers(viewer, gt_dir: Path, recon_dir: Path):
    """
    Obsługuje surface'y w:
      ground_truth/<case>/surfaces
      reconstructed/<case>/surfaces
      reconstructed/<case>/<method>/surfaces
    """
    surface_dirs = []

    gt_surface_dir = gt_dir / "surfaces"
    if gt_surface_dir.exists():
        surface_dirs.append(gt_surface_dir)

    recon_surface_dir = recon_dir / "surfaces"
    if recon_surface_dir.exists():
        surface_dirs.append(recon_surface_dir)

    if recon_dir.exists():
        for method_dir in sorted([p for p in recon_dir.iterdir() if p.is_dir()]):
            method_surface_dir = method_dir / "surfaces"
            if method_surface_dir.exists():
                surface_dirs.append(method_surface_dir)

    for surface_dir in surface_dirs:
        for path in sorted(surface_dir.glob("*_surface.npz")):
            print(f"[INFO] loading surface: {path}")
            add_surface_npz(viewer, path)


def visualize_case_napari(case_name=None):
    case_name = find_case_name(case_name)

    if case_name is None:
        raise ValueError(
            "case_name is None. Ustaw VIS_CASE w config.py albo CASE_NAMES = ['nazwa_case']."
        )

    recon_dir = RECONSTRUCTED_DIR / case_name
    gt_dir = GROUND_TRUTH_DIR / case_name

    print("============================================================")
    print(f"[INFO] Napari visualization")
    print(f"[INFO] case_name : {case_name}")
    print(f"[INFO] recon_dir : {recon_dir}")
    print(f"[INFO] gt_dir    : {gt_dir}")

    viewer = napari.Viewer(ndisplay=NAPARI_NDISPLAY)

    if SHOW_RECONSTRUCTION:
        add_reconstruction_layers(
            viewer=viewer,
            recon_dir=recon_dir,
        )

    if SHOW_GROUND_TRUTH_VOLUME:
        gt_vol_path = gt_dir / f"{case_name}_volume.npz"

        if gt_vol_path.exists():
            print(f"[INFO] loading ground truth volume: {gt_vol_path}")
            vol, voxel = load_volume(gt_vol_path)

            viewer.add_image(
                vol.astype(np.float32),
                name="ground_truth_volume",
                scale=voxel,
                colormap="green",
                rendering="mip",
                opacity=0.35,
            )
        else:
            print(f"[WARN] Ground truth volume not found: {gt_vol_path}")

    if SHOW_SEGMENTATION_LABELS:
        seg_path = gt_dir / f"{case_name}_segmentation_labels.npz"

        if seg_path.exists():
            print(f"[INFO] loading segmentation labels: {seg_path}")
            vol, voxel = load_volume(seg_path)

            viewer.add_labels(
                vol.astype(np.int32),
                name="segmentation_labels",
                scale=voxel,
                opacity=0.45,
            )
        else:
            print(f"[WARN] Segmentation labels not found: {seg_path}")

    if SHOW_SEGMENTATION_CLASSES:
        seg_classes_dir = gt_dir / "seg_classes"

        if seg_classes_dir.exists():
            for path in sorted(seg_classes_dir.glob("*.npz")):
                print(f"[INFO] loading segmentation class: {path}")
                vol, voxel = load_volume(path)

                viewer.add_image(
                    vol.astype(np.float32),
                    name=f"gt/{path.stem}",
                    scale=voxel,
                    rendering="iso",
                    opacity=0.45,
                )
        else:
            print(f"[WARN] Segmentation classes dir not found: {seg_classes_dir}")

    if SHOW_SURFACES:
        add_surface_layers(
            viewer=viewer,
            gt_dir=gt_dir,
            recon_dir=recon_dir,
        )

    print("[INFO] Starting napari...")
    napari.run()