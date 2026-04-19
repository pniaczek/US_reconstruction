from pathlib import Path
import matplotlib.pyplot as plt

from config import RECONSTRUCTED_DIR, GROUND_TRUTH_DIR
from visualization.common import load_volume, find_case_name, find_first_existing


def visualize_case_simple(case_name=None):
    case_name = find_case_name(case_name)
    recon_dir = RECONSTRUCTED_DIR / case_name
    gt_dir = GROUND_TRUTH_DIR / case_name

    recon_path = find_first_existing([recon_dir / f"{case_name}_from_dicom.npz", *sorted(recon_dir.glob("*.npz")), *sorted(recon_dir.glob("*.vti"))])
    gt_path = gt_dir / f"{case_name}_volume.npz"
    seg_path = gt_dir / f"{case_name}_segmentation_labels.npz"

    items = []
    if recon_path is not None and recon_path.exists():
        items.append(("reconstruction", load_volume(recon_path)[0]))
    if gt_path.exists():
        items.append(("ground_truth", load_volume(gt_path)[0]))
    if seg_path.exists():
        items.append(("segmentation", load_volume(seg_path)[0]))

    if not items:
        raise RuntimeError(f"No volumes found for case: {case_name}")

    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 5))
    if len(items) == 1:
        axes = [axes]

    for ax, (title, vol) in zip(axes, items):
        z = vol.shape[0] // 2
        ax.imshow(vol[z], cmap="gray")
        ax.set_title(f"{title} | z={z}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
