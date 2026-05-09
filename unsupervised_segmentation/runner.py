from pathlib import Path
import os
import shutil
import subprocess
import textwrap

import h5py
import numpy as np
from PIL import Image

from config import (
    ROOT,
    FOR_RECON_DIR,
    UNSUPERVISED_REPO_DIR,
    UNSUPERVISED_REPO_URL,
    UNSUPERVISED_DINOV2_REPO_URL,
    UNSUPERVISED_DATASET_NAME,
    UNSUPERVISED_CASE_NAME,
    UNSUPERVISED_IMAGE_KEY,
    UNSUPERVISED_FRAME_START,
    UNSUPERVISED_FRAME_END,
    UNSUPERVISED_FRAME_STEP,
    UNSUPERVISED_PNG_NORMALIZATION,
    UNSUPERVISED_P_LOW,
    UNSUPERVISED_P_HIGH,
    UNSUPERVISED_N_CLASSES,
    UNSUPERVISED_SEGMENTS_NUM,
    UNSUPERVISED_CLUSTERS_NUM,
    UNSUPERVISED_WANDB_PROJECT,
    UNSUPERVISED_WANDB_ENTITY,
    UNSUPERVISED_WANDB_MODE,
    UNSUPERVISED_WANDB_KEY,
    UNSUPERVISED_WANDB_TAG,
    UNSUPERVISED_USE_WANDB_DISABLED,
    UNSUPERVISED_PYTHON_EXECUTABLE,
    UNSUPERVISED_OVERWRITE_DATASET,
    UNSUPERVISED_OVERWRITE_CONFIGS,
)

from data_loading.h5_utils import list_h5_files, read_h5_meta, guess_keys


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_case_names(case_names):
    if case_names is None:
        return None

    if isinstance(case_names, str):
        return [case_names]

    return list(case_names)


def run_command(cmd, cwd=None, env=None, check=True):
    print("============================================================")
    print("[CMD]", " ".join(str(x) for x in cmd))
    if cwd is not None:
        print("[CWD]", cwd)

    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")

    return result.returncode


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_configs_root(repo_dir: Path):
    """
    Repo w README używa configs/.
    Starsze kopie/notebooki czasem miały config/.
    Wybieramy istniejące, a jeśli nie istnieje żadne, tworzymy configs/.
    """
    configs = repo_dir / "configs"
    config = repo_dir / "config"

    if configs.exists():
        return configs

    if config.exists():
        return config

    configs.mkdir(parents=True, exist_ok=True)
    return configs


# ============================================================
# GIT SETUP
# ============================================================

def ensure_external_repo():
    repo_dir = Path(UNSUPERVISED_REPO_DIR)

    if repo_dir.exists():
        print(f"[OK] External repo exists: {repo_dir}")
        return repo_dir

    ensure_dir(repo_dir.parent)

    run_command(
        [
            "git",
            "clone",
            UNSUPERVISED_REPO_URL,
            str(repo_dir),
        ],
        cwd=repo_dir.parent,
        check=True,
    )

    return repo_dir


def ensure_dinov2_repo(repo_dir: Path):
    """
    Zgodnie z Twoim notebookiem:
      cd UnsupervisedSegmentor4Ultrasound
      git clone https://github.com/3cology/dinov2_with_attention_extraction.git

    Dodatkowo dodajemy ten folder później do PYTHONPATH.
    """
    dinov2_root = repo_dir / "dinov2_with_attention_extraction"

    if dinov2_root.exists():
        print(f"[OK] DINOv2 repo exists: {dinov2_root}")
    else:
        run_command(
            [
                "git",
                "clone",
                UNSUPERVISED_DINOV2_REPO_URL,
                str(dinov2_root),
            ],
            cwd=repo_dir,
            check=True,
        )

    # Często pomaga przy importach pakietowych.
    init_paths = [
        dinov2_root / "__init__.py",
        dinov2_root / "dinov2" / "__init__.py",
        dinov2_root / "dinov2" / "models" / "__init__.py",
        dinov2_root / "dinov2" / "layers" / "__init__.py",
    ]

    for path in init_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        print(f"[OK] touch {path}")

    return dinov2_root


# ============================================================
# H5 -> PNG DATASET
# ============================================================

def load_h5_image(path: Path, image_key=None):
    meta = read_h5_meta(path)

    if image_key is None:
        image_key, _ = guess_keys(meta)

    if image_key is None:
        raise RuntimeError(f"Nie znaleziono datasetu z obrazami w H5: {path}")

    with h5py.File(path, "r") as f:
        img = np.asarray(f[image_key])

    if img.ndim == 2:
        img = img[None, ...]

    if img.ndim != 3:
        raise ValueError(f"Oczekiwano obrazu (N,H,W), dostałem: {img.shape}")

    return img.astype(np.float32), image_key


def normalize_frame_to_uint8(frame):
    x = frame.astype(np.float32)

    if UNSUPERVISED_PNG_NORMALIZATION == "raw":
        return np.clip(x, 0, 255).astype(np.uint8)

    if UNSUPERVISED_PNG_NORMALIZATION == "minmax":
        mn = float(np.nanmin(x))
        mx = float(np.nanmax(x))
        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.uint8)
        y = (x - mn) / (mx - mn) * 255.0
        return np.clip(y, 0, 255).astype(np.uint8)

    # percentile
    lo = float(np.percentile(x, UNSUPERVISED_P_LOW))
    hi = float(np.percentile(x, UNSUPERVISED_P_HIGH))

    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    y = (np.clip(x, lo, hi) - lo) / (hi - lo) * 255.0

    return np.clip(y, 0, 255).astype(np.uint8)


def get_external_dataset_dir(repo_dir: Path):
    return repo_dir / "data" / UNSUPERVISED_DATASET_NAME / UNSUPERVISED_CASE_NAME


def write_list_file(path: Path, filenames):
    with open(path, "w", encoding="utf-8") as f:
        for name in filenames:
            f.write(f"{name}\n")


def prepare_dataset_from_h5(repo_dir: Path, case_name: str):
    case_dir = FOR_RECON_DIR / case_name

    if not case_dir.exists():
        raise FileNotFoundError(f"Brak case dir: {case_dir}")

    h5_files = list_h5_files(case_dir)

    if not h5_files:
        raise RuntimeError(f"Brak plików H5 w: {case_dir}")

    if len(h5_files) > 1:
        print("[WARN] W case jest więcej niż jeden H5. Używam pierwszego:")
        for p in h5_files:
            print("  -", p)

    h5_path = Path(h5_files[0])

    dataset_dir = get_external_dataset_dir(repo_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    lists_dir = dataset_dir / "lists"

    if dataset_dir.exists() and UNSUPERVISED_OVERWRITE_DATASET:
        shutil.rmtree(dataset_dir)

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    lists_dir.mkdir(parents=True, exist_ok=True)

    img, resolved_key = load_h5_image(
        h5_path,
        image_key=UNSUPERVISED_IMAGE_KEY,
    )

    n_frames = img.shape[0]

    start = int(UNSUPERVISED_FRAME_START)
    end = n_frames if UNSUPERVISED_FRAME_END is None else min(int(UNSUPERVISED_FRAME_END), n_frames)
    step = int(UNSUPERVISED_FRAME_STEP)

    filenames = []

    print("============================================================")
    print("[INFO] Preparing unsupervised dataset")
    print(f"[INFO] case_name      : {case_name}")
    print(f"[INFO] source h5      : {h5_path}")
    print(f"[INFO] image key      : {resolved_key}")
    print(f"[INFO] image shape    : {img.shape}")
    print(f"[INFO] target dataset : {dataset_dir}")
    print(f"[INFO] frames         : {start}:{end}:{step}")

    out_idx = 0

    for frame_idx in range(start, end, step):
        frame_u8 = normalize_frame_to_uint8(img[frame_idx])

        filename = f"frame_{out_idx:06d}.png"
        out_path = images_dir / filename

        Image.fromarray(frame_u8, mode="L").save(out_path)

        filenames.append(filename)
        out_idx += 1

        if out_idx % 50 == 0:
            print(f"[INFO] exported frames: {out_idx}")

    # Repo oczekuje listy plików w lists/.
    write_list_file(lists_dir / "images.txt", filenames)
    write_list_file(lists_dir / "all.txt", filenames)
    write_list_file(lists_dir / "train.txt", filenames)
    write_list_file(lists_dir / "val.txt", filenames)
    write_list_file(lists_dir / "test.txt", filenames)

    print(f"[OK] exported PNG frames: {len(filenames)}")
    print(f"[OK] dataset dir: {dataset_dir}")

    return dataset_dir


# ============================================================
# CONFIG GENERATION
# ============================================================

def yaml_list(values):
    return "[" + ",".join(str(v) for v in values) + "]"


def write_text_if_needed(path: Path, content: str):
    if path.exists() and not UNSUPERVISED_OVERWRITE_CONFIGS:
        print(f"[OK] config exists, not overwritten: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] wrote config: {path}")


def generate_wandb_defaults(configs_root: Path):
    content = f"""setup:
    project: {UNSUPERVISED_WANDB_PROJECT}
    entity: {UNSUPERVISED_WANDB_ENTITY}
    mode: {UNSUPERVISED_WANDB_MODE}
key: {UNSUPERVISED_WANDB_KEY}
tag: '{UNSUPERVISED_WANDB_TAG}'
watch:
    log: all
    log_freq: 1
mode: local
"""
    write_text_if_needed(
        configs_root / "wandb" / "defaults.yaml",
        content,
    )


def generate_sweep_config(configs_root: Path):
    sweep_name = f"{UNSUPERVISED_CASE_NAME}_clusters"

    content = f"""name: {sweep_name}
seg_for_eval: ['crf_multi_region']
method: grid
count:
simple: True
sweep_id: null
config:
    # generic
    segments_num: {yaml_list(UNSUPERVISED_SEGMENTS_NUM)}
    clusters_num: {yaml_list(UNSUPERVISED_CLUSTERS_NUM)}

    # postprocessing (CRF)
    crf:
        num_classes: [10]
        w1: [15]
        alpha: [7]
        beta: [10]
        w2: [5]
        gamma: [5]
        it: [10]
"""
    write_text_if_needed(
        configs_root / "sweep" / f"{sweep_name}.yaml",
        content,
    )

    return sweep_name


def generate_dataset_config(configs_root: Path):
    """
    Tworzymy odpowiednik forearm.yaml, ale automatycznie, np.:
      configs/dataset/kosartur_unsupervised.yaml
    """
    dataset_root = f"{UNSUPERVISED_DATASET_NAME}/{UNSUPERVISED_CASE_NAME}"

    content = f"""name: {UNSUPERVISED_CASE_NAME}
dataset_root: {dataset_root}
dataset_type: folders
images_root: images
list: lists/images.txt
gt_dir: labels
pred_dir: ""
n_classes: {UNSUPERVISED_N_CLASSES}
features_dir: ""
preprocessed_dir: ""
derained_dir: ""
eigenseg_dir: ""
"""
    write_text_if_needed(
        configs_root / "dataset" / f"{UNSUPERVISED_CASE_NAME}.yaml",
        content,
    )

    return UNSUPERVISED_CASE_NAME


def generate_vis_selected_config(configs_root: Path):
    content = """segmaps: true
eigen: false
crf_segmaps: false
multiregion_segmaps: false
crf_multi_region: false
dino_attn_maps: false
"""
    write_text_if_needed(
        configs_root / "vis" / "selected.yaml",
        content,
    )


def generate_run_pipeline_script(repo_dir: Path, dataset_config_name: str, sweep_config_name: str):
    """
    README repo pokazuje uruchomienie:
      cd deep-spectral-segmentation
      python -m pipeline.pipeline_sweep_subfolders vis=selected pipeline_steps=defaults dataset=thyroid wandb.tag=test sweep=defaults

    Generujemy analogiczny skrypt dla Twojego datasetu.
    """
    script = f"""#!/usr/bin/env bash
set -e

cd deep-spectral-segmentation

export WANDB_API_KEY={UNSUPERVISED_WANDB_KEY}
export WANDB_CONFIG_DIR=/tmp/
export WANDB_CACHE_DIR=/tmp/
export WANDB_AGENT_MAX_INITIAL_FAILURE=20
export WANDB__SERVICE_WAIT=600
export XFORMERS_DISABLED=True
export WANDB_MODE={UNSUPERVISED_WANDB_MODE}

{UNSUPERVISED_PYTHON_EXECUTABLE} -m pipeline.pipeline_sweep_subfolders \\
    vis=selected \\
    pipeline_steps=defaults \\
    dataset={dataset_config_name} \\
    wandb=defaults \\
    wandb.tag={UNSUPERVISED_WANDB_TAG} \\
    sweep={sweep_config_name}
"""
    path = repo_dir / "run_pipeline.sh"

    write_text_if_needed(path, script)

    try:
        path.chmod(0o755)
    except Exception:
        pass

    return path


def generate_all_configs(repo_dir: Path):
    configs_root = get_configs_root(repo_dir)

    generate_wandb_defaults(configs_root)
    sweep_config_name = generate_sweep_config(configs_root)
    dataset_config_name = generate_dataset_config(configs_root)
    generate_vis_selected_config(configs_root)
    run_script = generate_run_pipeline_script(
        repo_dir=repo_dir,
        dataset_config_name=dataset_config_name,
        sweep_config_name=sweep_config_name,
    )

    return {
        "configs_root": configs_root,
        "sweep_config_name": sweep_config_name,
        "dataset_config_name": dataset_config_name,
        "run_script": run_script,
    }


# ============================================================
# RUN EXTERNAL PIPELINE
# ============================================================

def run_external_pipeline(repo_dir: Path, dinov2_root: Path):
    env = os.environ.copy()

    pythonpath_parts = [
        str(dinov2_root),
        str(repo_dir / "deep-spectral-segmentation"),
    ]

    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        pythonpath_parts.append(old_pythonpath)

    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    if UNSUPERVISED_USE_WANDB_DISABLED:
        env["WANDB_MODE"] = "disabled"

    print("============================================================")
    print("[INFO] Running external unsupervised pipeline")
    print(f"[INFO] repo       : {repo_dir}")
    print(f"[INFO] dinov2     : {dinov2_root}")
    print(f"[INFO] PYTHONPATH : {env['PYTHONPATH']}")
    print("============================================================")

    run_command(
        ["bash", "run_pipeline.sh"],
        cwd=repo_dir,
        env=env,
        check=True,
    )

    print("[OK] Unsupervised pipeline finished")


# ============================================================
# MAIN ENTRY
# ============================================================

def run_unsupervised_segmentation(case_names=None):
    case_names = normalize_case_names(case_names)

    if case_names is None:
        case_dirs = sorted([p for p in FOR_RECON_DIR.iterdir() if p.is_dir()])
        case_names = [p.name for p in case_dirs]

    if not case_names:
        raise RuntimeError("Brak case_names do unsupervised segmentation.")

    if len(case_names) > 1:
        print("[WARN] Na razie external pipeline przygotowuje jeden dataset na raz.")
        print("[WARN] Używam pierwszego case'a:", case_names[0])

    case_name = case_names[0]

    repo_dir = ensure_external_repo()
    dinov2_root = ensure_dinov2_repo(repo_dir)

    prepare_dataset_from_h5(
        repo_dir=repo_dir,
        case_name=case_name,
    )

    generated = generate_all_configs(repo_dir)

    print("============================================================")
    print("[INFO] Generated config summary")
    for k, v in generated.items():
        print(f"[INFO] {k}: {v}")

    run_external_pipeline(
        repo_dir=repo_dir,
        dinov2_root=dinov2_root,
    )