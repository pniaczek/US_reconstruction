from pathlib import Path
import os
import numpy as np
import h5py
import torch
import vtk
from vtk.util import numpy_support as vtk_np
from scipy.ndimage import map_coordinates, spline_filter

from config import (
    FOR_RECON_DIR,
    RECONSTRUCTED_DIR,
    VOXEL_SIZE,
    PIXEL_SPACING,
    BACKGROUND_THRESHOLD,
    CENTER,
    CHUNK_PIXELS,
    IMAGE_KEY,
    N_NEAREST,
    SPLINE_ORDER,
    DISTANCE_EPS,
    DISTANCE_POWER,
    DEVICE,
)
from data_loading.h5_utils import list_h5_files, read_h5_meta, guess_keys

LIKELY_POSE_KEYS = [
    "tforms", "poses", "pose", "T", "transform", "transforms",
    "extrinsics", "world_T_probe", "probe_T_world", "sonda_swiat",
]


def find_pose_key(meta, n_frames):
    candidates = [k for k, (sh, _) in meta.items() if len(sh) == 3 and sh[-2:] == (4, 4)]
    if n_frames is not None:
        candidates_n = [k for k in candidates if int(meta[k][0][0]) == int(n_frames)]
    else:
        candidates_n = candidates

    for pref in LIKELY_POSE_KEYS:
        for k in candidates_n:
            name = os.path.basename(k).lower()
            if name.endswith(pref.lower()) or pref.lower() in name:
                return k

    if candidates_n:
        return candidates_n[0]
    return candidates[0]


def load_image_and_poses(path, image_key=None):
    meta = read_h5_meta(path)
    if image_key is None:
        image_key, _ = guess_keys(meta)

    with h5py.File(path, "r") as f:
        img = np.array(f[image_key])
        sx = float(f[image_key].attrs.get("spacing_x", 1.0))
        sy = float(f[image_key].attrs.get("spacing_y", 1.0))
        if "spacing" in f[image_key].attrs:
            sp = f[image_key].attrs["spacing"]
            if len(sp) >= 2:
                sx = float(sp[0])
                sy = float(sp[1])

        n_guess = img.shape[0] if img.ndim == 3 else None
        pose_key = find_pose_key(meta, n_guess)
        Ts = np.array(f[pose_key], dtype=np.float32)

    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3 and img.shape[0] != Ts.shape[0] and img.shape[-1] == Ts.shape[0]:
        img = np.moveaxis(img, -1, 0)

    return img.astype(np.float32), Ts.astype(np.float32), (sx, sy), image_key


def compute_bounds_for_sequence(Ts, W, H, sx, sy, margin=5.0, center=True):
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    corners_uv = np.array([
        [0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [cx, cy]
    ], dtype=np.float32)

    if center:
        xy = np.stack([
            (corners_uv[:, 0] - cx) * sx,
            (corners_uv[:, 1] - cy) * sy,
        ], axis=1)
    else:
        xy = np.stack([corners_uv[:, 0] * sx, corners_uv[:, 1] * sy], axis=1)

    pts_local = np.c_[xy, np.zeros((len(corners_uv), 1), dtype=np.float32), np.ones((len(corners_uv), 1), dtype=np.float32)]
    pts_world = np.einsum("tij,pj->tpi", Ts, pts_local)[:, :, :3]
    pmin = pts_world.reshape(-1, 3).min(axis=0) - margin
    pmax = pts_world.reshape(-1, 3).max(axis=0) + margin
    return pmin.astype(np.float32), pmax.astype(np.float32)


def save_as_vti(path_vti, vol_zyx, spacing_xyz, origin_xyz, array_name="intensity"):
    Dz, Dy, Dx = vol_zyx.shape
    img = vtk.vtkImageData()
    img.SetDimensions(Dx, Dy, Dz)
    img.SetSpacing(*map(float, spacing_xyz))
    img.SetOrigin(*map(float, origin_xyz))

    flat = np.ascontiguousarray(vol_zyx.transpose(2, 1, 0)).ravel(order="F")
    vtk_arr = vtk_np.numpy_to_vtk(num_array=flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_arr.SetName(array_name)
    img.GetPointData().SetScalars(vtk_arr)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path_vti))
    writer.SetInputData(img)
    writer.Write()


def prepare_spline_frames(img, order):
    if order <= 1:
        return img.astype(np.float32)

    out = np.empty_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        out[i] = spline_filter(img[i].astype(np.float32), order=order)
    return out


def auto_chunk_pixels(Tn, device):
    target_bytes = 700 * 1024 * 1024 if device.type == "cuda" else 256 * 1024 * 1024
    bytes_per_voxel = max(24 * int(Tn), 256)
    chunk = target_bytes // bytes_per_voxel
    chunk = max(2000, int(chunk))
    chunk = min(chunk, 100_000)
    return chunk


def make_world_points_block_flat(origin, voxel, Dx, Dy, Dz, start, end):
    ox, oy, oz = map(float, origin)
    vx, vy, vz = map(float, voxel)

    idx = np.arange(start, end, dtype=np.int64)
    yz_stride = Dx * Dz
    y = idx // yz_stride
    rem = idx % yz_stride
    x = rem // Dz
    z = rem % Dz

    xs = ox + (x.astype(np.float32) + 0.5) * vx
    ys = oy + (y.astype(np.float32) + 0.5) * vy
    zs = oz + (z.astype(np.float32) + 0.5) * vz
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def sample_bspline_grouped(spline_frames, frame_idx_flat, u_flat, v_flat, inside_flat, spline_order):
    vals_flat = np.zeros_like(u_flat, dtype=np.float32)
    valid_ids = np.flatnonzero(inside_flat)
    if valid_ids.size == 0:
        return vals_flat

    frame_valid = frame_idx_flat[valid_ids]
    unique_frames = np.unique(frame_valid)

    for fi in unique_frames:
        loc = valid_ids[frame_valid == fi]
        coords = np.vstack([v_flat[loc], u_flat[loc]]).astype(np.float32)
        sampled = map_coordinates(
            spline_frames[int(fi)],
            coords,
            order=spline_order,
            mode="nearest",
            prefilter=False if spline_order > 1 else True,
        ).astype(np.float32)
        vals_flat[loc] = sampled

    return vals_flat


def bspline_reconstruct_hybrid_torch(
    img,
    Ts,
    sx,
    sy,
    origin,
    voxel,
    background_threshold=None,
    center=True,
    chunk_pixels=None,
    binary_input=False,
    n_nearest=4,
    spline_order=3,
    distance_eps=1e-6,
    distance_power=1.0,
    device_str="cuda",
):
    device = torch.device(device_str)
    Tn, H, W = img.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    n_nearest = max(1, min(int(n_nearest), Tn))
    spline_order = int(np.clip(spline_order, 0, 5))

    pmin, pmax = compute_bounds_for_sequence(Ts, W, H, sx, sy, margin=5.0, center=center)
    origin = np.asarray(origin, dtype=np.float32)
    voxel = np.asarray(voxel, dtype=np.float32)

    size = np.maximum(np.ceil((pmax - origin) / voxel).astype(np.int32), 1)
    Dx, Dy, Dz = int(size[0]), int(size[1]), int(size[2])
    N = Dx * Dy * Dz

    if chunk_pixels is None or chunk_pixels <= 0:
        chunk_pixels = auto_chunk_pixels(Tn, device)

    print(
        f"[INFO] B-spline reconstruction: volume=(Dz={Dz}, Dy={Dy}, Dx={Dx}), "
        f"voxels={N}, chunk={chunk_pixels}, n_nearest={n_nearest}, "
        f"spline_order={spline_order}, device={device}"
    )

    Ts_t = torch.from_numpy(Ts).to(device=device, dtype=torch.float32)
    invTs_t = torch.linalg.inv(Ts_t).contiguous()
    spline_frames = prepare_spline_frames(img, spline_order)

    out_flat = np.zeros((N,), dtype=np.float32)
    wts_flat = np.zeros((N,), dtype=np.float32)

    for s in range(0, N, chunk_pixels):
        e = min(s + chunk_pixels, N)
        pts = make_world_points_block_flat(origin, voxel, Dx, Dy, Dz, s, e)
        M = pts.shape[0]

        pts_t = torch.from_numpy(pts).to(device=device, dtype=torch.float32)
        ones_t = torch.ones((M, 1), dtype=torch.float32, device=device)
        pts_h_t = torch.cat([pts_t, ones_t], dim=1)
        local = torch.einsum("tij,mj->mti", invTs_t, pts_h_t)

        x_local_all = local[..., 0]
        y_local_all = local[..., 1]
        abs_z_all = torch.abs(local[..., 2])

        nearest_dist_t, nearest_idx_t = torch.topk(abs_z_all, k=n_nearest, dim=1, largest=False, sorted=False)
        x_local_t = torch.gather(x_local_all, dim=1, index=nearest_idx_t)
        y_local_t = torch.gather(y_local_all, dim=1, index=nearest_idx_t)

        if center:
            u_t = x_local_t / float(sx) + float(cx)
            v_t = y_local_t / float(sy) + float(cy)
        else:
            u_t = x_local_t / float(sx)
            v_t = y_local_t / float(sy)

        inside_t = (u_t >= 0.0) & (u_t <= (W - 1)) & (v_t >= 0.0) & (v_t <= (H - 1))

        nearest_idx = nearest_idx_t.detach().cpu().numpy().astype(np.int32, copy=False)
        nearest_dist = nearest_dist_t.detach().cpu().numpy().astype(np.float32, copy=False)
        u = u_t.detach().cpu().numpy().astype(np.float32, copy=False)
        v = v_t.detach().cpu().numpy().astype(np.float32, copy=False)
        inside = inside_t.detach().cpu().numpy()

        frame_idx_flat = nearest_idx.reshape(-1)
        dist_flat = nearest_dist.reshape(-1)
        u_flat = u.reshape(-1)
        v_flat = v.reshape(-1)
        inside_flat = inside.reshape(-1)

        vals_flat = sample_bspline_grouped(
            spline_frames=spline_frames,
            frame_idx_flat=frame_idx_flat,
            u_flat=u_flat,
            v_flat=v_flat,
            inside_flat=inside_flat,
            spline_order=spline_order,
        )

        vals = vals_flat.reshape(M, n_nearest)
        z_abs = dist_flat.reshape(M, n_nearest)

        if binary_input:
            thr = 0.5 if background_threshold is None else background_threshold
            valid = inside & (vals > thr)
            vals_use = valid.astype(np.float32)
        else:
            valid = inside if background_threshold is None else (inside & (vals > background_threshold))
            vals_use = np.where(valid, vals, 0.0).astype(np.float32)

        w = np.zeros((M, n_nearest), dtype=np.float32)
        w[valid] = 1.0 / np.power(z_abs[valid] + distance_eps, distance_power)

        sum_vals = np.sum(vals_use * w, axis=1, dtype=np.float32)
        sum_wts = np.sum(w, axis=1, dtype=np.float32)
        nz = sum_wts > 0

        block_out = np.zeros((M,), dtype=np.float32)
        block_out[nz] = sum_vals[nz] / sum_wts[nz]
        out_flat[s:e] = block_out
        wts_flat[s:e] = sum_wts

        print(f"[INFO] processed voxels: {e}/{N}")

    out = out_flat.reshape(Dy, Dx, Dz).transpose(2, 0, 1).copy()
    wts = wts_flat.reshape(Dy, Dx, Dz).transpose(2, 0, 1).copy()
    return out, wts


def reconstruct_bspline_for_file(path, out_root, voxel=(1.0, 1.0, 1.0), pixel_spacing=None,
                                 background_threshold=None, center=True, chunk_pixels=None,
                                 image_key=None, n_nearest=4, spline_order=3,
                                 distance_eps=1e-6, distance_power=1.0, device_str="cuda"):
    path = Path(path)
    base = path.stem
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    img, Ts, (sx_attr, sy_attr), resolved_image_key = load_image_and_poses(path, image_key=image_key)
    H = img.shape[1]
    W = img.shape[2]

    sx = pixel_spacing[0] if pixel_spacing else sx_attr
    sy = pixel_spacing[1] if pixel_spacing else sy_attr

    key_l = resolved_image_key.lower() if resolved_image_key is not None else ""
    is_mask = ("mask" in key_l) or ("seg" in key_l) or ("label" in key_l)

    if background_threshold is None and is_mask:
        background_threshold = 0.5

    pmin, _ = compute_bounds_for_sequence(Ts, W, H, sx, sy, margin=5.0, center=center)
    voxel = np.asarray(voxel, dtype=np.float32)
    origin = pmin.astype(np.float32)

    out, wts = bspline_reconstruct_hybrid_torch(
        img=img,
        Ts=Ts,
        sx=sx,
        sy=sy,
        origin=origin,
        voxel=voxel,
        background_threshold=background_threshold,
        center=center,
        chunk_pixels=chunk_pixels,
        binary_input=is_mask,
        n_nearest=n_nearest,
        spline_order=spline_order,
        distance_eps=distance_eps,
        distance_power=distance_power,
        device_str=device_str,
    )

    vti_path = out_root / f"{base}.vti"
    npz_path = out_root / f"{base}.npz"

    save_as_vti(vti_path, out, voxel, origin, "mask" if is_mask else "intensity")
    np.savez_compressed(
        npz_path,
        volume=out.astype(np.float32),
        weights=wts.astype(np.float32),
        voxel_size=voxel.astype(np.float32),
        origin=origin.astype(np.float32),
        n_nearest=np.int32(n_nearest),
        spline_order=np.int32(spline_order),
        distance_power=np.float32(distance_power),
        distance_eps=np.float32(distance_eps),
    )

    print(f"[OK] {path.name} -> {vti_path.name} / {npz_path.name}")


def reconstruct_case(case_name):
    case_dir = FOR_RECON_DIR / case_name
    files = list_h5_files(case_dir)
    if not files:
        print(f"[WARN] No H5 files found for case: {case_name}")
        return

    out_dir = RECONSTRUCTED_DIR / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        reconstruct_bspline_for_file(
            path=path,
            out_root=out_dir,
            voxel=VOXEL_SIZE,
            pixel_spacing=PIXEL_SPACING,
            background_threshold=BACKGROUND_THRESHOLD,
            center=CENTER,
            chunk_pixels=CHUNK_PIXELS,
            image_key=IMAGE_KEY,
            n_nearest=N_NEAREST,
            spline_order=SPLINE_ORDER,
            distance_eps=DISTANCE_EPS,
            distance_power=DISTANCE_POWER,
            device_str=DEVICE,
        )


def reconstruct_cases(case_names=None):
    case_dirs = sorted([p for p in FOR_RECON_DIR.iterdir() if p.is_dir()])
    if case_names is not None:
        case_dirs = [p for p in case_dirs if p.name in case_names]

    if not case_dirs:
        print(f"[WARN] No cases found in: {FOR_RECON_DIR}")
        return

    for case_dir in case_dirs:
        reconstruct_case(case_dir.name)
