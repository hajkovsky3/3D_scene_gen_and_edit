"""Accuracy of 01_sfm camera poses against a reference (docs/evaluation.md).

The reference is either a COLMAP model with known poses (e.g. Mip-NeRF 360 `sparse/0`,
images matched by file name of the input photo) or the 01_sfm output of another workspace
(frames matched by source file + frame index, so different 00_ingest runs can be compared).

Poses are compared after a similarity alignment of the camera centers (Umeyama inside a
least-median-of-squares search, refit on inliers), so the arbitrary gauge of SfM does not matter:
- rotation error: angle between aligned and reference camera orientation [deg],
- center error: distance of aligned and reference centers, relative to the RMS spread of
  the reference centers [%],
- focal error: relative difference of fx / width.

    python -m scenepipe.eval.sfm_poses --scene room_test --ref-colmap workspace/_datasets/.../sparse/0
    python -m scenepipe.eval.sfm_poses --scene izba_new --ref-scene izba_ref
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenepipe.core.cameras import load_cameras
from scenepipe.core.workspace import Workspace

MIN_OUTLIER_ERR = 0.01  # center error (relative to spread) below which a camera is never an outlier
LMEDS_SAMPLES = 500

# Pose dict entry: key -> (camera-to-world rotation [3,3], center [3], fx / width)
Poses = dict[tuple[str, int | None], tuple[np.ndarray, np.ndarray, float]]


def workspace_poses(ws: Workspace) -> Poses:
    """Poses of a 01_sfm output, keyed by (source file name, source_index)."""
    frames = json.loads(ws.abs("00_ingest/frames.json").read_text(encoding="utf-8"))["frames"]
    key = {f["frame_id"]: (Path(f["source"]).name, f["source_index"]) for f in frames}
    out: Poses = {}
    for c in load_cameras(ws.abs("01_sfm/cameras.json")):
        out[key[c.frame_id]] = (c.world_to_camera[:3, :3].T, c.center, c.fx / c.width)
    return out


def colmap_poses(sparse: Path) -> Poses:
    """Poses of a COLMAP model, keyed by (image file name, None)."""
    import pycolmap

    rec = pycolmap.Reconstruction(sparse)
    out: Poses = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        m = np.asarray(img.cam_from_world().matrix(), np.float64)
        cam = rec.camera(img.camera_id)
        fx = float(cam.focal_length_x if hasattr(cam, "focal_length_x") else cam.params[0])
        out[(Path(img.name).name, None)] = (m[:3, :3].T, -m[:3, :3].T @ m[:3, 3], fx / cam.width)
    return out


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimizing ||dst - (s R src + t)||^2 (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1
    r = u @ d @ vt
    s = float(np.trace(np.diag(sig) @ d) / xs.var(0).sum())
    return s, r, mu_d - s * r @ mu_s


def compare(est: Poses, ref: Poses, outlier_factor: float = 5.0) -> dict:
    keys = sorted(set(est) & set(ref), key=str)
    if len(keys) < 3:
        raise ValueError(f"only {len(keys)} common frames")
    c_est = np.stack([est[k][1] for k in keys])
    c_ref = np.stack([ref[k][1] for k in keys])
    spread = float(np.sqrt(((c_ref - c_ref.mean(0)) ** 2).sum(1).mean()))

    def errors(sim: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
        s, r, t = sim
        return np.linalg.norm((s * (r @ c_est.T)).T + t - c_ref, axis=1) / spread

    # Least median of squares over random triples, so that a few grossly mislocalized
    # cameras cannot bias the alignment; then refit on the inliers.
    rng = np.random.default_rng(0)
    best = umeyama(c_est, c_ref)
    best_med = float(np.median(errors(best)))
    for _ in range(LMEDS_SAMPLES):
        idx = rng.choice(len(keys), 3, replace=False)
        sim = umeyama(c_est[idx], c_ref[idx])
        med = float(np.median(errors(sim)))
        if np.isfinite(med) and med < best_med:
            best, best_med = sim, med
    inl = errors(best) <= max(outlier_factor * best_med, MIN_OUTLIER_ERR)
    for _ in range(3):
        s, r, t = umeyama(c_est[inl], c_ref[inl])
        cerr = errors((s, r, t))
        new = cerr <= max(outlier_factor * float(np.median(cerr[inl])), MIN_OUTLIER_ERR)
        if (new == inl).all() or new.sum() < 3:
            break
        inl = new
    rerr = []
    for k in keys:
        d = (r @ est[k][0]).T @ ref[k][0]
        rerr.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    rerr = np.array(rerr)
    ferr = np.array([abs(est[k][2] / ref[k][2] - 1) for k in keys])
    return {
        "common": len(keys),
        "only_in_ref": len(set(ref) - set(est)),
        "only_in_est": len(set(est) - set(ref)),
        "rot_err_deg": _stats(rerr),
        "center_err_pct": _stats(100 * cerr),
        "focal_err_pct": round(100 * float(np.median(ferr)), 3),
        "outliers": int((~inl).sum()),
    }


def _stats(x: np.ndarray) -> dict:
    return {
        "median": round(float(np.median(x)), 4),
        "mean": round(float(np.mean(x)), 4),
        "p90": round(float(np.percentile(x, 90)), 4),
        "max": round(float(np.max(x)), 4),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", default="workspace")
    ref = ap.add_mutually_exclusive_group(required=True)
    ref.add_argument("--ref-colmap", type=Path, help="COLMAP sparse model with reference poses")
    ref.add_argument("--ref-scene", help="workspace scene whose 01_sfm is the reference")
    args = ap.parse_args(argv)
    est = workspace_poses(Workspace.open(Path(args.workspace), args.scene))
    if args.ref_colmap:
        reference = colmap_poses(args.ref_colmap)
    else:
        reference = workspace_poses(Workspace.open(Path(args.workspace), args.ref_scene))
    print(json.dumps(compare(est, reference), indent=2))


if __name__ == "__main__":
    main()
