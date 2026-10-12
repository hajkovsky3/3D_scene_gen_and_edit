"""Gravity alignment and normalization of an SfM reconstruction (pure numpy).

Spec: phases/01_sfm.md, step 7. The result is a similarity x' = s R (x - c) that maps the
estimated up direction to +Z, the median camera center to the origin and makes the RMS
distance of camera centers from it equal to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FLOOR_MAX_ANGLE_DEG = 25.0
FLOOR_MIN_CAMS_ABOVE = 0.8
FLOOR_INLIER_REL = 0.01
FLOOR_MIN_INLIER_FRAC = 0.05
FLOOR_ITERS = 2000


@dataclass
class Alignment:
    matrix: np.ndarray  # 4x4, sim3_from_raw
    scale: float
    up_method: str  # method actually used (may differ from requested after fallback)
    up_raw: list[float]  # up direction in the raw SfM frame
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "sim3_from_raw": self.matrix.tolist(),
            "scale": self.scale,
            "up_method": self.up_method,
            "up_raw": self.up_raw,
            "notes": self.notes,
        }


def camera_up(world_to_cam_rotations: np.ndarray) -> np.ndarray:
    """Mean of the cameras' -y axes expressed in world coordinates, normalized."""
    ups = -world_to_cam_rotations[:, 1, :]  # row 1 of R_cw is the camera y axis in world
    return _unit(ups.mean(axis=0))


def floor_plane(
    points: np.ndarray, centers: np.ndarray, up0: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, float] | None:
    """RANSAC for the largest plane with normal near up0 that has the cameras above it.

    Returns (normal oriented toward the cameras, inlier fraction) or None.
    """
    if len(points) < 3:
        return None
    ref = np.median(centers, axis=0)
    thr = FLOOR_INLIER_REL * float(np.median(np.linalg.norm(points - ref, axis=1)))
    cos_max = np.cos(np.deg2rad(FLOOR_MAX_ANGLE_DEG))
    rng = np.random.default_rng(seed)

    best_count, best = 0, None
    for _ in range(FLOOR_ITERS):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if n @ up0 < 0:
            n = -n
        if n @ up0 < cos_max:
            continue
        d = -n @ a
        if np.mean(centers @ n + d > 0) < FLOOR_MIN_CAMS_ABOVE:
            continue
        count = int(np.count_nonzero(np.abs(points @ n + d) < thr))
        if count > best_count:
            best_count, best = count, (n, d)

    if best is None or best_count < FLOOR_MIN_INLIER_FRAC * len(points):
        return None
    n, d = best
    inliers = points[np.abs(points @ n + d) < thr]
    # Least-squares refit: normal = smallest singular vector of centered inliers.
    _, _, vt = np.linalg.svd(inliers - inliers.mean(axis=0), full_matrices=False)
    n_fit = vt[2] if vt[2] @ n > 0 else -vt[2]
    return n_fit, best_count / len(points)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation R with R @ a = b (a, b unit vectors)."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-9:  # opposite: rotate 180 deg around any axis orthogonal to a
        axis = _unit(np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def compute_alignment(
    method: str, world_to_cam_rotations: np.ndarray, centers: np.ndarray, points: np.ndarray
) -> Alignment:
    """method: camera_up | floor_ransac | none"""
    notes: list[str] = []
    up0 = camera_up(world_to_cam_rotations)
    used = method
    if method == "none":
        up = np.array([0.0, 0.0, 1.0])
    elif method == "camera_up":
        up = up0
    elif method == "floor_ransac":
        found = floor_plane(points, centers, up0)
        if found is None:
            up, used = up0, "camera_up"
            notes.append("floor_ransac: no floor plane found, fell back to camera_up")
        else:
            up, frac = found
            angle = np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))
            notes.append(f"floor plane: {frac:.1%} of points, {angle:.1f} deg from camera_up")
    else:
        raise ValueError(f"unknown align_up {method!r}")

    rot = np.eye(3) if method == "none" else rotation_between(up, np.array([0.0, 0, 1]))
    c = np.median(centers, axis=0)
    rms = float(np.sqrt(np.mean(np.sum((centers - c) ** 2, axis=1))))
    s = 1.0 / rms if rms > 1e-12 else 1.0
    if rms <= 1e-12:
        notes.append("all cameras at one position, scale left at 1")

    m = np.eye(4)
    m[:3, :3] = s * rot
    m[:3, 3] = -s * rot @ c
    return Alignment(m, s, used, up.tolist(), notes)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)
