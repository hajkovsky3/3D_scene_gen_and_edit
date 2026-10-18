"""Tiny ray-casting renderer of a textured box room with two boxes, for SfM and reconstruction tests.

World: +Z up, floor at z=0. Cameras use the OpenCV convention (x right, y down, z forward)
and the COLMAP pixel convention (pixel centers at +0.5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

PX_PER_M = 150


@dataclass
class Rect:
    origin: np.ndarray
    u: np.ndarray  # unit axis, extent su
    v: np.ndarray  # unit axis, extent sv
    su: float
    sv: float
    texture: np.ndarray

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.u, self.v)


def _texture(rng: np.random.Generator, su: float, sv: float) -> np.ndarray:
    h, w = max(8, int(sv * PX_PER_M)), max(8, int(su * PX_PER_M))
    tex = np.zeros((h, w, 3), np.float32)
    for sigma, amp in ((1.2, 0.6), (4, 1.0), (14, 1.4)):
        noise = rng.normal(size=(h, w, 3)).astype(np.float32)
        tex += amp * cv2.GaussianBlur(noise, (0, 0), sigma) * sigma
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)
    return np.clip(128 + 45 * tex, 0, 255).astype(np.float32)


def _box_faces(rng, x0, y0, z0, sx, sy, sz) -> list[Rect]:
    o = np.array([x0, y0, z0], float)
    ex, ey, ez = np.eye(3)
    faces = [
        (o + [0, 0, sz], ex, ey, sx, sy),  # top
        (o, ex, ez, sx, sz),  # front (y = y0)
        (o + [sx, sy, 0], -ex, ez, sx, sz),  # back
        (o + [0, sy, 0], -ey, ez, sy, sz),  # left (x = x0)
        (o + [sx, 0, 0], ey, ez, sy, sz),  # right
    ]
    return [
        Rect(np.asarray(f[0]), f[1], f[2], f[3], f[4], _texture(rng, f[3], f[4])) for f in faces
    ]


def build_room(seed: int = 0) -> list[Rect]:
    rng = np.random.default_rng(seed)
    lx, ly, lz = 6.0, 5.0, 2.6
    o = np.array([-lx / 2, -ly / 2, 0.0])
    ex, ey, ez = np.eye(3)
    walls = [
        (o, ex, ey, lx, ly),  # floor
        (o, ex, ez, lx, lz),  # wall y = -ly/2
        (o + [lx, ly, 0], -ex, ez, lx, lz),  # wall y = +ly/2
        (o + [0, ly, 0], -ey, ez, ly, lz),  # wall x = -lx/2
        (o + [lx, 0, 0], ey, ez, ly, lz),  # wall x = +lx/2
    ]
    rects = [
        Rect(np.asarray(w[0]), w[1], w[2], w[3], w[4], _texture(rng, w[3], w[4])) for w in walls
    ]
    rects += _box_faces(rng, -0.6, -0.4, 0.0, 1.0, 0.8, 0.75)  # table
    rects += _box_faces(rng, 0.7, 0.3, 0.0, 0.5, 0.5, 0.45)  # stool
    return rects


def look_at(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """4x4 world_to_camera, OpenCV convention, world +Z up."""
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = -r @ center
    return m


def render(rects: list[Rect], w2c: np.ndarray, width: int, height: int, f: float) -> np.ndarray:
    r, t = w2c[:3, :3], w2c[:3, 3]
    center = -r.T @ t
    us, vs = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    d_cam = np.stack([(us - width / 2) / f, (vs - height / 2) / f, np.ones_like(us)], -1)
    d = d_cam @ r  # = (R^T d_cam^T)^T, ray directions in world
    depth = np.full((height, width), np.inf)
    img = np.zeros((height, width, 3), np.float32)
    for rect in rects:
        n = rect.normal
        denom = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = ((rect.origin - center) @ n) / denom
        p = center + tt[..., None] * d
        a = (p - rect.origin) @ rect.u
        b = (p - rect.origin) @ rect.v
        hit = (tt > 1e-6) & (tt < depth) & (a >= 0) & (a <= rect.su) & (b >= 0) & (b <= rect.sv)
        if not hit.any():
            continue
        mx = np.where(hit, a * PX_PER_M, 0).astype(np.float32)
        my = np.where(hit, b * PX_PER_M, 0).astype(np.float32)
        col = cv2.remap(rect.texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img[hit] = col[hit]
        depth[hit] = tt[hit]
    return np.clip(img, 0, 255).astype(np.uint8)


def sample_points(rects: list[Rect], n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Points spread over all surfaces (area-weighted) with their texture colors."""
    rng = np.random.default_rng(seed)
    areas = np.array([r.su * r.sv for r in rects])
    counts = rng.multinomial(n, areas / areas.sum())
    xyz, rgb = [], []
    for rect, k in zip(rects, counts, strict=True):
        a, b = rng.uniform(0, rect.su, k), rng.uniform(0, rect.sv, k)
        xyz.append(rect.origin + a[:, None] * rect.u + b[:, None] * rect.v)
        h, w = rect.texture.shape[:2]
        iy = np.clip((b * PX_PER_M).astype(int), 0, h - 1)
        ix = np.clip((a * PX_PER_M).astype(int), 0, w - 1)
        rgb.append(rect.texture[iy, ix])
    return np.concatenate(xyz).astype(np.float32), np.concatenate(rgb).astype(np.uint8)


def orbit_cameras(n: int, radius: float = 1.7, height: float = 1.4) -> list[np.ndarray]:
    """Cameras on a circle looking at the room center slightly above the floor."""
    cams = []
    for k in range(n):
        a = 2 * np.pi * k / n
        c = np.array([radius * np.cos(a), radius * np.sin(a), height])
        cams.append(look_at(c, np.array([0.0, 0.0, 0.4])))
    return cams


def make_sfm(ws, n_views: int = 24, w: int = 160, h: int = 120, f: float = 115.0) -> None:
    """Write a 01_sfm output (cameras.json, images_undistorted, points.npz) with GT poses."""
    out = ws.root / "01_sfm"
    (out / "images_undistorted").mkdir(parents=True)
    rects = build_room()
    cams = []
    for i, w2c in enumerate(orbit_cameras(n_views), start=1):
        fid = f"f{i:06d}"
        img = render(rects, w2c, w, h, f)
        Image.fromarray(img).save(out / "images_undistorted" / f"{fid}.jpg", quality=95)
        cams.append(
            {
                "frame_id": fid,
                "image": f"01_sfm/images_undistorted/{fid}.jpg",
                "width": w,
                "height": h,
                "fx": f,
                "fy": f,
                "cx": w / 2,
                "cy": h / 2,
                "world_to_camera": w2c.tolist(),
            }
        )
    data = {
        "contract_version": 1,
        "convention": "opencv_world_to_camera",
        "up_axis": "+z",
        "cameras": cams,
    }
    (out / "cameras.json").write_text(json.dumps(data), encoding="utf-8")
    xyz, rgb = sample_points(rects, 5000)
    np.savez_compressed(
        out / "points.npz",
        xyz=xyz,
        rgb=rgb,
        error=np.zeros(len(xyz), np.float32),
        track_len=np.full(len(xyz), 3, np.int32),
    )
