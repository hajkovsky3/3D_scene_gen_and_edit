"""Phase 03_refinement, implementation `prune`: crops the scene to the box of the SfM points,
removes nearly transparent, unobserved, oversized and isolated gaussians and shortens (or
removes) needle-like ones. Spec: phases/03_refinement.md

NumPy only; no GPU needed unless `eval: true`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scenepipe.core.cameras import Camera
from scenepipe.core.gaussians import Gaussians
from scenepipe.core.phase import register
from scenepipe.phases.p03_refinement.common import CRITERIA, REFINE_END, RefinementBase

MAX_VOXEL_OFFSET = 2**19  # voxels from the median; 3 axes x 20 bits fit in an int64 code


@register
class RefinementPrune(RefinementBase):
    impl = "prune"
    impl_version = "2"

    def refine_steps(self) -> list[str]:
        return [*(["viditeľnosť"] if int(self.config["min_views"]) > 0 else []), "orezanie"]

    def refine(
        self, g: Gaussians, train: list[Camera], radius: float
    ) -> tuple[Gaussians | None, dict[str, int], int, dict[str, Any]]:
        cfg = self.config
        min_opacity, min_views = float(cfg["min_opacity"]), int(cfg["min_views"])
        max_scale_frac, max_aniso = float(cfg["max_scale_frac"]), float(cfg["max_aniso"])
        if cfg["aniso_action"] not in ("clamp", "remove"):
            raise ValueError(f"unknown aniso_action {cfg['aniso_action']!r}")
        end = REFINE_END if cfg["eval"] else 1.0
        n = len(g)
        none = np.zeros(n, bool)
        extra: dict[str, Any] = {}
        outside = none
        if cfg["crop"]:
            with np.load(self.ws.abs("01_sfm/points.npz")) as pts:
                box = crop_box(pts["xyz"], float(cfg["crop_quantile"]), float(cfg["crop_margin"]))
            outside = ~inside_box(g.means, box)
            extra["crop_box"] = {k: np.round(v, 6).tolist() for k, v in box.items()}

        opacity = 1 / (1 + np.exp(-g.opacity.astype(np.float64)))
        scales = np.exp(g.scales.astype(np.float64))
        needle = needle_mask(scales, max_aniso) if max_aniso > 0 else none
        if min_views > 0:
            stage = self.progress.stage(0.0, end * 0.9, "viditeľnosť")
            extent = 3 * scales.max(1)
            views = view_counts(g.means, extent, train, lambda f, m: stage.update(f, m))
            unseen = views < min_views
        else:
            unseen = none
        masks = {
            "crop": outside,
            "opacity": opacity < min_opacity if min_opacity > 0 else none,
            "views": unseen,
            "scale": scales.max(1) > max_scale_frac * radius if max_scale_frac > 0 else none,
            "aniso": needle if cfg["aniso_action"] == "remove" else none,
        }
        masks["isolated"] = none.copy()
        if min_count := int(cfg["isolated_min_count"]):
            # among the survivors only: a cluster of haze or of cropped gaussians keeps nothing alive
            alive = ~(masks["crop"] | masks["opacity"] | masks["views"] | masks["scale"])
            voxel = float(cfg["isolated_voxel"]) * radius
            masks["isolated"][alive] = isolated_mask(g.means[alive], voxel, min_count)
        removed, gone = {}, np.zeros(n, bool)
        for key in CRITERIA:  # a gaussian counts for the first criterion that removes it
            removed[key] = int((masks[key] & ~gone).sum())
            gone |= masks[key]
        keep = ~gone

        log_scales = g.scales.copy()
        clamp = needle & keep if cfg["aniso_action"] == "clamp" else none
        if clamp.any():
            log_scales[clamp] = clamp_needles(g.scales[clamp], max_aniso)
        self.progress.update(end, "orezanie", f"ponechaných {int(keep.sum())} z {n}")
        out = Gaussians(
            means=g.means[keep],
            sh_dc=g.sh_dc[keep],
            sh_rest=g.sh_rest[keep],
            opacity=g.opacity[keep],
            scales=log_scales[keep],
            quats=g.quats[keep],
        )
        return out, removed, int(clamp.sum()), extra


def crop_box(points: np.ndarray, quantile: float, margin: float) -> dict[str, np.ndarray]:
    """Box around the scene: horizontal axes from the PCA of the points' XY (+Z stays up,
    so the box follows the walls of a room), extent from the quantile..1-quantile range of
    the points along each axis, enlarged by `margin` x the extent on every side.

    `axes` rows are the box axes in world coordinates; local = (x - center) @ axes.T.
    """
    p = np.asarray(points, np.float64)
    center = np.median(p, axis=0)
    xy = p[:, :2] - center[:2]
    _, _, vt = np.linalg.svd(xy - xy.mean(0), full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[1] *= -1
    axes = np.eye(3)
    axes[:2, :2] = vt
    local = (p - center) @ axes.T
    lo, hi = np.quantile(local, quantile, axis=0), np.quantile(local, 1 - quantile, axis=0)
    pad = margin * (hi - lo)
    return {"center": center, "axes": axes, "lo": lo - pad, "hi": hi + pad}


def inside_box(x: np.ndarray, box: dict[str, np.ndarray]) -> np.ndarray:
    local = (np.asarray(x, np.float64) - box["center"]) @ np.asarray(box["axes"]).T
    return ((local >= box["lo"]) & (local <= box["hi"])).all(axis=1)


def isolated_mask(means: np.ndarray, voxel: float, min_count: int) -> np.ndarray:
    """Gaussians with fewer than `min_count` gaussians (themselves included) in their voxel and
    the 26 neighbouring ones (voxel edge `voxel`, world units)."""
    key = np.floor(np.asarray(means, np.float64) / voxel)
    far = (np.abs(key - np.median(key, axis=0)) > MAX_VOXEL_OFFSET).any(axis=1)
    out = far.copy()  # absurdly far away: isolated, and kept out of the int64 voxel codes
    if far.all():
        return out
    k = key[~far].astype(np.int64)
    k -= k.min(axis=0) - 1  # >= 1, so the -1 neighbour stays non-negative
    dims = k.max(axis=0) + 2
    code = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    cells, inverse, counts = np.unique(code, return_inverse=True, return_counts=True)
    total = np.zeros(len(cells), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nb = cells + (dx * dims[1] + dy) * dims[2] + dz
                idx = np.minimum(np.searchsorted(cells, nb), len(cells) - 1)
                hit = cells[idx] == nb
                total[hit] += counts[idx[hit]]
    out[~far] = total[inverse.ravel()] < min_count
    return out


def view_counts(
    means: np.ndarray, extent: np.ndarray, cams: list[Camera], report=None
) -> np.ndarray:
    """Number of cameras whose image the gaussian reaches: its center projected with a margin
    of its 3-sigma `extent` (world units) in pixels, so large gaussians centered just outside
    the image still count."""
    x = np.asarray(means, np.float32)
    ext = np.asarray(extent, np.float32)
    counts = np.zeros(len(x), np.int32)
    for i, c in enumerate(cams):
        if report is not None:
            report(i / max(len(cams), 1), f"kamera {i + 1}/{len(cams)}")
        r = c.world_to_camera[:3, :3].astype(np.float32)
        t = c.world_to_camera[:3, 3].astype(np.float32)
        p = x @ r.T + t
        z = p[:, 2]
        front = z + ext > 1e-6  # some part of the gaussian is in front of the camera
        zs = np.maximum(z, 1e-6)
        u = c.fx * p[:, 0] / zs + c.cx
        v = c.fy * p[:, 1] / zs + c.cy
        m = np.where(z > ext, max(c.fx, c.fy) * ext / zs, np.inf)  # camera inside: any pixel
        inside = (u >= -m) & (u < c.width + m) & (v >= -m) & (v < c.height + m)
        counts += front & inside
    return counts


def needle_mask(scales: np.ndarray, max_aniso: float) -> np.ndarray:
    """Longest axis longer than max_aniso x the middle axis (linear scales [N,3])."""
    s = np.sort(scales, axis=1)
    return s[:, 2] > max_aniso * s[:, 1]


def clamp_needles(log_scales: np.ndarray, max_aniso: float) -> np.ndarray:
    """Shorten the longest axis to max_aniso x the middle axis (log scales [N,3])."""
    out = log_scales.copy()
    rows = np.arange(len(out))
    longest = out.argmax(1)
    middle = np.sort(out, axis=1)[:, 1]
    out[rows, longest] = middle + np.float32(np.log(max_aniso))
    return out
