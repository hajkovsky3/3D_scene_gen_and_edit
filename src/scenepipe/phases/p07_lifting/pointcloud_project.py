"""Phase 07_lifting, baseline `pointcloud_project`: the original design. Gaussian centers are
projected into each frame; a center counts as visible if it lies in the image, in front of
the camera and not behind the rendered surface (|z - depth| <= depth_tol x depth), and as
inside a mask if its pixel is in the mask. Weights are 0/1, so the extent and opacity of a
gaussian are ignored. Merging, voting and cleanup are shared with gsplat_vote (common.py),
so the comparison isolates the weights. Spec: phases/07_lifting.md

The depth comes from rendering the 03 gaussians with gsplat (02 does not export depth by
default, and its depth would be of the unpruned gaussians).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np

from scenepipe.core.cameras import Camera
from scenepipe.core.gaussians import Gaussians
from scenepipe.core.phase import register
from scenepipe.core.progress import StageProgress
from scenepipe.phases.p07_lifting.common import FrameMasks, FrameWeights, LiftingBase

log = logging.getLogger(__name__)


@register
class LiftingPointcloudProject(LiftingBase):
    impl = "pointcloud_project"
    impl_version = "1"
    defaults: ClassVar[dict[str, Any]] = {
        **LiftingBase.defaults,
        "depth_tol": 0.05,  # relative depth tolerance of the occlusion test
        "render_scale": 1.0,  # resolution of the depth render
        "device": "cuda",
    }

    def frame_weights(
        self, g: Gaussians, frames: list[FrameMasks], stage: StageProgress
    ) -> Iterator[FrameWeights]:
        for i, fm in enumerate(frames):
            px, py, ok = project_visible(
                g.means, fm.camera, self._depth(g, fm.camera), float(self.config["depth_tol"])
            )
            vis_idx = np.flatnonzero(ok)
            u, v = px[vis_idx], py[vis_idx]
            mask_idx, mask_w = [], []
            for m in fm.masks:
                inside = vis_idx[m[v, u]]
                mask_idx.append(inside)
                mask_w.append(np.ones(len(inside), np.float32))
            stage.update((i + 1) / len(frames), f"{i + 1}/{len(frames)} snímok")
            yield FrameWeights(vis_idx, np.ones(len(vis_idx), np.float32), mask_idx, mask_w)

    def _depth(self, g: Gaussians, cam: Camera) -> np.ndarray:
        """Expected z-depth at the camera resolution (inf where nothing is rendered)."""
        import torch

        from scenepipe.core.render import render, to_params

        if getattr(self, "_params", None) is None:
            self._params = to_params(g, self.config["device"])
        s = float(self.config["render_scale"])
        small = cam.scaled(round(cam.width * s), round(cam.height * s)) if s != 1.0 else cam
        with torch.no_grad():
            _, z, alpha, _ = render(self._params, [small], 0, depth=True)
        z = z[0].cpu().numpy()  # "ED": already normalized by alpha
        z[alpha[0].cpu().numpy() < 0.5] = np.inf
        if s != 1.0:
            ys = np.minimum((np.arange(cam.height) * s).astype(int), small.height - 1)
            xs = np.minimum((np.arange(cam.width) * s).astype(int), small.width - 1)
            z = z[ys[:, None], xs[None, :]]
        return z

    def render_instances(self, g: Gaussians, colors: np.ndarray, cam: Camera) -> np.ndarray:
        import torch

        from scenepipe.core.gaussians import rgb_to_sh_dc
        from scenepipe.core.render import render, to_params

        flat = Gaussians(
            g.means, rgb_to_sh_dc(colors.astype(np.float32) / 255),
            np.zeros((len(g), 0, 3), np.float32), g.opacity, g.scales, g.quats,
        )  # fmt: skip
        with torch.no_grad():
            rgb, _, _, _ = render(to_params(flat, self.config["device"]), [cam], 0)
        return (rgb[0].clamp(0, 1) * 255).byte().cpu().numpy()


def project_visible(
    means: np.ndarray, cam: Camera, depth: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates (int) of the gaussian centers and whether each is visible."""
    w2c = cam.world_to_camera
    p = means.astype(np.float64) @ w2c[:3, :3].T + w2c[:3, 3]
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(cam.fx * p[:, 0] / z + cam.cx).astype(np.int64, copy=False)
        v = np.floor(cam.fy * p[:, 1] / z + cam.cy).astype(np.int64, copy=False)
    ok = (z > 1e-6) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    u = np.where(ok, u, 0)
    v = np.where(ok, v, 0)
    ok &= z <= depth[v, u] * (1 + tol)
    return u, v, ok
