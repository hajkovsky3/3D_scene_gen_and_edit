"""Phase 07_lifting, implementation `gsplat_vote`: W_i(m) from gsplat (Apache 2.0) by the
gradient trick of phases/07_lifting.md: render a zero feature f[N, K] whose channels are
masks, take L = sum_p sum_k M_k(p) F_k(p); then dL/df[i, k] = sum_{p in m_k} alpha_i(p) T_i(p)
= W_i(m_k) exactly. One extra all-ones channel gives the visibility of each gaussian in the
frame. gsplat culls gaussians outside the view itself, so no culling is needed here.

The rasterize mode (antialiased / classic) follows the config of 02, so the weights are those
of the images the gaussians were trained on.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any, ClassVar

import cv2
import numpy as np

from scenepipe.core.cameras import Camera
from scenepipe.core.gaussians import Gaussians
from scenepipe.core.phase import register
from scenepipe.core.progress import StageProgress
from scenepipe.phases.p07_lifting.common import FrameMasks, FrameWeights, LiftingBase

log = logging.getLogger(__name__)


@register
class LiftingGsplatVote(LiftingBase):
    impl = "gsplat_vote"
    impl_version = "1"
    defaults: ClassVar[dict[str, Any]] = {
        **LiftingBase.defaults,
        "mask_batch": 32,  # mask channels per render (+1 visibility channel)
        "render_scale": 1.0,  # render (and downsample masks) at this share of the resolution
        "device": "cuda",
    }

    def frame_weights(
        self, g: Gaussians, frames: list[FrameMasks], stage: StageProgress
    ) -> Iterator[FrameWeights]:
        import torch

        params = self._params(g)
        dev = self.config["device"]
        batch = max(int(self.config["mask_batch"]), 1)
        scale = float(self.config["render_scale"])
        for i, fm in enumerate(frames):
            cam = fm.camera
            if scale != 1.0:
                cam = cam.scaled(round(cam.width * scale), round(cam.height * scale))
            masks = [_resize_mask(m, cam.width, cam.height) for m in fm.masks]
            chans = [np.ones((cam.height, cam.width), np.float32), *masks]  # 0 = visibility
            idx_out, w_out = [], []
            # first batch carries the visibility channel, the rest only masks
            starts = [0, *range(batch + 1, len(chans), batch)]
            for s0 in starts:
                s1 = s0 + batch + 1 if s0 == 0 else s0 + batch
                m = torch.from_numpy(np.stack(chans[s0:s1], axis=-1)).to(dev)
                grad = self._weights(params, cam, m)
                for k in range(grad.shape[1]):
                    nz = torch.nonzero(grad[:, k] > 0)[:, 0]
                    idx_out.append(nz.cpu().numpy())
                    w_out.append(grad[nz, k].float().cpu().numpy())
            stage.update((i + 1) / len(frames), f"{i + 1}/{len(frames)} snímok")
            yield FrameWeights(idx_out[0], w_out[0], idx_out[1:], w_out[1:])
        if dev == "cuda":
            log.info("peak VRAM %.0f MB", torch.cuda.max_memory_allocated() / 2**20)

    def render_instances(self, g: Gaussians, colors: np.ndarray, cam: Camera) -> np.ndarray:
        import torch

        params = self._params(g)
        feat = torch.from_numpy(colors.astype(np.float32) / 255).to(self.config["device"])
        with torch.no_grad():
            img = self._rasterize(params, cam, feat)
        return (img.clamp(0, 1) * 255).byte().cpu().numpy()

    # --- gsplat ----------------------------------------------------------------------------

    def _params(self, g: Gaussians) -> dict:
        import torch

        if getattr(self, "_cache", None) is None:
            dev = self.config["device"]

            def t(a: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(dev)

            self._cache = {
                "means": t(g.means),
                "quats": t(g.quats),
                "scales": torch.exp(t(g.scales)),
                "opacities": torch.sigmoid(t(g.opacity)),
            }
            self._antialiased = self._antialiased_02()
        return self._cache

    def _weights(self, params: dict, cam: Camera, masks) -> Any:
        """[N, C] weights of every gaussian in each of the C mask channels."""
        import torch

        n = params["means"].shape[0]
        f = torch.zeros((n, masks.shape[-1]), device=masks.device, requires_grad=True)
        img = self._rasterize(params, cam, f)
        (img * masks).sum().backward()
        return f.grad

    def _rasterize(self, params: dict, cam: Camera, feat) -> Any:
        """[H, W, C] render of per-gaussian features (no SH, black background)."""
        import torch
        from gsplat import rasterization

        dev = params["means"].device
        out, _, _ = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"],
            colors=feat,
            viewmats=torch.tensor(cam.world_to_camera, dtype=torch.float32, device=dev)[None],
            Ks=torch.tensor(cam.K, dtype=torch.float32, device=dev)[None],
            width=cam.width,
            height=cam.height,
            sh_degree=None,
            rasterize_mode="antialiased" if self._antialiased else "classic",
            packed=False,
        )
        return out[0]

    def _antialiased_02(self) -> bool:
        path = self.ws.abs("02_reconstruction/manifest.json")
        if not path.is_file():
            return True
        manifest = json.loads(path.read_text("utf-8"))
        return bool(manifest.get("config", {}).get("antialiased", True))


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """bool mask -> float32 at the render size (area-averaged, so soft at the edges)."""
    m = mask.astype(np.float32)
    if m.shape != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_AREA)
    return m
