"""Rendering gaussians with gsplat (Apache 2.0). Shared by phases 02, 07, 12 and 13.

Requires torch + gsplat (environment `sp-gs`); imported lazily so `scenepipe.core` stays
importable in environments without them. Parameters use the stored (pre-activation)
parametrization of docs/data-contracts.md, section 5, in the gsplat naming:

    means [N,3], scales [N,3] (log), quats [N,4] (wxyz), opacities [N] (logit),
    sh0 [N,1,3], shN [N,K,3]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scenepipe.core.cameras import Camera
from scenepipe.core.gaussians import Gaussians

if TYPE_CHECKING:
    import torch


def to_params(g: Gaussians, device: str = "cuda") -> dict[str, torch.Tensor]:
    import torch

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)

    return {
        "means": t(g.means),
        "scales": t(g.scales),
        "quats": t(g.quats),
        "opacities": t(g.opacity),
        "sh0": t(g.sh_dc[:, None, :]),
        "shN": t(g.sh_rest),
    }


def from_params(params: dict[str, torch.Tensor]) -> Gaussians:
    def a(name: str) -> np.ndarray:
        return params[name].detach().float().cpu().numpy()

    return Gaussians(
        means=a("means"),
        sh_dc=a("sh0")[:, 0, :],
        sh_rest=a("shN"),
        opacity=a("opacities"),
        scales=a("scales"),
        quats=a("quats"),
    )


def render(
    params: dict[str, torch.Tensor],
    cameras: list[Camera],
    sh_degree: int,
    *,
    antialiased: bool = True,
    background: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    depth: bool = False,
    absgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict]:
    """Render a batch of same-size cameras.

    Returns (rgb [C,H,W,3] in [0,1] (not clamped), expected z-depth [C,H,W] or None,
    alpha [C,H,W], gsplat meta dict needed by densification strategies).
    """
    import torch
    from gsplat import rasterization

    w, h = cameras[0].width, cameras[0].height
    if any((c.width, c.height) != (w, h) for c in cameras):
        raise ValueError("render(): all cameras in a batch must have the same size")
    dev = params["means"].device
    viewmats = torch.tensor(np.stack([c.world_to_camera for c in cameras]), dtype=torch.float32)
    ks = torch.tensor(np.stack([c.K for c in cameras]), dtype=torch.float32)
    bg = None
    if background is not None:
        # RGB only; gsplat pads the depth channel itself
        bg = torch.tensor(background, dtype=torch.float32, device=dev).expand(len(cameras), 3)
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    k = (sh_degree + 1) ** 2
    out, alpha, meta = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors[:, :k],
        viewmats=viewmats.to(dev),
        Ks=ks.to(dev),
        width=w,
        height=h,
        sh_degree=sh_degree,
        backgrounds=bg,
        render_mode="RGB+ED" if depth else "RGB",
        rasterize_mode="antialiased" if antialiased else "classic",
        absgrad=absgrad,
        packed=False,
    )
    rgb = out[..., :3]
    z = out[..., 3] if depth else None
    return rgb, z, alpha[..., 0], meta
