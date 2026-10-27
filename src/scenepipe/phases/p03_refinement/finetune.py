"""Phase 03_refinement: short fine-tuning of the gaussians after pruning (`finetune_steps > 0`).

Gaussians around the removed ones adapt to fill the gaps, and shortened needles keep their
limit (projected back after every step). No densification: the set of gaussians stays fixed,
so the report's removed counts and the order of the gaussians still hold. Learning rates are
those of phase 02 at the end of its training (positions with the decayed rate). Only the
training views of 02 are used, the test views stay held out for the before/after metrics.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from scenepipe.core.cameras import Camera
from scenepipe.core.gaussians import Gaussians

SSIM_LAMBDA = 0.2  # same loss as phase 02
MEANS_LR = 1.6e-6  # x scene radius: 02's position rate after its exponential decay
LRS = {"scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}


def finetune(
    g: Gaussians,
    train: list[Camera],
    load: Callable[[Camera], np.ndarray],
    steps: int,
    radius: float,
    sh_degree: int,
    antialiased: bool,
    max_aniso: float = 0.0,
    seed: int = 0,
    report: Callable[[float, str], None] | None = None,
) -> Gaussians:
    """Fine-tune `g` for `steps` steps on the `train` cameras; `load(cam)` gives uint8 [H,W,3]
    at the camera's size. `max_aniso > 0` keeps the longest axis <= max_aniso x middle axis."""
    import torch

    from scenepipe.core.metrics import ssim
    from scenepipe.core.render import from_params, render, to_params

    params = {k: torch.nn.Parameter(v) for k, v in to_params(g.with_sh_degree(3)).items()}
    lrs = {"means": MEANS_LR * radius, **LRS}
    optimizers = [
        torch.optim.Adam([params[name]], lr=lr, eps=1e-15, fused=True) for name, lr in lrs.items()
    ]
    images = [torch.from_numpy(load(c)) for c in train]  # CPU, uint8
    gen = torch.Generator().manual_seed(seed)
    order: list[int] = []
    for step in range(steps):
        if not order:
            order = torch.randperm(len(train), generator=gen).tolist()
        i = order.pop()
        gt = images[i].cuda(non_blocking=True).float()[None] / 255
        rgb, _, _, _ = render(params, [train[i]], sh_degree, antialiased=antialiased)
        l1 = torch.abs(rgb - gt).mean()
        loss = (1 - SSIM_LAMBDA) * l1 + SSIM_LAMBDA * (1 - ssim(rgb, gt))
        loss.backward()
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if max_aniso > 0:
            with torch.no_grad():
                limit_needles(params["scales"], max_aniso)
        if report is not None and step % 50 == 0:
            report(step / steps, f"krok {step + 1}/{steps}, loss {loss.item():.4f}")
    return from_params(params)


def limit_needles(log_scales, max_aniso: float) -> None:
    """In place: longest axis <= max_aniso x middle axis (log scales [N,3], torch)."""
    import torch

    s, idx = torch.sort(log_scales, dim=1)
    limit = s[:, 1] + math.log(max_aniso)
    longest = idx[:, 2:3]
    log_scales.scatter_(1, longest, torch.minimum(s[:, 2], limit)[:, None])
