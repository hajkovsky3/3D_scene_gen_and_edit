"""Image metrics shared by phases 02, 11 and 12 (torch; environment `sp-gs`).

PSNR and SSIM follow the usual 3DGS evaluation (images in [0, 1], SSIM with an 11x11
gaussian window, sigma 1.5). LPIPS uses torchmetrics with the AlexNet backbone, as gsplat
and most 3DGS papers report it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def psnr(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """PSNR in dB for [..., H, W, 3] images in [0, 1] (averaged over everything)."""
    import torch

    mse = torch.mean((img.clamp(0, 1) - ref) ** 2)
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def _gaussian_1d(size: int, sigma: float, device, dtype) -> torch.Tensor:
    import torch

    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def ssim(img: torch.Tensor, ref: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Differentiable mean SSIM of [B, H, W, 3] images in [0, 1] ('same' padding, like 3DGS)."""
    import torch.nn.functional as F

    x = img.permute(0, 3, 1, 2)
    y = ref.permute(0, 3, 1, 2)
    g = _gaussian_1d(size, sigma, x.device, x.dtype)
    win = (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()
    pad = size // 2

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, win, padding=pad, groups=3)

    mu_x, mu_y = blur(x), blur(y)
    return _ssim_from_moments(mu_x, mu_y, blur(x * x), blur(y * y), blur(x * y))


def ssim_shifted(
    img: torch.Tensor, ref: torch.Tensor, size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """Same value as `ssim`, written for torch.compile (training loss in 02).

    The separable gaussian blur is a weighted sum of shifted copies (zero padding, exactly
    the 2D convolution), so the compiler fuses the whole SSIM into a few kernels: ~4x faster
    than `ssim` forward + backward on a GTX 1650. Without compiling it is slow.
    """
    import math

    import torch
    import torch.nn.functional as F

    x = img.permute(0, 3, 1, 2)
    y = ref.permute(0, 3, 1, 2)
    w = [math.exp(-((i - (size - 1) / 2) ** 2) / (2 * sigma**2)) for i in range(size)]
    w = [v / sum(w) for v in w]
    pad = size // 2
    s = torch.cat([x, y, x * x, y * y, x * y], dim=1)  # [B, 15, H, W]
    h, wd = s.shape[2:]
    sp = F.pad(s, (pad, pad, 0, 0))
    s = sum(w[i] * sp[..., i : i + wd] for i in range(size))
    sp = F.pad(s, (0, 0, pad, pad))
    s = sum(w[i] * sp[..., i : i + h, :] for i in range(size))
    return _ssim_from_moments(*s.split(3, dim=1))


def _ssim_from_moments(mu_x, mu_y, exx, eyy, exy):
    sxx = exx - mu_x**2
    syy = eyy - mu_y**2
    sxy = exy - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
    return (num / den).mean()


class Lpips:
    """LPIPS (AlexNet). Downloads torchvision AlexNet weights on first use (~233 MB)."""

    def __init__(self, device: str = "cuda"):
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        self.model = self.model.to(device).eval()

    def __call__(self, img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """[B, H, W, 3] images in [0, 1] -> mean LPIPS."""
        import torch

        with torch.no_grad():
            return self.model(
                img.clamp(0, 1).permute(0, 3, 1, 2).contiguous(),
                ref.permute(0, 3, 1, 2).contiguous(),
            )
