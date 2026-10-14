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


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    import torch

    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(img: torch.Tensor, ref: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Differentiable mean SSIM of [B, H, W, 3] images in [0, 1] ('same' padding, like 3DGS)."""
    import torch.nn.functional as F

    x = img.permute(0, 3, 1, 2)
    y = ref.permute(0, 3, 1, 2)
    win = _gaussian_window(size, sigma, x.device, x.dtype)
    pad = size // 2

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, win, padding=pad, groups=3)

    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x**2
    syy = blur(y * y) - mu_y**2
    sxy = blur(x * y) - mu_x * mu_y
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
