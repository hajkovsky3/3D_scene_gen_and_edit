"""Training-speed helpers of phase 02: shifted SSIM equals the convolution SSIM; schedules."""

import pytest

torch = pytest.importorskip("torch")  # environment sp-gs

from scenepipe.core.cameras import Camera
from scenepipe.core.metrics import ssim, ssim_shifted


def test_ssim_shifted_matches_ssim():
    g = torch.Generator().manual_seed(0)
    ref = torch.rand(1, 37, 53, 3, generator=g)
    img = (ref + 0.1 * torch.randn(ref.shape, generator=g)).clamp(0, 1).requires_grad_(True)
    a = ssim(img, ref)
    (ga,) = torch.autograd.grad(a, img)
    b = ssim_shifted(img, ref)
    (gb,) = torch.autograd.grad(b, img)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(ga, gb, atol=1e-7)


def test_resolution_schedule_and_downscale():
    pytest.importorskip("gsplat")
    import numpy as np

    from scenepipe.phases.p02_reconstruction.gsplat import _downscaled, _resolution_schedule

    s = _resolution_schedule([[0.5, 0.5], [0.25, 0.25]], 100)
    assert [s(i) for i in (0, 24, 25, 49, 50, 99)] == [0.25, 0.25, 0.5, 0.5, 1.0, 1.0]
    assert _resolution_schedule([], 100)(0) == 1.0

    cam = Camera("f1", "x.jpg", 80, 60, 100.0, 100.0, 40.0, 30.0, np.eye(4))
    small, gt = _downscaled(cam, torch.ones(1, 60, 80, 3), 0.25)
    assert (small.width, small.height) == (20, 15) and gt.shape == (1, 15, 20, 3)
    assert small.fx == pytest.approx(25.0) and small.cx == pytest.approx(10.0)
    assert torch.allclose(gt, torch.ones_like(gt))
