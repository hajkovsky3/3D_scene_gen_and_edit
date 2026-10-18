"""Training helpers of phase 02: shifted SSIM equals the convolution SSIM; schedules; anisotropy."""

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


def test_aniso_loss_penalizes_needles_not_disks():
    from scenepipe.phases.p02_reconstruction.gsplat import aniso_loss

    s = torch.tensor([[0.5, 0.01, 0.01], [0.5, 0.5, 0.001], [0.1, 0.1, 0.1]])  # needle, disk, ball
    log_s = torch.log(s).requires_grad_(True)
    loss = aniso_loss(log_s, 10.0)
    assert torch.isclose(loss, torch.tensor((50.0 - 10.0) / 3))
    (grad,) = torch.autograd.grad(loss, log_s)
    assert grad[0, 0] > 0 and grad[0, 1:].sum() < 0  # shorten the long axis, widen the others
    assert torch.all(grad[1:] == 0)


def test_depth_corr_loss_is_scale_shift_invariant():
    from scenepipe.phases.p02_reconstruction.gsplat import depth_corr_loss

    g = torch.Generator().manual_seed(0)
    z = 1 + 4 * torch.rand(20, 30, generator=g)
    alpha = torch.ones(20, 30)
    mono = 3.0 / z + 7.0  # inverse depth up to an unknown scale and shift
    assert depth_corr_loss(z, alpha, mono).abs() < 1e-5
    assert torch.isclose(depth_corr_loss(z, alpha, -mono), torch.tensor(2.0), atol=1e-5)
    alpha[:, 15:] = 0  # uncovered pixels are ignored
    mono[:, 15:] = 0
    assert depth_corr_loss(z, alpha, mono).abs() < 1e-5
    assert depth_corr_loss(z, torch.zeros_like(alpha), mono) == 0
