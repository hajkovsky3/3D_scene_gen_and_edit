"""Reference-free novel-view scores (scenepipe.eval.novel_views): camera shift and depth score."""

import numpy as np

from scenepipe.core.cameras import Camera
from scenepipe.eval.novel_views import depth_inconsistency, shifted


def camera(w2c: np.ndarray) -> Camera:
    return Camera("f000001", "x.jpg", 100, 80, 90.0, 90.0, 50.0, 40.0, w2c)


def test_shifted_moves_center_up_and_back():
    # camera at (1, 2, 1.5) looking along world +X (camera z = world x, camera y = world -z)
    r = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
    c = np.array([1.0, 2.0, 1.5])
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = r, -r @ c
    s = shifted(camera(w2c), up=0.3, back=0.5)
    assert np.allclose(s.center, [0.5, 2.0, 1.8])
    assert np.allclose(s.world_to_camera[:3, :3], r)  # same orientation
    assert np.allclose(shifted(camera(w2c), 0, 0).world_to_camera, w2c)


def test_depth_inconsistency():
    rng = np.random.default_rng(0)
    inv = rng.uniform(0.2, 1.0, (40, 50))
    mask = np.ones_like(inv, bool)
    assert abs(depth_inconsistency(inv, 3 * inv + 7, mask)) < 1e-9  # scale and shift invariant
    assert depth_inconsistency(inv, -inv, mask) == 2.0
    noise = rng.uniform(0, 1, inv.shape)
    assert 0.8 < depth_inconsistency(inv, noise, mask) < 1.2
    assert depth_inconsistency(inv, inv, np.zeros_like(mask)) is None
