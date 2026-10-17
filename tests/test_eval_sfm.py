"""Pose accuracy evaluation (scenepipe.eval.sfm_poses) on synthetic poses."""

import numpy as np

from scenepipe.eval.sfm_poses import compare, umeyama


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    r = random_rotation(rng)
    dst = 2.5 * src @ r.T + [1, 2, 3]
    s, r2, t = umeyama(src, dst)
    assert np.isclose(s, 2.5) and np.allclose(r2, r) and np.allclose(t, [1, 2, 3])


def test_compare_ignores_gauge_and_flags_outlier():
    rng = np.random.default_rng(1)
    ref = {}
    for i in range(30):
        rot = random_rotation(rng)
        ref[("v.mp4", i)] = (rot, rng.normal(size=3), 0.8)
    g = rot_z(40)
    est = {k: (g @ rot, 0.5 * g @ c + 7, f) for k, (rot, c, f) in ref.items()}
    res = compare(est, ref)
    assert res["common"] == 30 and res["outliers"] == 0
    assert res["rot_err_deg"]["max"] < 1e-6 and res["center_err_pct"]["max"] < 1e-6

    rot, c, f = est[("v.mp4", 3)]
    est[("v.mp4", 3)] = (rot, c + 5.0, f)  # one mislocalized camera
    res = compare(est, ref)
    assert res["outliers"] == 1 and res["center_err_pct"]["median"] < 1e-6
