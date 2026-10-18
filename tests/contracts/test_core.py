import json

import numpy as np
import pytest

from scenepipe.core.cameras import load_cameras
from scenepipe.core.gaussians import PLY_PROPERTIES, Gaussians, read_ply, rgb_to_sh_dc, write_ply
from scenepipe.core.workspace import BUILD_PHASES, EDIT_PHASES, PHASES, Workspace


def test_phase_dirs(tmp_path):
    ws = Workspace.open(tmp_path, "demo")
    assert ws.phase_dir("01_sfm") == tmp_path / "demo" / "01_sfm"
    assert ws.phase_dir("12_editing", "e001") == tmp_path / "demo" / "edits" / "e001" / "12_editing"
    assert len(PHASES) == 14 and len(BUILD_PHASES) == 9 and len(EDIT_PHASES) == 4


def random_gaussians(n: int, sh_degree: int, seed: int = 0) -> Gaussians:
    rng = np.random.default_rng(seed)
    k = (sh_degree + 1) ** 2 - 1

    def r(*shape):
        return rng.normal(size=shape).astype(np.float32)

    return Gaussians(r(n, 3), r(n, 3), r(n, k, 3), r(n), r(n, 3), r(n, 4))


@pytest.mark.parametrize("sh_degree", [0, 1, 3])
def test_ply_roundtrip(tmp_path, sh_degree):
    g = random_gaussians(50, sh_degree)
    path = tmp_path / "g.ply"
    write_ply(path, g)
    back = read_ply(path, sh_degree=sh_degree)
    for name in ("means", "sh_dc", "sh_rest", "opacity", "scales", "quats"):
        assert np.array_equal(getattr(back, name), getattr(g, name)), name
    assert back.sh_degree == sh_degree
    full = read_ply(path)
    assert full.sh_rest.shape == (50, 15, 3)
    assert not full.sh_rest[:, (sh_degree + 1) ** 2 - 1 :].any()


def test_ply_layout(tmp_path):
    g = random_gaussians(4, 3)
    path = tmp_path / "g.ply"
    write_ply(path, g)
    raw = path.read_bytes()
    header, body = raw.split(b"end_header\n")
    props = [ln.split()[2] for ln in header.decode().splitlines() if ln.startswith("property")]
    assert tuple(props) == PLY_PROPERTIES and len(props) == 62
    data = np.frombuffer(body, "<f4").reshape(4, 62)
    # f_rest is channel-major: f_rest_0..14 = red channel of SH coefficients 1..15
    assert np.array_equal(data[:, 9:24], g.sh_rest[:, :, 0])
    assert np.array_equal(data[:, 24:39], g.sh_rest[:, :, 1])
    assert np.array_equal(data[:, 58:62], g.quats)


def test_ply_rejects_other_layout(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nend_header\n"
        + np.zeros(1, "<f4").tobytes()
    )
    with pytest.raises(ValueError):
        read_ply(path)


def test_sh_dc_rgb():
    rgb = np.array([[0.0, 0.5, 1.0]])
    g = Gaussians(
        np.zeros((1, 3), np.float32),
        rgb_to_sh_dc(rgb),
        np.zeros((1, 0, 3), np.float32),
        np.zeros(1, np.float32),
        np.zeros((1, 3), np.float32),
        np.array([[1, 0, 0, 0]], np.float32),
    )
    assert np.allclose(g.rgb(), rgb, atol=1e-6)


def test_load_cameras(tmp_path):
    w2c = np.eye(4)
    w2c[:3, 3] = [1, 2, 3]
    cam = {"frame_id": "f000001", "image": "01_sfm/images_undistorted/f000001.jpg"}
    cam |= {"width": 200, "height": 100, "fx": 150.0, "fy": 151.0, "cx": 100.0, "cy": 50.0}
    data = {
        "convention": "opencv_world_to_camera",
        "cameras": [cam | {"world_to_camera": w2c.tolist()}],
    }
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    (c,) = load_cameras(path)
    assert np.allclose(c.center, [-1, -2, -3])
    s = c.scaled(100, 50)
    assert (s.fx, s.fy, s.cx, s.cy) == (75.0, 75.5, 50.0, 25.0)
