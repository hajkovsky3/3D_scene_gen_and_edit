"""Contract tests for phase 03_refinement on a synthetic room.

The 02 output is built by hand (no training): gaussians on the room surfaces plus injected
artifacts of every kind. Pruning runs on the CPU; the before/after evaluation needs CUDA.
"""

import json
import math

import numpy as np
import pytest
from PIL import Image
from synthetic_room import build_room, make_sfm, sample_points

from scenepipe.core.gaussians import PLY_PROPERTIES, Gaussians, read_ply, rgb_to_sh_dc, write_ply
from scenepipe.core.workspace import Workspace
from scenepipe.phases.p03_refinement.none import RefinementNone
from scenepipe.phases.p03_refinement.prune import (
    RefinementPrune,
    clamp_needles,
    crop_box,
    inside_box,
    needle_mask,
)

W, H = 160, 120
N_VIEWS = 24
TEST_IDS = ["f000001", "f000009", "f000017"]
N_SURFACE = 3000
N_TRANSPARENT, N_FLOATERS, N_HUGE, N_NEEDLES = 40, 30, 20, 25
CPU = {"eval": False}


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def make_reconstruction(ws: Workspace) -> None:
    """02_reconstruction output: gaussians.ply, report.json, manifest.json."""
    xyz, rgb = sample_points(build_room(), 4 * N_SURFACE, seed=1)
    low = xyz[:, 2] < 1.0  # the cameras (at 1.4 m, looking down) never see the upper walls
    k = N_SURFACE + N_TRANSPARENT + N_HUGE + N_NEEDLES
    xyz, rgb = xyz[low][:k], rgb[low][:k]
    n = len(xyz) + N_FLOATERS
    rng = np.random.default_rng(0)
    floaters = np.column_stack([rng.uniform(-1, 1, (N_FLOATERS, 2)), np.full(N_FLOATERS, 100.0)])
    means = np.vstack([xyz, floaters]).astype(np.float32)
    rgb = np.vstack([rgb, np.full((N_FLOATERS, 3), 128)]).astype(np.float32) / 255
    opacity = np.full(n, _logit(0.8), np.float32)
    scales = np.full((n, 3), math.log(0.02), np.float32)

    i = N_SURFACE
    opacity[i : i + N_TRANSPARENT] = _logit(0.005)
    i += N_TRANSPARENT
    scales[i : i + N_HUGE] = math.log(10.0)  # > max_scale_frac x scene radius
    i += N_HUGE
    scales[i : i + N_NEEDLES] = np.log([0.5, 0.01, 0.01])  # aniso 50, longest axis still small

    g = Gaussians(
        means=means,
        sh_dc=rgb_to_sh_dc(rgb),
        sh_rest=np.zeros((n, 15, 3), np.float32),
        opacity=opacity,
        scales=scales,
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
    )
    out = ws.root / "02_reconstruction"
    out.mkdir(parents=True)
    write_ply(out / "gaussians.ply", g)
    report = {"sh_degree": 3, "test_frame_ids": TEST_IDS, "eval_resolution": [W, H]}
    (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
    manifest = {"phase": "02_reconstruction", "config": {"antialiased": True}}
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    ws = Workspace.open(tmp_path_factory.mktemp("p03"), "room")
    make_sfm(ws, N_VIEWS, W, H)
    make_reconstruction(ws)
    return ws


def run(scene_ws: Workspace, cls, cfg: dict) -> tuple[dict, Gaussians]:
    cls(scene_ws, cfg).execute()
    out = scene_ws.root / "03_refinement"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return report, read_ply(out / "gaussians.ply")


def test_prune_outputs_format(scene):
    cfg = CPU | {"crop": True, "max_scale_frac": 0.5, "max_aniso": 10, "aniso_action": "remove"}
    report, g = run(scene, RefinementPrune, cfg)
    out = scene.root / "03_refinement"

    header = (out / "gaussians.ply").read_bytes().split(b"end_header\n")[0].decode()
    props = [ln.split()[2] for ln in header.splitlines() if ln.startswith("property float")]
    assert tuple(props) == PLY_PROPERTIES

    removed = report["removed"]
    assert removed["opacity"] == N_TRANSPARENT and removed["scale"] == N_HUGE
    assert removed["aniso"] == N_NEEDLES and report["clamped"] == 0
    assert removed["crop"] == N_FLOATERS  # far above the room, outside the SfM points box
    assert removed["views"] <= 0.05 * N_SURFACE
    assert (g.means[:, 2] < 50).all()
    assert report["num_input"] == N_SURFACE + N_TRANSPARENT + N_HUGE + N_NEEDLES + N_FLOATERS
    assert report["num_output"] == len(g) == report["num_input"] - sum(removed.values())
    assert report["num_train_views"] == N_VIEWS - len(TEST_IDS)
    assert report["per_view"] == [] and report["before"]["psnr"] is None
    assert set(report["crop_box"]) == {"center", "axes", "lo", "hi"}

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "03_refinement" and manifest["impl"] == "prune"
    assert manifest["inputs"]["gaussians"]["path"] == "02_reconstruction/gaussians.ply"
    assert manifest["outputs"]["gaussians"]["path"] == "03_refinement/gaussians.ply"


def test_prune_views_without_crop(scene):
    report, g = run(scene, RefinementPrune, CPU | {"crop": False})
    # every floater goes; a few surface points may reach fewer than 3 training cameras
    assert N_FLOATERS <= report["removed"]["views"] <= N_FLOATERS + 0.05 * N_SURFACE
    assert report["removed"]["crop"] == 0 and report["crop_box"] is None
    assert (g.means[:, 2] < 50).all()


def test_prune_clamps_needles(scene):
    report, g = run(scene, RefinementPrune, CPU | {"max_aniso": 10, "min_views": 0})
    assert report["clamped"] == N_NEEDLES and report["removed"]["aniso"] == 0
    assert report["removed"]["views"] == 0
    assert not needle_mask(np.exp(g.scales), 10 * 1.0001).any()


def test_none_copies_input(scene):
    report, _ = run(scene, RefinementNone, CPU)
    out = scene.root / "03_refinement" / "gaussians.ply"
    src = scene.root / "02_reconstruction" / "gaussians.ply"
    assert out.read_bytes() == src.read_bytes()
    assert set(report["removed"].values()) == {0} and report["num_output"] == report["num_input"]


def test_crop_box_follows_rotated_room():
    rng = np.random.default_rng(0)
    local = rng.uniform([-3, -1, 0], [3, 1, 2.5], (2000, 3))
    a = np.deg2rad(30)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    pts = local @ rot.T + [5, -2, 0]
    box = crop_box(np.vstack([pts, [[40, 40, 1]]]), 0.0, 0.0)  # one far point: not robust
    assert inside_box(pts, box).all()
    box = crop_box(np.vstack([pts, [[40, 40, 1]]]), 0.01, 0.1)
    assert inside_box(pts, box).mean() > 0.99 and not inside_box([[40, 40, 1]], box).any()
    corner = rot @ [0, 2, 1] + [5, -2, 0]  # outside the rotated room, inside its AABB
    assert not inside_box(corner[None], box).any()
    assert np.allclose(box["axes"][2], [0, 0, 1])


def test_clamp_needles_keeps_axis_order():
    log_s = np.log(np.array([[0.01, 0.5, 0.02], [0.1, 0.1, 0.1]], np.float32))
    out = np.exp(clamp_needles(log_s, 5.0))
    np.testing.assert_allclose(out[0], [0.01, 0.1, 0.02], rtol=1e-5)
    assert needle_mask(np.exp(log_s), 5.0).tolist() == [True, False]


def test_eval_before_after(scene):
    torch = pytest.importorskip("torch")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("evaluation needs a CUDA GPU")
    report, _ = run(scene, RefinementPrune, {"lpips": False, "num_test_renders": 2})
    assert report["test_frame_ids"] == TEST_IDS and len(report["per_view"]) == 3
    assert report["eval_resolution"] == [W, H] and report["lpips_net"] is None
    for key in ("before", "after"):
        assert isinstance(report[key]["psnr"], float) and report[key]["lpips"] is None
    renders = sorted((scene.root / "03_refinement" / "test_renders").glob("*.jpg"))
    assert len(renders) == 2
    with Image.open(renders[0]) as im:
        assert im.size == (3 * W, H)  # before | after | reference
