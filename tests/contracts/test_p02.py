"""Contract tests for phase 02_reconstruction on a synthetic room (needs CUDA + gsplat)."""

import json

import numpy as np
import pytest
from PIL import Image
from synthetic_room import build_room, orbit_cameras, render, sample_points

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")
if not torch.cuda.is_available():
    pytest.skip("02_reconstruction needs a CUDA GPU", allow_module_level=True)

from scenepipe.core.gaussians import PLY_PROPERTIES, read_ply
from scenepipe.core.workspace import Workspace
from scenepipe.phases.p02_reconstruction.gsplat import (
    ReconstructionGsplat,
    split_views,
)

W, H, F = 160, 120, 115.0
N_VIEWS = 24


def make_sfm(ws: Workspace) -> None:
    """Write a 01_sfm output (cameras.json, images_undistorted, points.npz) with GT poses."""
    out = ws.root / "01_sfm"
    (out / "images_undistorted").mkdir(parents=True)
    rects = build_room()
    cams = []
    for i, w2c in enumerate(orbit_cameras(N_VIEWS), start=1):
        fid = f"f{i:06d}"
        img = render(rects, w2c, W, H, F)
        Image.fromarray(img).save(out / "images_undistorted" / f"{fid}.jpg", quality=95)
        cams.append(
            {
                "frame_id": fid,
                "image": f"01_sfm/images_undistorted/{fid}.jpg",
                "width": W,
                "height": H,
                "fx": F,
                "fy": F,
                "cx": W / 2,
                "cy": H / 2,
                "world_to_camera": w2c.tolist(),
            }
        )
    data = {
        "contract_version": 1,
        "convention": "opencv_world_to_camera",
        "up_axis": "+z",
        "cameras": cams,
    }
    (out / "cameras.json").write_text(json.dumps(data), encoding="utf-8")
    xyz, rgb = sample_points(rects, 5000)
    np.savez_compressed(
        out / "points.npz",
        xyz=xyz,
        rgb=rgb,
        error=np.zeros(len(xyz), np.float32),
        track_len=np.full(len(xyz), 3, np.int32),
    )


@pytest.fixture(scope="module")
def mcmc_run(tmp_path_factory):
    ws = Workspace.open(tmp_path_factory.mktemp("p02"), "room")
    make_sfm(ws)
    cfg = {"iterations": 600, "cap_max": 20000, "export_depth": True, "num_test_renders": 2}
    ReconstructionGsplat(ws, cfg).execute()
    return ws


def load(ws: Workspace, name: str) -> dict:
    return json.loads((ws.root / "02_reconstruction" / name).read_text(encoding="utf-8"))


def test_outputs_format(mcmc_run):
    ws = mcmc_run
    out = ws.root / "02_reconstruction"

    header = (out / "gaussians.ply").read_bytes().split(b"end_header\n")[0].decode()
    props = [ln.split()[2] for ln in header.splitlines() if ln.startswith("property float")]
    assert tuple(props) == PLY_PROPERTIES
    g = read_ply(out / "gaussians.ply")
    assert 0 < len(g) <= 20000

    report = load(ws, "report.json")
    assert report["num_gaussians"] == len(g) and report["sh_degree"] == 3
    assert report["iterations"] == 600 and report["strategy"] == "mcmc"
    assert report["test_frame_ids"] == ["f000001", "f000009", "f000017"]
    assert report["num_train_views"] == N_VIEWS - 3
    assert len(report["per_view"]) == 3
    for key in ("psnr", "ssim", "lpips"):
        assert isinstance(report[key], float)
    assert len(list((out / "test_renders").glob("*.jpg"))) == 2
    with Image.open(out / "test_renders" / "f000001.jpg") as im:
        assert im.size == (2 * W, H)  # render | reference

    depth_files = sorted((out / "depth").glob("*.npy"))
    assert len(depth_files) == N_VIEWS - 3
    d = np.load(depth_files[0])
    assert d.dtype == np.float16 and d.shape == (H, W)
    assert np.isfinite(d).all() and np.median(d) > 0

    manifest = load(ws, "manifest.json")
    assert manifest["phase"] == "02_reconstruction" and manifest["impl"] == "gsplat"
    assert manifest["inputs"]["points"]["path"] == "01_sfm/points.npz"
    assert manifest["outputs"]["gaussians"]["path"] == "02_reconstruction/gaussians.ply"


def test_quality_sanity(mcmc_run):
    # not a quality benchmark: catches broken camera / SH / quaternion conventions
    report = load(mcmc_run, "report.json")
    assert report["psnr"] > 20.0


def test_default_strategy_low_res(tmp_path):
    ws = Workspace.open(tmp_path, "room")
    make_sfm(ws)
    cfg = {"iterations": 300, "strategy": "default", "sh_degree": 1, "resolution_scale": 0.5}
    cfg |= {"lpips": False, "test_every": 0}
    ReconstructionGsplat(ws, cfg).execute()
    report = load(ws, "report.json")
    assert report["test_frame_ids"] == [] and report["psnr"] is None
    assert report["eval_resolution"] == [W // 2, H // 2] and report["warnings"]
    assert read_ply(ws.root / "02_reconstruction" / "gaussians.ply", sh_degree=1).sh_degree == 1


def test_split_views():
    cams = list(range(10))
    train, test = split_views(cams, 4)
    assert test == [0, 4, 8] and train == [1, 2, 3, 5, 6, 7, 9]
    assert split_views(cams, 0) == (cams, [])
