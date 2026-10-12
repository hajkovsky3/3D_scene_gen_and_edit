"""Contract tests for phase 01_sfm on a synthetic room with known cameras."""

import json

import numpy as np
import pycolmap
import pytest
from PIL import Image
from synthetic_room import build_room, orbit_cameras, render

from scenepipe.core.workspace import Workspace
from scenepipe.phases.p01_sfm.align import compute_alignment, floor_plane, rotation_between
from scenepipe.phases.p01_sfm.colmap import SfmColmap

W, H, F = 320, 240, 230.0


def make_ingest(ws: Workspace, gt: list[np.ndarray], source: str | None) -> None:
    """Write a 00_ingest output (frames.json + images) for the given camera poses."""
    images = ws.root / "00_ingest" / "images"
    images.mkdir(parents=True)
    rects = build_room()
    frames = []
    for i, w2c in enumerate(gt, start=1):
        fid = f"f{i:06d}"
        Image.fromarray(render(rects, w2c, W, H, F)).save(images / f"{fid}.jpg", quality=95)
        frames.append(
            {
                "frame_id": fid,
                "image": f"00_ingest/images/{fid}.jpg",
                "width": W,
                "height": H,
                "source": source or f"input/photo_{i:02d}.jpg",
                "source_index": i if source else None,
                "timestamp_s": i / 3 if source else None,
                "sharpness": 1.0,
                "exif_focal_mm": None,
                "exif_focal_35mm": None,
            }
        )
    data = {"contract_version": 1, "frames": frames, "rejected": []}
    (ws.root / "00_ingest" / "frames.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture(scope="module")
def video_run(tmp_path_factory):
    ws = Workspace.open(tmp_path_factory.mktemp("p01"), "room")
    gt = orbit_cameras(30)
    make_ingest(ws, gt, source="input/room.mp4")
    SfmColmap(ws, {"max_num_features": 2048}).execute()
    return ws, gt


def load(ws: Workspace, name: str) -> dict:
    return json.loads((ws.root / "01_sfm" / name).read_text(encoding="utf-8"))


def test_outputs_format(video_run):
    ws, _ = video_run
    out = ws.root / "01_sfm"
    assert not (out / "_work").exists()

    cams = load(ws, "cameras.json")
    assert cams["contract_version"] == 1
    assert cams["convention"] == "opencv_world_to_camera" and cams["up_axis"] == "+z"
    for c in cams["cameras"]:
        assert set(c) == {
            "frame_id",
            "image",
            "width",
            "height",
            "fx",
            "fy",
            "cx",
            "cy",
            "world_to_camera",
        }
        assert np.asarray(c["world_to_camera"]).shape == (4, 4)
        with Image.open(ws.abs(c["image"])) as im:
            assert im.size == (c["width"], c["height"])

    with np.load(out / "points.npz") as pts:
        p = len(pts["xyz"])
        assert p > 0
        assert pts["xyz"].dtype == np.float32 and pts["rgb"].dtype == np.uint8
        assert pts["error"].dtype == np.float32 and pts["track_len"].dtype == np.int32
        assert pts["rgb"].shape == (p, 3) and pts["track_len"].min() >= 2

    recon = pycolmap.Reconstruction(out / "sparse")
    assert recon.num_reg_images() == len(cams["cameras"]) and recon.num_points3D() == p
    assert all(c.model_name == "PINHOLE" for c in recon.cameras.values())

    alignment = load(ws, "alignment.json")
    assert np.asarray(alignment["sim3_from_raw"]).shape == (4, 4)
    assert alignment["up_method"] == "floor_ransac"

    report = load(ws, "report.json")
    assert report["num_frames"] == 30 and report["matcher"] == "sequential"
    assert report["num_registered"] == len(cams["cameras"])

    manifest = load(ws, "manifest.json")
    assert manifest["phase"] == "01_sfm" and manifest["impl"] == "colmap"
    assert manifest["inputs"]["frames"]["path"] == "00_ingest/frames.json"


def test_reconstruction_matches_ground_truth(video_run):
    ws, gt = video_run
    cams = load(ws, "cameras.json")["cameras"]
    assert len(cams) >= 27  # >= 90 % registered
    assert load(ws, "report.json")["mean_reprojection_error_px"] < 1.0

    for c in cams:
        assert c["fx"] == pytest.approx(F, rel=0.03)
        w2c = np.asarray(c["world_to_camera"])
        gt_w2c = gt[int(c["frame_id"][1:]) - 1]
        # Gravity alignment: camera tilt w.r.t. +Z must match the ground truth.
        assert -w2c[1, 2] == pytest.approx(-gt_w2c[1, 2], abs=0.03)

    # Normalization: the raw-frame median camera center goes to the origin (a per-axis
    # median is not rotation invariant, so it is only near zero here) and RMS spread is 1.
    w2c = np.array([c["world_to_camera"] for c in cams])
    centers = -np.einsum("nji,nj->ni", w2c[:, :3, :3], w2c[:, :3, 3])
    assert np.allclose(np.median(centers, axis=0), 0, atol=0.02)
    rms = np.sqrt(np.mean(np.sum(centers**2, axis=1)))
    assert rms == pytest.approx(1.0, abs=0.01)


def test_photos_exhaustive_incremental(tmp_path):
    ws = Workspace.open(tmp_path, "room")
    make_ingest(ws, orbit_cameras(15), source=None)
    SfmColmap(
        ws, {"max_num_features": 2048, "mapper": "incremental", "min_registered": 12}
    ).execute()
    report = load(ws, "report.json")
    assert report["matcher"] == "exhaustive" and report["mapper"] == "incremental"


def test_unsupported_features(tmp_path):
    ws = Workspace.open(tmp_path, "room")
    make_ingest(ws, orbit_cameras(3), source=None)
    with pytest.raises(NotImplementedError):
        SfmColmap(ws, {"features": "aliked"}).execute()


# --- alignment math (no COLMAP) ---


def test_rotation_between():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.normal(size=(2, 3))
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        r = rotation_between(a, b)
        assert np.allclose(r @ a, b) and np.isclose(np.linalg.det(r), 1)
    r = rotation_between(np.array([0, 0, -1.0]), np.array([0, 0, 1.0]))
    assert np.allclose(r @ [0, 0, -1], [0, 0, 1])


def test_floor_plane_ignores_ceiling_and_walls():
    rng = np.random.default_rng(0)
    tilt = rotation_between(np.array([0, 0, 1.0]), np.array([0.2, 0.1, 1.0]))
    floor = np.c_[rng.uniform(-3, 3, (400, 2)), np.zeros(400)]
    ceiling = np.c_[rng.uniform(-3, 3, (600, 2)), np.full(600, 2.6)]  # bigger, but above cams
    wall = np.c_[np.full(500, 3.0), rng.uniform(-3, 3, 500), rng.uniform(0, 2.6, 500)]
    pts = np.vstack([floor, ceiling, wall]) @ tilt.T
    centers = np.c_[rng.uniform(-1, 1, (20, 2)), np.full(20, 1.5)] @ tilt.T
    up0 = tilt @ np.array([0.1, 0.0, 1.0])  # rough guess, ~6 deg off
    n, frac = floor_plane(pts, centers, up0 / np.linalg.norm(up0))
    assert np.degrees(np.arccos(n @ (tilt @ [0, 0, 1.0]))) < 0.5
    assert frac == pytest.approx(400 / 1500, abs=0.02)


def test_compute_alignment_falls_back_without_floor():
    # camera y axis = world -Z, so the camera up (-y) is already +Z
    rot = np.tile(np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]), (5, 1, 1))
    centers = np.random.default_rng(0).normal(size=(5, 3))
    a = compute_alignment("floor_ransac", rot, centers, np.zeros((0, 3)))
    assert a.up_method == "camera_up" and a.notes
    assert np.allclose(a.matrix[:3, :3] / a.scale, np.eye(3))
