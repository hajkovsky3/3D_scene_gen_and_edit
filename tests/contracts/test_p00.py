"""Contract tests for phase 00_ingest (format of outputs, not quality)."""

import itertools
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from scenepipe.core.phase import ContractError
from scenepipe.core.workspace import Workspace
from scenepipe.phases.p00_ingest.basic import REJECT_REASONS, IngestBasic, _Frame

W, H = 640, 480


@pytest.fixture(scope="module")
def texture() -> np.ndarray:
    """Large smooth random RGB texture; crops at different offsets look like camera motion."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (H + 600, W + 2400, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 6)


def crop(texture: np.ndarray, x: int, y: int = 100) -> np.ndarray:
    return np.ascontiguousarray(texture[y : y + H, x : x + W])


def make_ws(tmp_path) -> Workspace:
    ws = Workspace.open(tmp_path, "demo")
    (ws.root / "input").mkdir(parents=True)
    return ws


def run(ws: Workspace, **config) -> dict:
    IngestBasic(ws, config).execute()
    return json.loads((ws.root / "00_ingest" / "frames.json").read_text(encoding="utf-8"))


def check_contract(ws: Workspace, data: dict, max_side: int) -> None:
    assert data["contract_version"] == 1
    for i, fr in enumerate(data["frames"], start=1):
        assert fr["frame_id"] == f"f{i:06d}"
        assert fr["image"] == f"00_ingest/images/{fr['frame_id']}.jpg"
        with Image.open(ws.abs(fr["image"])) as im:
            assert im.size == (fr["width"], fr["height"])
            assert im.mode == "RGB"
        assert max(fr["width"], fr["height"]) <= max_side
        assert fr["source"].startswith("input/")
        assert set(fr) >= {"source_index", "timestamp_s", "sharpness", "exif_focal_mm"}
    assert all(r["reason"] in REJECT_REASONS for r in data["rejected"])
    manifest = json.loads((ws.root / "00_ingest" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "00_ingest" and manifest["impl"] == "basic"
    assert manifest["outputs"]["frames"]["path"] == "00_ingest/frames.json"
    assert manifest["config"]["max_side"] == max_side
    assert not (ws.root / "00_ingest" / "_staging").exists()


def test_photos(tmp_path, texture):
    ws = make_ws(tmp_path)
    inp = ws.root / "input"
    for i in range(22):
        Image.fromarray(crop(texture, i * 100)).save(inp / f"img_{i:02d}.jpg", quality=95)
    Image.fromarray(crop(texture, 0)).save(inp / "img_00_copy.png")  # duplicate of img_00
    (inp / "broken.jpg").write_bytes(b"not a jpeg")
    (inp / "iphone.HEIC").write_bytes(b"\x00" * 16)

    data = run(ws, max_side=320, min_frames=20)

    check_contract(ws, data, 320)
    assert len(data["frames"]) == 22
    assert data["frames"][0]["width"] == 320 and data["frames"][0]["height"] == 240
    assert data["frames"][0]["source_index"] is None
    reasons = {r["source"]: r["reason"] for r in data["rejected"]}
    assert reasons == {
        "input/broken.jpg": "unreadable",
        "input/img_00_copy.png": "duplicate",
        "input/iphone.HEIC": "unsupported_format",
    }


def test_exif_orientation_focal_and_order(tmp_path, texture):
    ws = make_ws(tmp_path)
    inp = ws.root / "input"
    for i, name in enumerate(["c.jpg", "b.jpg", "a.jpg"]):  # EXIF time order: c, b, a
        exif = Image.Exif()
        exif[0x0112] = 6 if name == "a.jpg" else 1  # a.jpg: rotate 90 deg on display
        sub = exif.get_ifd(0x8769)
        sub[36867] = f"2026:01:01 10:00:0{i}"
        sub[37386] = 4.2
        sub[41989] = 26
        Image.fromarray(crop(texture, i * 300)).save(inp / name, exif=exif)

    data = run(ws, min_frames=1)

    check_contract(ws, data, 1600)
    assert [f["source"] for f in data["frames"]] == ["input/c.jpg", "input/b.jpg", "input/a.jpg"]
    a = data["frames"][2]
    assert (a["width"], a["height"]) == (H, W)  # orientation applied
    assert a["exif_focal_mm"] == 4.2 and a["exif_focal_35mm"] == 26


def test_video(tmp_path, texture):
    ws = make_ws(tmp_path)
    path = ws.root / "input" / "room.mp4"
    fps, n = 30, 300  # 10 s
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert writer.isOpened()
    for i in range(n):
        rgb = crop(texture, i * 7)
        if i % 4 == 0:
            rgb = cv2.GaussianBlur(rgb, (0, 0), 5)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    data = run(ws, video_fps=2, sharpness_window=3, keyframe_motion=0, min_frames=5)

    check_contract(ws, data, 1600)
    frames = data["frames"]
    assert 10 <= len(frames) <= 20  # ~2 fps over 10 s, minus blur/duplicates
    idx = [f["source_index"] for f in frames]
    ts = [f["timestamp_s"] for f in frames]
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    assert ts == sorted(ts) and 0 <= ts[0] and ts[-1] <= 10.0
    assert all(f["source"] == "input/room.mp4" for f in frames)
    assert all(f["exif_focal_mm"] is None for f in frames)
    assert any(r["reason"] == "blur" for r in data["rejected"])


def test_video_motion_keyframes(tmp_path, texture):
    """keyframe_motion: frames follow camera motion, a still camera gives one per max gap."""
    ws = make_ws(tmp_path)
    path = ws.root / "input" / "pan.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for i in range(300):  # 4 s still, then 6 s panning 10 px per frame
        writer.write(cv2.cvtColor(crop(texture, max(0, i - 120) * 10), cv2.COLOR_RGB2BGR))
    writer.release()

    data = run(
        ws,
        keyframe_motion=0.1,
        keyframe_max_gap_s=2.0,
        sharpness_percentile_reject=0,
        dedup_threshold=1.0,
        min_frames=5,
    )

    check_contract(ws, data, 1600)
    ts = [f["timestamp_s"] for f in data["frames"]]
    # pan: 1800 px = 2.8 image widths (motion is relative to the long side) -> ~28 windows
    assert sum(t < 4 for t in ts) <= 3
    assert 20 <= sum(t >= 4 for t in ts) <= 36
    assert max(b - a for a, b in itertools.pairwise(ts)) <= 2 * 2.0  # two windows at most


def test_blur_gap_filling(tmp_path):
    """A long blurry stretch keeps a frame every <= blur_max_gap_s, none at the ends."""
    phase = IngestBasic(make_ws(tmp_path), {"blur_max_gap_s": 1.0})
    ts = [round(0.25 * i, 2) for i in range(41)]  # 0 .. 10 s
    frames = [
        _Frame("input/v.mp4", i, t, tmp_path / f"{i}.jpg", W, H, 1.0, np.zeros(1), "input/v.mp4")
        for i, t in enumerate(ts)
    ]
    blurry = [t < 1 or 3 <= t <= 7 or t > 9 for t in ts]
    phase._fill_gaps(frames, blurry)
    kept = [t for t, b in zip(ts, blurry) if not b]
    assert kept[0] == 1.0 and kept[-1] == 9.0  # leading/trailing blur stays rejected
    assert max(b - a for a, b in itertools.pairwise(kept)) <= 1.0
    assert sum(3 <= t <= 7 for t in kept) <= 5  # only sparse fill inside the stretch

    off = [t < 1 or 3 <= t <= 7 or t > 9 for t in ts]
    IngestBasic(make_ws(tmp_path / "off"), {"blur_max_gap_s": 0})._fill_gaps(frames, off)
    assert off == [t < 1 or 3 <= t <= 7 or t > 9 for t in ts]


def test_max_frames(tmp_path, texture):
    ws = make_ws(tmp_path)
    for i in range(10):
        Image.fromarray(crop(texture, i * 200)).save(ws.root / "input" / f"{i}.jpg")

    data = run(ws, max_frames=4, min_frames=1)

    check_contract(ws, data, 1600)
    assert [f["source"] for f in data["frames"]] == [f"input/{i}.jpg" for i in (0, 3, 6, 9)]
    assert sum(r["reason"] == "over_max_frames" for r in data["rejected"]) == 6


def test_too_few_frames(tmp_path, texture):
    ws = make_ws(tmp_path)
    Image.fromarray(crop(texture, 0)).save(ws.root / "input" / "only.jpg")

    with pytest.raises(ContractError, match="kept 1 frames"):
        IngestBasic(ws, {"min_frames": 20}).execute()
    assert not (ws.root / "00_ingest" / "manifest.json").exists()


def test_missing_input(tmp_path):
    ws = make_ws(tmp_path)
    with pytest.raises(FileNotFoundError):
        IngestBasic(ws, {}).execute()


def test_blur_gap_filling_skips_very_blurry(tmp_path):
    """Frames far below the median sharpness never fill a gap (SfM would mislocalize them)."""
    phase = IngestBasic(make_ws(tmp_path), {"blur_max_gap_s": 1.0, "blur_gap_min_ratio": 0.3})
    ts = [round(0.25 * i, 2) for i in range(41)]
    sharp = [0.1 if 3 <= t <= 7 else 1.0 for t in ts]  # the stretch is hopelessly blurry
    frames = [
        _Frame("input/v.mp4", i, t, tmp_path / f"{i}.jpg", W, H, s, np.zeros(1), "input/v.mp4")
        for i, (t, s) in enumerate(zip(ts, sharp, strict=True))
    ]
    blurry = [3 <= t <= 7 for t in ts]
    phase._fill_gaps(frames, blurry)
    assert blurry == [3 <= t <= 7 for t in ts]  # nothing re-admitted
