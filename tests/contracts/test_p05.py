"""Contract tests for phase 05_segmentation (format of outputs, not quality).

A fake implementation draws known rectangles; the model-free helpers of `gdino_sam2` are
tested directly. The real models run only where transformers and the weights are available.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scenepipe.core.masks import rle_decode
from scenepipe.core.phase import ContractError, register
from scenepipe.core.progress import StageProgress
from scenepipe.core.workspace import Workspace
from scenepipe.phases.p05_segmentation.common import (
    FrameRef,
    RawMask,
    SegmentationBase,
    clean_mask,
)
from scenepipe.phases.p05_segmentation.gdino_sam2 import (
    Detection,
    build_prompt,
    match_boxes,
    nms,
    phrase_token_spans,
)

W, H = 80, 60
N_FRAMES = 5


def rect(x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


@register
class SegmentationFake(SegmentationBase):
    """Per frame: a sofa (track 0) with a 1-px hole and a speck, a lamp (track 1) at the
    border, a tiny mask (dropped) and a chair outside the tracks."""

    impl = "fake"
    tracks = True

    def segment(self, frames: list[FrameRef], vocabulary: list[str], stage: StageProgress):
        self.has_tracks = self.tracks
        for i, f in enumerate(frames):
            sofa = rect(10, 10, 40, 30)
            sofa[20, 20] = False  # hole
            sofa[50, 70] = True  # speck
            masks = [
                RawMask(sofa, "sofa", 0.8, 0.95, {"sofa": 0.8, "chair": 0.3}, 0),
                RawMask(rect(70, 0, 80, 20), "lamp", 0.5, 0.9, {"lamp": 0.5}, 1),
                RawMask(rect(0, 0, 2, 2), "lamp", 0.9, 0.9, {}, 2),
            ]
            if i % 2:
                masks.append(RawMask(rect(45, 35, 60, 55), "chair", 0.6, 0.7, {}, 3))
            stage.update((i + 1) / len(frames))
            yield f.frame_id, masks


def make_ws(tmp_path: Path) -> Workspace:
    ws = Workspace.open(tmp_path, "demo")
    img_dir = ws.root / "01_sfm" / "images_undistorted"
    img_dir.mkdir(parents=True)
    cams = []
    for i in range(N_FRAMES):
        fid = f"f{i + 1:06d}"
        Image.new("RGB", (W, H), (90, 90, 90)).save(img_dir / f"{fid}.jpg")
        cams.append({
            "frame_id": fid, "image": f"01_sfm/images_undistorted/{fid}.jpg",
            "width": W, "height": H, "model": "PINHOLE",
            "fx": 50.0, "fy": 50.0, "cx": W / 2, "cy": H / 2,
            "world_to_camera": np.eye(4).tolist(),
        })  # fmt: skip
    doc = {"contract_version": 2, "convention": "opencv_world_to_camera", "cameras": cams}
    (ws.root / "01_sfm" / "cameras.json").write_text(json.dumps(doc), encoding="utf-8")
    tags = ws.root / "04_tagging"
    tags.mkdir()
    vocab = [{"tag": t, "frames": 1, "max_score": 0.9, "source": "model", "kind": "object"}
             for t in ("sofa", "lamp", "chair")]  # fmt: skip
    doc = {"contract_version": 1, "per_frame": {}, "vocabulary": vocab, "excluded": []}
    (tags / "tags.json").write_text(json.dumps(doc), encoding="utf-8")
    return ws


def run(ws: Workspace, **config) -> dict:
    SegmentationFake(ws, {"num_previews": 2, **config}).execute()
    return json.loads((ws.root / "05_segmentation" / "detections.json").read_text("utf-8"))


def test_contract(tmp_path):
    ws = make_ws(tmp_path)
    doc = run(ws)
    out = ws.root / "05_segmentation"
    assert doc["contract_version"] == 1 and doc["has_tracks"] is True
    assert doc["frames"] == [f"f{i + 1:06d}" for i in range(N_FRAMES)]
    dets = doc["detections"]
    assert len(dets) == N_FRAMES * 2 + N_FRAMES // 2  # tiny masks dropped
    assert len({d["mask_id"] for d in dets}) == len(dets)
    first = [d for d in dets if d["frame_id"] == "f000001"]
    assert [d["label"] for d in first] == ["sofa", "lamp"]  # best det_score first
    sofa, lamp = first
    assert sofa["mask_id"] == "f000001_m000" and sofa["track_id"] == "t0000"
    assert sofa["bbox_xyxy"] == [10, 10, 40, 30] and sofa["area_px"] == 600  # cleaned
    assert sofa["label_scores"] == {"sofa": 0.8, "chair": 0.3}
    assert sofa["touches_border"] is False and lamp["touches_border"] is True
    assert set(sofa) == {
        "mask_id", "frame_id", "label", "label_scores", "det_score", "mask_score",
        "bbox_xyxy", "area_px", "track_id", "touches_border",
    }  # fmt: skip

    rles = json.loads((out / "masks" / "f000001.json").read_text("utf-8"))
    assert [r["mask_id"] for r in rles] == ["f000001_m000", "f000001_m001"]
    m = rle_decode(rles[0]["rle"])
    assert m.shape == (H, W) and m[20, 20] and not m[50, 70]
    assert sorted(p.name for p in (out / "previews").iterdir()) == ["f000001.jpg", "f000005.jpg"]
    report = json.loads((out / "report.json").read_text("utf-8"))
    assert report["num_tracks"] == 3 and report["labels"]["sofa"] == N_FRAMES
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["inputs"]["tags"]["path"] == "04_tagging/tags.json"


def test_without_tracks(tmp_path):
    SegmentationFake.tracks = False
    try:
        doc = run(make_ws(tmp_path), max_masks_per_frame=1)
    finally:
        SegmentationFake.tracks = True
    assert doc["has_tracks"] is False
    assert all(d["track_id"] is None for d in doc["detections"])
    assert len(doc["detections"]) == N_FRAMES  # one mask per frame


def test_validation_catches_bad_label(tmp_path):
    ws = make_ws(tmp_path)
    tags = ws.root / "04_tagging" / "tags.json"
    doc = json.loads(tags.read_text("utf-8"))
    doc["vocabulary"] = doc["vocabulary"][:2]  # no "chair"
    tags.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ContractError):
        run(ws)


def test_clean_mask():
    m = rect(10, 10, 40, 30)
    m[15:17, 15:17] = False  # 4-px hole
    m[0:3, 0:3] = True  # 9-px part
    out = clean_mask(m, max_hole_px=5, min_part_px=10)
    assert out[15, 15] and not out[0, 0]
    assert not clean_mask(m, 3, 0)[15, 15]  # hole above the limit
    single = rect(0, 0, 2, 2)
    assert clean_mask(single, 5, 100).sum() == 4  # the largest part always stays


def test_prompt_and_spans():
    text, spans = build_prompt(["coffee table", "sofa"])
    assert text == "coffee table . sofa ."
    assert [text[a:b] for a, b in spans] == ["coffee table", "sofa"]
    # offsets as from a BERT tokenizer: [CLS] coffee table . sofa . [SEP]
    offsets = [(0, 0), (0, 6), (7, 12), (13, 14), (15, 19), (20, 21), (0, 0)]
    assert phrase_token_spans(offsets, spans) == [[1, 2], [4]]


def test_nms_merges_label_scores():
    a = Detection([0, 0, 10, 10], "sofa", 0.8, {"sofa": 0.8})
    b = Detection([0, 0, 10, 11], "couch", 0.6, {"couch": 0.6, "sofa": 0.5})
    c = Detection([20, 20, 30, 30], "lamp", 0.4, {"lamp": 0.4})
    kept = nms([b, c, a], 0.7)
    assert [d.label for d in kept] == ["sofa", "lamp"]
    assert kept[0].label_scores == {"sofa": 0.8, "couch": 0.6}


def test_match_boxes():
    prev = [[0, 0, 40, 40], [40, 0, 80, 40]]
    new = [[42, 0, 80, 40], [0, 40, 80, 60], [0, 0, 38, 40]]
    assert match_boxes(new, prev, 0.5) == [1, None, 0]
    assert match_boxes(new, [], 0.5) == [None, None, None]
    assert match_boxes([[0, 0, 40, 40], [1, 0, 40, 40]], prev[:1], 0.5) == [0, None]  # 1:1


def test_frame_stride(tmp_path):
    doc = run(make_ws(tmp_path), frame_stride=2)
    assert doc["frames"] == ["f000001", "f000003", "f000005"]
    assert {d["frame_id"] for d in doc["detections"]} == set(doc["frames"])
    masks = sorted(p.name for p in (tmp_path / "demo" / "05_segmentation" / "masks").iterdir())
    assert masks == ["f000001.json", "f000003.json", "f000005.json"]
