"""Contract tests for phase 06_mask_embedding (format of outputs, not quality).

A fake encoder maps a crop to its mean color, so the phase mechanics (crops, batching,
averaging, skipped masks, zero-shot scores, validation) run without a model in a few seconds.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scenepipe.core.masks import mask_bbox, rle_encode
from scenepipe.core.phase import ContractError, register
from scenepipe.core.workspace import Workspace
from scenepipe.phases.p06_mask_embedding.common import (
    FILL_GRAY,
    MaskEmbeddingBase,
    crop_square,
    mask_background,
    square_box,
)

W, H = 80, 60
TEXTS = {"red": [1.0, 0.0, 0.0, 0.0], "green": [0.0, 1.0, 0.0, 0.0], "blue": [0.0, 0.0, 1.0, 0.0]}


@register
class MaskEmbeddingFake(MaskEmbeddingBase):
    impl = "fake"
    encoded: int = 0

    def load(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "fake:mean-color"

    @property
    def image_size(self) -> int:
        return 16

    def encode_images(self, crops: np.ndarray) -> np.ndarray:
        assert crops.dtype == np.uint8 and crops.shape[1:] == (16, 16, 3)
        type(self).encoded += len(crops)
        rgb = (crops.reshape(len(crops), -1, 3).mean(axis=1) - FILL_GRAY) / 128
        return np.concatenate([rgb, np.full((len(crops), 1), 0.05)], axis=1)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([TEXTS[t.split()[-1]] for t in texts])


def rect(x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def make_ws(tmp_path: Path) -> Workspace:
    """Two frames: a red square on f000001, a green one and a tiny blue one on f000002."""
    ws = Workspace.open(tmp_path, "demo")
    img_dir = ws.root / "01_sfm" / "images_undistorted"
    img_dir.mkdir(parents=True)
    seg = ws.root / "05_segmentation"
    (seg / "masks").mkdir(parents=True)
    objects = {
        "f000001": [("red", (255, 0, 0), rect(10, 10, 50, 50))],
        "f000002": [("green", (0, 255, 0), rect(30, 5, 75, 45)),
                    ("blue", (0, 0, 255), rect(0, 50, 8, 58))],
    }  # fmt: skip
    cams, dets = [], []
    for fid, objs in objects.items():
        img = np.full((H, W, 3), FILL_GRAY, np.uint8)
        rles = []
        for i, (label, color, m) in enumerate(objs):
            img[m] = color
            mask_id = f"{fid}_m{i:03d}"
            rles.append({"mask_id": mask_id, "rle": rle_encode(m)})
            dets.append({
                "mask_id": mask_id, "frame_id": fid, "label": label, "label_scores": {},
                "det_score": 0.9, "mask_score": 0.9, "bbox_xyxy": mask_bbox(m),
                "area_px": int(m.sum()), "track_id": None, "touches_border": False,
            })  # fmt: skip
        Image.fromarray(img).save(img_dir / f"{fid}.png")
        (seg / "masks" / f"{fid}.json").write_text(json.dumps(rles), encoding="utf-8")
        cams.append({
            "frame_id": fid, "image": f"01_sfm/images_undistorted/{fid}.png",
            "width": W, "height": H, "model": "PINHOLE",
            "fx": 50.0, "fy": 50.0, "cx": W / 2, "cy": H / 2,
            "world_to_camera": np.eye(4).tolist(),
        })  # fmt: skip
    doc = {"contract_version": 2, "convention": "opencv_world_to_camera", "cameras": cams}
    (ws.root / "01_sfm" / "cameras.json").write_text(json.dumps(doc), encoding="utf-8")
    doc = {"contract_version": 1, "has_tracks": False, "frames": list(objects), "detections": dets}
    (seg / "detections.json").write_text(json.dumps(doc), encoding="utf-8")
    tags = ws.root / "04_tagging"
    tags.mkdir()
    vocab = [{"tag": t, "frames": 1, "max_score": 0.9, "source": "model", "kind": "object"}
             for t in TEXTS]  # fmt: skip
    doc = {"contract_version": 1, "per_frame": {}, "vocabulary": vocab, "excluded": []}
    (tags / "tags.json").write_text(json.dumps(doc), encoding="utf-8")
    return ws


def test_contract(tmp_path):
    ws = make_ws(tmp_path)
    MaskEmbeddingFake.encoded = 0
    MaskEmbeddingFake(ws, {"batch_size": 3}).execute()
    out = ws.root / "06_mask_embedding"
    index = json.loads((out / "index.json").read_text("utf-8"))
    assert index == {
        "contract_version": 1, "mask_ids": ["f000001_m000", "f000002_m000"],
        "skipped": ["f000002_m001"],  # 8 x 8 px < min_crop_px
    }  # fmt: skip
    assert MaskEmbeddingFake.encoded == 4  # 2 masks x (bbox + masked)
    emb = np.load(out / "embeddings.npy")
    assert emb.dtype == np.float16 and emb.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1, atol=1e-3)
    assert emb[0].argmax() == 0 and emb[1].argmax() == 1  # red, green
    meta = json.loads((out / "meta.json").read_text("utf-8"))
    assert meta == {
        "model": "fake:mean-color", "dim": 4, "normalized": True,
        "crop_strategy": "multi", "text_template": "a photo of a {}",
    }  # fmt: skip
    scores = json.loads((out / "label_scores.json").read_text("utf-8"))
    assert scores["labels"] == ["red", "green", "blue"]
    assert scores["scores"]["f000001_m000"]["best"] == "red"
    assert scores["scores"]["f000002_m000"]["best"] == "green"
    assert sum(scores["scores"]["f000001_m000"]["probs"].values()) == pytest.approx(1, abs=1e-3)
    report = json.loads((out / "report.json").read_text("utf-8"))
    assert report["num_embedded"] == 2 and report["zero_shot_agrees_with_detector"] == 1.0
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["phase"] == "06_mask_embedding" and "tags" in manifest["inputs"]


def test_single_crop_without_label_scores(tmp_path):
    ws = make_ws(tmp_path)
    MaskEmbeddingFake.encoded = 0
    cfg = {"crop_strategy": "bbox", "compute_label_scores": False, "min_crop_px": 1}
    MaskEmbeddingFake(ws, cfg).execute()
    out = ws.root / "06_mask_embedding"
    assert MaskEmbeddingFake.encoded == 3
    assert not (out / "label_scores.json").exists()
    assert json.loads((out / "index.json").read_text("utf-8"))["skipped"] == []


def test_validation_rejects_bad_output(tmp_path):
    ws = make_ws(tmp_path)
    phase = MaskEmbeddingFake(ws, {})
    outs = phase.execute()
    np.save(outs["embeddings"], np.ones((2, 4), np.float16))  # not normalized
    with pytest.raises(ContractError, match="norms"):
        phase.validate_outputs(outs)
    np.save(outs["embeddings"], np.eye(3, 4, dtype=np.float16))  # one row too many
    with pytest.raises(ContractError, match="mask_ids"):
        phase.validate_outputs(outs)


def test_crop_helpers():
    assert square_box([10, 20, 30, 30], 0.0) == (10, 15, 20)
    assert square_box([10, 20, 30, 30], 0.5) == (0, 5, 40)
    img = np.arange(H * W, dtype=np.int32).reshape(H, W)
    c = crop_square(img, [0, 0, 10, 10], 0.5, fill=-1)  # reaches outside the image
    assert c.shape == (20, 20) and c[0, 0] == -1 and c[5, 5] == img[0, 0]
    crop = np.full((4, 4, 3), 200, np.uint8)
    m = np.zeros((4, 4), bool)
    m[1:3, 1:3] = True
    g = mask_background(crop, m, "gray")
    assert g[0, 0, 0] == FILL_GRAY and g[1, 1, 0] == 200
    assert mask_background(crop, m, "blur")[1, 1, 0] == 200
