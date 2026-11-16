"""Annotation store and the evaluations against it (tagging recall, 05 masks vs GT)."""

import base64
import json

import numpy as np
import pytest

from scenepipe.annotate.__main__ import pack_mask, unpack_mask
from scenepipe.annotate.store import AnnotatedObject, AnnotationStore, label_of
from scenepipe.core.masks import rle_encode
from scenepipe.core.workspace import Workspace
from scenepipe.eval import segmentation_gt, tagging_recall
from scenepipe.eval.gt import LabelMatcher, greedy_match, iou_matrix

H, W = 20, 30


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_label_of():
    assert label_of("Chair#2") == "chair"
    assert label_of("  oven   mitt ") == "oven mitt"


def test_pack_roundtrip():
    m = np.random.default_rng(0).random((H, W)) > 0.5
    assert np.array_equal(unpack_mask(pack_mask(m), W, H), m)
    # same bit order as the page: first pixel is the highest bit of the first byte
    assert base64.b64decode(pack_mask(box(0, 1, 0, 1)))[0] == 0x80


def test_store_roundtrip(tmp_path):
    ws = Workspace.open(tmp_path, "s")
    st = AnnotationStore(ws)
    st.set_frames(["f000001", "f000002"])
    st.save_frame("f000001", W, H, [AnnotatedObject("cup#1", "cup", box(0, 5, 0, 5))])
    assert st.annotated_frames() == ["f000001"]
    objs = st.load_frame("f000001")
    assert objs[0].instance == "cup#1" and objs[0].mask.sum() == 25
    st.save_vocabulary(["Cup#1", "table", "", "cup"])
    assert st.vocabulary() == ["cup", "table"]


def test_label_matcher(tmp_path):
    ws = Workspace.open(tmp_path, "s")
    st = AnnotationStore(ws)
    st.root.mkdir(parents=True)
    (st.root / "aliases.yaml").write_text("oven mitt: [glove]\n", encoding="utf-8")
    m = LabelMatcher(st)
    assert m.same("couch", "sofa")  # configs/tag_synonyms.yaml
    assert m.same("oven mitt", "glove") and not m.same("glove", "oven mitt")


def test_iou_and_greedy():
    a, b = box(0, 10, 0, 10), box(0, 10, 5, 15)
    iou = iou_matrix([a, b], [b, a])
    assert iou[0, 1] == 1 and iou[1, 0] == 1 and iou[0, 0] == pytest.approx(50 / 150)
    assert greedy_match(iou) == [(0, 1, 1.0), (1, 0, 1.0)]


def test_tagging_recall(tmp_path):
    ws = Workspace.open(tmp_path, "s")
    st = AnnotationStore(ws)
    st.save_vocabulary(["couch", "lamp", "cup"])
    (ws.root / "04_tagging").mkdir()
    tags = {"vocabulary": [{"tag": t} for t in ("sofa", "cup", "window")]}
    (ws.root / "04_tagging" / "tags.json").write_text(json.dumps(tags), encoding="utf-8")
    r = tagging_recall.evaluate(ws)
    assert r["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["missed"] == ["lamp"] and r["extra"] == ["window"]


def test_segmentation_gt(tmp_path):
    ws = Workspace.open(tmp_path, "s")
    st = AnnotationStore(ws)
    st.set_frames(["f000001"])
    gt = [
        AnnotatedObject("cup#1", "cup", box(0, 10, 0, 10)),
        AnnotatedObject("lamp#1", "lamp", box(10, 20, 20, 30)),
        AnnotatedObject("floor", "floor", box(15, 20, 0, 20)),  # stuff: left out
    ]
    st.save_frame("f000001", W, H, gt)
    seg = ws.root / "05_segmentation"
    (seg / "masks").mkdir(parents=True)
    preds = [("m0", "mug", box(0, 10, 0, 8)), ("m1", "lamp", box(0, 3, 25, 30))]
    det = {
        "frames": ["f000001"],
        "detections": [{"mask_id": i, "frame_id": "f000001", "label": lab} for i, lab, _ in preds],
    }
    (seg / "detections.json").write_text(json.dumps(det), encoding="utf-8")
    masks = [{"mask_id": i, "rle": rle_encode(m)} for i, _, m in preds]
    (seg / "masks" / "f000001.json").write_text(json.dumps(masks), encoding="utf-8")
    r = segmentation_gt.evaluate(ws)
    assert r["num_gt_objects"] == 2 and r["num_pred_masks"] == 2
    assert r["miou"] == pytest.approx((0.8 + 0.0) / 2)  # lamp mask does not overlap the GT
    assert r["recall_at_0.5"] == 0.5 and r["precision_at_0.5"] == 0.5
    assert r["label_accuracy"] == 0.0  # mug != cup without an alias


def test_sweep_variants():
    from scenepipe.eval.lifting_sweep import variants

    grid = {"a": [1, 2], "b": [3, 4]}
    assert len(variants(grid, False)) == 4 and variants(grid, False)[0] == {"a": 1, "b": 3}
    assert variants(grid, True) == [{}, {"a": 1}, {"a": 2}, {"b": 3}, {"b": 4}]
    assert variants({}, False) == [{}]
