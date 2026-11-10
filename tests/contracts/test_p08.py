"""Contract tests for phase 08_scene_graph (format of outputs, relations on synthetic boxes).

A synthetic room of boxes filled with gaussians: floor (structure), a rotated table with thin
legs, a cup on it, a chair next to it, a sofa with a backrest and a pillow on the seat (the
pillow lies inside the sofa's box but is on it), a cabinet with a box inside, a bowl with an
apple, a plant wider than its pot, and one unlabeled object. Everything runs on the CPU.
"""

import json
import re
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from scenepipe.core.gaussians import Gaussians, rgb_to_sh_dc, write_ply
from scenepipe.core.masks import rle_encode
from scenepipe.core.phase import ContractError
from scenepipe.core.scene_graph import validate_scene_graph
from scenepipe.core.state import current, read_state, write_initial_state
from scenepipe.core.workspace import Workspace
from scenepipe.phases.p08_scene_graph.common import (
    clip_polygon,
    convex_hull,
    oriented_box,
    polygon_area,
)
from scenepipe.phases.p08_scene_graph.florence import caption_attributes
from scenepipe.phases.p08_scene_graph.geometric_only import SceneGraphGeometricOnly
from scenepipe.phases.p08_scene_graph.geometric_vlm import (
    SceneGraphGeometricVlm,
    apply_description,
    build_prompt,
    check_vote,
    normalize_name,
    outlined_crop,
    parse_response,
)

FRAMES = [f"f{i + 1:06d}" for i in range(12)]
TABLE_YAW = np.deg2rad(30)
DENSITY = 60000  # points per unit volume (min 300 per part)


def _box(rng, lo, hi, yaw=0.0, origin=(0, 0, 0)) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = max(300, int(DENSITY * np.prod(hi - lo)))
    p = rng.uniform(lo, hi, (n, 3))
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return p @ rot.T + np.asarray(origin, float)


def _objects(rng) -> dict[str, tuple[str | None, np.ndarray]]:
    """name -> (label, points); instance ids follow this order sorted by size later."""
    t = (0.0, 0.0, 0.0)
    legs = [_box(rng, [x - 0.02, y - 0.02, 0], [x + 0.02, y + 0.02, 0.7], TABLE_YAW, t)
            for x in (-0.45, 0.45) for y in (-0.25, 0.25)]  # fmt: skip
    table = np.concatenate([_box(rng, [-0.5, -0.3, 0.7], [0.5, 0.3, 0.75], TABLE_YAW, t), *legs])
    cup = _box(rng, [0.15, -0.05, 0.75], [0.25, 0.05, 0.87], TABLE_YAW, t)
    cabinet = np.concatenate([
        _box(rng, [1.0, -1.6, 0.0], [1.6, -1.0, 0.03]),  # bottom
        _box(rng, [1.0, -1.6, 0.97], [1.6, -1.0, 1.0]),  # top
        _box(rng, [1.0, -1.6, 0.03], [1.03, -1.0, 0.97]),  # sides and back
        _box(rng, [1.57, -1.6, 0.03], [1.6, -1.0, 0.97]),
        _box(rng, [1.0, -1.6, 0.03], [1.6, -1.57, 0.97]),
    ])  # fmt: skip
    return {
        "floor": ("floor", _box(rng, [-2.5, -2.5, -0.01], [2.5, 2.5, 0.0])),
        "table": ("table", table),
        "cup": ("cup", cup),
        "chair": ("chair", _box(rng, [0.62, -0.15, 0.0], [0.92, 0.15, 0.9])),
        "sofa": (
            "sofa",
            np.concatenate(
                [
                    _box(rng, [-1.9, 1.0, 0.0], [-0.9, 1.6, 0.45]),  # seat
                    _box(rng, [-1.9, 1.45, 0.45], [-0.9, 1.6, 0.9]),  # backrest
                ]
            ),
        ),
        "pillow": ("pillow", _box(rng, [-1.6, 1.1, 0.45], [-1.3, 1.4, 0.6])),
        "cabinet": ("cabinet", cabinet),
        "box": ("box", _box(rng, [1.2, -1.4, 0.03], [1.4, -1.2, 0.2])),
        "bowl": (
            "bowl",
            np.concatenate(
                [
                    _box(rng, [-1.1, -1.1, 0.0], [-0.9, -0.9, 0.02]),  # bottom and four walls
                    _box(rng, [-1.1, -1.1, 0.02], [-1.08, -0.9, 0.12]),
                    _box(rng, [-0.92, -1.1, 0.02], [-0.9, -0.9, 0.12]),
                    _box(rng, [-1.08, -1.1, 0.02], [-0.92, -1.08, 0.12]),
                    _box(rng, [-1.08, -0.92, 0.02], [-0.92, -0.9, 0.12]),
                ]
            ),
        ),
        "apple": ("apple", _box(rng, [-1.04, -1.04, 0.02], [-0.96, -0.96, 0.1])),
        "pot": ("pot", _box(rng, [0.0, -1.5, 0.0], [0.1, -1.4, 0.15])),
        "plant": ("plant", _box(rng, [-0.1, -1.6, 0.15], [0.2, -1.3, 0.45])),  # wider than the pot
        "thing": (None, _box(rng, [-1.8, -1.8, 0.0], [-1.6, -1.6, 0.2])),
    }


def make_ws(tmp_path: Path) -> tuple[Workspace, dict[str, int]]:
    ws = Workspace.open(tmp_path, "demo")
    rng = np.random.default_rng(0)
    objs = _objects(rng)
    names = sorted(objs, key=lambda k: -len(objs[k][1]))  # instance 0 = the largest
    ids = {name: i for i, name in enumerate(names)}
    noise = rng.uniform(-3, 3, (200, 3))  # unassigned gaussians
    means = np.concatenate([*(objs[k][1] for k in names), noise]).astype(np.float32)
    inst = np.concatenate([np.full(len(objs[k][1]), ids[k]) for k in names] + [np.full(200, -1)])
    n = len(means)
    opacity = np.full(n, 2.0, np.float32)
    opacity[: n // 50] = -6.0  # a few transparent gaussians, ignored by the geometry
    perm = rng.permutation(n)  # the file order of gaussians is arbitrary
    means, inst, opacity = means[perm], inst[perm].astype(np.int32), opacity[perm]
    (ws.root / "03_refinement").mkdir(parents=True)
    write_ply(ws.root / "03_refinement" / "gaussians.ply", Gaussians(
        means=means, sh_dc=rgb_to_sh_dc(np.full((n, 3), 0.5, np.float32)),
        sh_rest=np.zeros((n, 0, 3), np.float32), opacity=opacity,
        scales=np.full((n, 3), -4, np.float32), quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
    ))  # fmt: skip

    sfm = ws.root / "01_sfm"
    sfm.mkdir()
    cams = []
    for i, fid in enumerate(FRAMES):  # ring of cameras with radius 2 -> scene scale 2.2
        a = 2 * np.pi * i / len(FRAMES)
        w2c = np.eye(4)
        w2c[:3, 3] = -np.array([2 * np.cos(a), 2 * np.sin(a), 1.0])  # rotation irrelevant here
        cams.append({"frame_id": fid, "image": f"01_sfm/images_undistorted/{fid}.jpg",
                     "width": 100, "height": 80, "fx": 80.0, "fy": 80.0, "cx": 50.0, "cy": 40.0,
                     "world_to_camera": w2c.tolist()})  # fmt: skip
    doc = {"contract_version": 2, "convention": "opencv_world_to_camera", "cameras": cams}
    (sfm / "cameras.json").write_text(json.dumps(doc), encoding="utf-8")

    tags = ws.root / "04_tagging"
    tags.mkdir()
    vocab = [{"tag": objs[k][0], "kind": "stuff" if k == "floor" else "object"}
             for k in names if objs[k][0]]  # fmt: skip
    doc = {"contract_version": 1, "per_frame": {}, "vocabulary": vocab, "excluded": []}
    (tags / "tags.json").write_text(json.dumps(doc), encoding="utf-8")

    # masks: every object in every frame, area growing with the frame index for the cup
    seg = ws.root / "05_segmentation"
    seg.mkdir()
    dets, sources = [], {k: [] for k in names}
    for fi, fid in enumerate(FRAMES):
        for j, k in enumerate(names):
            mask_id = f"{fid}_m{j:03d}"
            area = 100 + (fi * 10 if k == "cup" else 0)
            dets.append({"mask_id": mask_id, "frame_id": fid, "label": objs[k][0],
                         "label_scores": {}, "det_score": 0.9, "mask_score": 0.9,
                         "bbox_xyxy": [0, 0, 10, 10], "area_px": area, "track_id": None,
                         "touches_border": k == "cup" and fi == 11})  # fmt: skip
            sources[k].append(mask_id)
    doc = {"contract_version": 1, "has_tracks": False, "frames": FRAMES, "detections": dets}
    (seg / "detections.json").write_text(json.dumps(doc), encoding="utf-8")

    lift = ws.root / "07_lifting"
    lift.mkdir()
    np.savez_compressed(lift / "gaussian_instances.npz", instance_id=inst,
                        confidence=np.ones(n, np.float16), visibility=np.ones(n, np.float32))  # fmt: skip
    insts = [{"instance_id": ids[k], "label": objs[k][0],
              "label_votes": {objs[k][0]: 1.0} if objs[k][0] else {},
              "num_gaussians": int((inst == ids[k]).sum()), "num_views": len(FRAMES),
              "source_masks": sources[k], "track_ids": [], "feature_row": ids[k]}
             for k in names]  # fmt: skip
    doc = {"contract_version": 2, "embedding_model": "fake", "instances": insts}
    (lift / "instances.json").write_text(json.dumps(doc), encoding="utf-8")
    feat = np.eye(len(names), 16, dtype=np.float16)
    np.save(lift / "instance_features.npy", feat)
    return ws, ids


def _relations(doc: dict) -> set[tuple[str, str, str]]:
    label = {n["node_id"]: n["label"] or "?" for n in doc["nodes"]}
    return {(label[e["source"]], e["relation"], label[e["target"]]) for e in doc["edges"]}


def test_contract_and_relations(tmp_path):
    ws, ids = make_ws(tmp_path)
    outs = SceneGraphGeometricOnly(ws, {}).execute()
    out = ws.root / "08_scene_graph"
    assert (out / "manifest.json").is_file() and (out / "graph.html").is_file()
    doc = json.loads(outs["scene_graph"].read_text("utf-8"))
    assert doc["contract_version"] == 2 and doc["embedding_model"] == "fake"
    assert doc["scene_scale"] == pytest.approx(2.2, rel=1e-3)
    nodes = {n["label"]: n for n in doc["nodes"]}
    assert [n["instance_id"] for n in doc["nodes"]] == list(range(len(ids)))
    assert nodes["floor"]["is_structure"] and not nodes["table"]["is_structure"]
    assert nodes["cup"]["caption"] == "cup" and nodes[None]["caption"] == ""

    rel = _relations(doc)
    expected_on = {("table", "on", "floor"), ("cup", "on", "table"), ("chair", "on", "floor"),
                   ("sofa", "on", "floor"), ("pillow", "on", "sofa"), ("cabinet", "on", "floor"),
                   ("?", "on", "floor"), ("bowl", "on", "floor"), ("pot", "on", "floor"),
                   ("plant", "on", "pot")}  # fmt: skip
    assert {r for r in rel if r[1] == "on"} == expected_on
    inside = {("box", "inside", "cabinet"), ("apple", "inside", "bowl")}
    assert {r for r in rel if r[1] == "inside"} == inside
    assert ("chair", "near", "table") in rel or ("table", "near", "chair") in rel
    assert not any(r[1] == "near" and "floor" in (r[0], r[2]) for r in rel)

    # rotated table: heading recovered, extent of the top (legs inside it)
    table = nodes["table"]["obb"]
    rot = np.array(table["rotation"])
    assert np.degrees(np.arctan2(rot[1, 0], rot[0, 0])) == pytest.approx(30, abs=2)
    np.testing.assert_allclose(table["extent"], [1.0, 0.6, 0.76], atol=0.05)
    np.testing.assert_allclose(rot[:, 2], [0, 0, 1])

    # best views: the cup is largest in the last frame, but that mask touches the border
    cup_views = nodes["cup"]["best_views"]
    assert len(cup_views) == 3 and cup_views[0] == "f000011"
    pos = sorted(FRAMES.index(f) for f in cup_views)
    assert min(np.diff(pos)) >= 5

    feat = np.load(outs["node_features"])
    assert feat.dtype == np.float16 and feat.shape == (len(ids), 16)
    report = json.loads(outs["report"].read_text("utf-8"))
    assert report["num_structure"] == 1 and report["num_unlabeled"] == 1
    assert report["unsupported_nodes"] == []


def test_initial_state(tmp_path):
    ws, _ = make_ws(tmp_path)
    SceneGraphGeometricOnly(ws, {}).execute()
    path = write_initial_state(ws)
    assert path == ws.root / "state" / "v000" / "state.json"
    st = read_state(ws, "v000")
    assert st["parent"] is None and st["scene_graph"] == "08_scene_graph/scene_graph.json"
    assert st["gaussians"] == "03_refinement/gaussians.ply" and st["embedding_model"] == "fake"
    assert all(not Path(v).is_absolute() for v in st.values() if isinstance(v, str))
    assert current(ws) == "v000"


def test_full_obb_mode(tmp_path):
    ws, _ = make_ws(tmp_path)
    outs = SceneGraphGeometricOnly(ws, {"obb_mode": "full"}).execute()
    doc = json.loads(outs["scene_graph"].read_text("utf-8"))
    for n in doc["nodes"]:
        rot = np.array(n["obb"]["rotation"])
        assert np.linalg.det(rot) == pytest.approx(1, abs=1e-6)


def test_validation_rejects_bad_output(tmp_path):
    ws, _ = make_ws(tmp_path)
    phase = SceneGraphGeometricOnly(ws, {})
    outs = phase.execute()
    doc = json.loads(outs["scene_graph"].read_text("utf-8"))
    good = json.dumps(doc)

    def broken(edit) -> None:
        d = json.loads(good)
        edit(d)
        outs["scene_graph"].write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(ContractError):
            phase.validate_outputs(outs)

    a, b = doc["nodes"][0]["node_id"], doc["nodes"][1]["node_id"]
    edge = {"confidence": 0.5, "origin": "geometric"}
    broken(lambda d: d["edges"].append({**edge, "source": a, "target": a, "relation": "on"}))
    broken(lambda d: d["edges"].append({**edge, "source": a, "target": "n9999", "relation": "on"}))
    broken(lambda d: d["edges"].append({**edge, "source": a, "target": b, "relation": "left_of"}))
    broken(lambda d: d["edges"].extend([
        {**edge, "source": a, "target": b, "relation": "on"},
        {**edge, "source": b, "target": a, "relation": "on"},
    ]))  # fmt: skip
    broken(lambda d: d["nodes"][0]["obb"].update(extent=[1, 0, 1]))
    broken(lambda d: d["nodes"][0]["obb"].update(rotation=[[2, 0, 0], [0, 1, 0], [0, 0, 1]]))
    broken(lambda d: d["nodes"].pop())  # one node per instance of 07
    broken(lambda d: d["nodes"][0].update(best_views=["f999999"]))


def test_validate_scene_graph_on_cycle():
    rot = np.eye(3).tolist()
    node = {"label": "x", "obb": {"center": [0, 0, 0], "extent": [1, 1, 1], "rotation": rot},
            "aabb": {"min": [0, 0, 0], "max": [1, 1, 1]}, "num_gaussians": 1, "feature_row": 0,
            "best_views": []}  # fmt: skip
    nodes = [{**node, "node_id": f"n{i:04d}", "instance_id": i} for i in range(3)]
    edges = [{"source": f"n{i:04d}", "target": f"n{(i + 1) % 3:04d}", "relation": "on",
              "confidence": 1.0, "origin": "geometric"} for i in range(3)]  # fmt: skip
    doc = {"up_axis": "+z", "nodes": nodes, "edges": edges[:2]}
    validate_scene_graph(doc)
    with pytest.raises(ValueError, match="cycle"):
        validate_scene_graph({**doc, "edges": edges})


def test_polygons_and_box():
    square = convex_hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float))
    assert len(square) == 4 and polygon_area(square) == pytest.approx(1)
    shifted = square + 0.5
    assert polygon_area(clip_polygon(shifted, square)) == pytest.approx(0.25)
    assert polygon_area(clip_polygon(square + 2, square)) == pytest.approx(0)
    rng = np.random.default_rng(1)
    pts = rng.uniform([-2, -0.5, 0], [2, 0.5, 1], (5000, 3))
    c, s = np.cos(-0.4), np.sin(-0.4)
    pts = pts @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    center, extent, rot = oriented_box(pts, "gravity_aligned", 0.5, 1e-3)
    assert np.arctan2(rot[1, 0], rot[0, 0]) == pytest.approx(-0.4, abs=0.02)
    np.testing.assert_allclose(extent, [4, 1, 1], atol=0.05)
    np.testing.assert_allclose(center, [0, 0, 0.5], atol=0.03)


# --- geometric_vlm with a fake model ----------------------------------------------------------


class FakeVlm(SceneGraphGeometricVlm):
    """No model: canned answers keyed by the label in the prompt, requests recorded."""

    answers: ClassVar[dict[str | None, str]] = {
        "chair": '{"name": "Sofas", "label_correct": false, "caption": "a small sofa",'
        ' "color": "Red", "material": "unknown"}',
        "cup": 'Sure! ```json\n{"name": "mug", "label_correct": false, "caption": "a mug",'
        ' "color": "white", "material": "ceramic"}\n```',
        "table": "I cannot tell.",
        "pot": '{"name": "bowl", "label_correct": false, "caption": "a pot"}',
        None: '{"name": "a bowl", "label_correct": false, "caption": "a thing",'
        ' "color": "none", "material": "wood"}',
    }

    def load(self) -> None:
        self.requests: list[tuple[list, str]] = []

    def generate(self, requests):
        self.requests += requests
        out = []
        for _images, prompt in requests:
            if prompt.startswith("Which word"):  # relabel check: keep the pot a pot
                out.append("Pot." if '"pot"' in prompt else "sofa")
                continue
            m = re.search(r'outlined in red "([^"]+)"', prompt)
            label = m.group(1) if m else None
            default = (f'{{"name": "{label}", "label_correct": true, "caption": "a gray {label}",'
                       ' "color": "gray", "material": "plastic"}')  # fmt: skip
            out.append(self.answers.get(label, default))
        return out


def _add_images_and_masks(ws: Workspace) -> None:
    img_dir = ws.root / "01_sfm" / "images_undistorted"
    img_dir.mkdir()
    dets = json.loads((ws.root / "05_segmentation" / "detections.json").read_text("utf-8"))
    (ws.root / "05_segmentation" / "masks").mkdir()
    for fid in FRAMES:
        Image.fromarray(np.full((80, 100, 3), 90, np.uint8)).save(img_dir / f"{fid}.jpg")
        mask = np.zeros((80, 100), bool)
        mask[2:9, 3:8] = True
        rles = [{"mask_id": d["mask_id"], "rle": rle_encode(mask)}
                for d in dets["detections"] if d["frame_id"] == fid]  # fmt: skip
        (ws.root / "05_segmentation" / "masks" / f"{fid}.json").write_text(json.dumps(rles))


def test_geometric_vlm_fake_model(tmp_path):
    ws, ids = make_ws(tmp_path)
    _add_images_and_masks(ws)
    phase = FakeVlm(ws, {"vlm_batch_size": 3})
    outs = phase.execute()
    doc = json.loads(outs["scene_graph"].read_text("utf-8"))
    nodes = {n["instance_id"]: n for n in doc["nodes"]}
    by_old = {name: nodes[i] for name, i in ids.items()}

    # floor (structure) is not described; every other node with 2 crops of the right size
    described = [r for r in phase.requests if not r[1].startswith("Which word")]
    assert len(described) == len(ids) - 1
    assert all(len(imgs) == 2 and imgs[0].size == (448, 448) for imgs, _ in described)
    checks = [r for r in phase.requests if r[1].startswith("Which word")]
    assert len(checks) == 4 and all(len(imgs) == 1 for imgs, _ in checks)  # chair, pot
    assert by_old["floor"]["caption"] == "floor" and by_old["floor"]["attributes"] == {}

    # accepted correction (plural of a vocabulary tag), rejected one (not in the vocabulary)
    chair = by_old["chair"]
    assert (chair["label"], chair["label_original"]) == ("sofa", "chair")
    assert chair["attributes"] == {"color": "red"} and chair["caption"] == "a small sofa"
    cup = by_old["cup"]
    assert (cup["label"], cup["label_original"]) == ("cup", None)
    assert cup["attributes"] == {"color": "white", "material": "ceramic"}
    thing = by_old["thing"]  # unlabeled: named from the vocabulary
    assert (thing["label"], thing["label_original"]) == ("bowl", None)
    assert thing["attributes"] == {"material": "wood"}
    assert by_old["table"]["caption"] == "table"  # unparsable answer: unchanged
    assert by_old["pot"]["label"] == "pot" and by_old["pot"]["caption"] == "a pot"

    report = json.loads(outs["report"].read_text("utf-8"))
    assert report["num_vlm_failed"] == 1 and report["vlm_failed"] == [by_old["table"]["node_id"]]
    assert {(r["from"], r["to"]) for r in report["relabeled"]} == {("chair", "sofa"),
                                                                    (None, "bowl")}  # fmt: skip
    assert [(r["from"], r["to"]) for r in report["relabel_rejected"]] == [("pot", "bowl")]
    assert report["num_unlabeled"] == 0


def test_vlm_helpers():
    img = np.zeros((50, 80, 3), np.uint8)
    mask = np.zeros((50, 80), bool)
    mask[10:20, 60:78] = True
    crop = outlined_crop(img, mask, [60, 10, 78, 20], 1.6, 64)
    assert crop.shape == (64, 64, 3)
    red = (crop[..., 0] > 200) & (crop[..., 1] < 50)
    assert red.any() and (crop == 127).all(axis=2).any()  # outline, gray beyond the image

    assert parse_response('x {"a": 1} y') == {"a": 1}
    assert parse_response("[1, 2]") is None and parse_response("{bad") is None
    assert normalize_name("  The  Coffee-Table!") == "coffee-table"
    assert "not identified" in build_prompt(None, ["b", "a"]) and "a, b" in build_prompt(
        None, ["b", "a"]
    )
    node = {"label": "cup", "label_original": None, "label_votes": {"mug": 0.3},
            "caption": "cup", "attributes": {}}  # fmt: skip
    resp = {"name": "mug", "label_correct": False}
    assert apply_description(dict(node), resp, [], "off") is None
    assert apply_description(dict(node), resp, [], "vocabulary") == "mug"
    assert (
        apply_description(dict(node), {"name": "vase", "label_correct": False}, [], "any") == "vase"
    )
    assert apply_description(dict(node), {"name": "vase", "label_correct": True}, [], "any") is None
    assert check_vote("Sofas.", "pillow", "sofa") and not check_vote("pillow", "pillow", "sofa")
    assert not check_vote("pillow or sofa", "pillow", "sofa")


def test_florence_caption_attributes():
    assert caption_attributes("A Grey wooden chair on a white background.") == {
        "color": "gray", "material": "wood",
    }  # fmt: skip
    assert caption_attributes("a close up of a piece of cloth on a gray surface") == {}
    assert caption_attributes("A pair of black gloves") == {"color": "black"}
