"""Contract tests for phase 07_lifting (format of outputs, not quality).

A fake implementation returns known weights: a sofa whose masks also cover a pillow on it,
the pillow tracked in two broken tracks, a background and a short noise track. The gradient
trick of `gsplat_vote` is checked on a few gaussians where CUDA is available.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scenepipe.core.gaussians import Gaussians, rgb_to_sh_dc, write_ply
from scenepipe.core.masks import rle_encode
from scenepipe.core.phase import ContractError, register
from scenepipe.core.progress import StageProgress
from scenepipe.core.workspace import Workspace
from scenepipe.phases.p07_lifting.common import FrameWeights, LiftingBase

FRAMES = [f"f{i + 1:06d}" for i in range(4)]
SOFA, PILLOW, BACKGROUND = np.arange(0, 300), np.arange(300, 600), np.arange(600, 900)
OUTLIER = 0  # a sofa gaussian far away from the others
N = 900
# mask -> (track, label, gaussians it covers, embedding)
MASKS = {}
for i, fid in enumerate(FRAMES):
    MASKS[f"{fid}_m000"] = ("t0000", "sofa", np.concatenate([SOFA, PILLOW]), 0)
    MASKS[f"{fid}_m001"] = ("t0001" if i < 2 else "t0002", "pillow", PILLOW, 1)
MASKS["f000001_m002"] = ("t0009", "lamp", BACKGROUND[:60], 2)  # seen once: dropped


@register
class LiftingFake(LiftingBase):
    impl = "fake"

    def frame_weights(self, g, frames, stage: StageProgress):
        for fm in frames:
            vis_idx = np.arange(N)
            yield FrameWeights(
                vis_idx,
                np.ones(N, np.float32),
                [MASKS[m][2] for m in fm.mask_ids],
                [np.ones(len(MASKS[m][2]), np.float32) for m in fm.mask_ids],
            )


def make_ws(tmp_path: Path) -> Workspace:
    ws = Workspace.open(tmp_path, "demo")
    rng = np.random.default_rng(0)
    means = np.concatenate([
        rng.uniform([0, 0, 0], [2, 1, 0.5], (300, 3)),  # sofa
        rng.uniform([0.5, 0.3, 0.5], [0.9, 0.6, 0.7], (300, 3)),  # pillow on it
        rng.uniform([-3, -3, 0], [3, 3, 0.01], (300, 3)),  # floor
    ]).astype(np.float32)  # fmt: skip
    means[OUTLIER] = [50, 50, 50]
    (ws.root / "03_refinement").mkdir(parents=True)
    write_ply(ws.root / "03_refinement" / "gaussians.ply", Gaussians(
        means=means, sh_dc=rgb_to_sh_dc(np.full((N, 3), 0.5, np.float32)),
        sh_rest=np.zeros((N, 0, 3), np.float32), opacity=np.zeros(N, np.float32),
        scales=np.full((N, 3), -4, np.float32), quats=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
    ))  # fmt: skip

    sfm = ws.root / "01_sfm"
    sfm.mkdir()
    cams = [{
        "frame_id": fid, "image": f"01_sfm/images_undistorted/{fid}.jpg", "width": 8,
        "height": 6, "model": "PINHOLE", "fx": 5.0, "fy": 5.0, "cx": 4.0, "cy": 3.0,
        "world_to_camera": np.eye(4).tolist(),
    } for fid in FRAMES]  # fmt: skip
    doc = {"contract_version": 2, "convention": "opencv_world_to_camera", "cameras": cams}
    (sfm / "cameras.json").write_text(json.dumps(doc), encoding="utf-8")

    seg = ws.root / "05_segmentation"
    (seg / "masks").mkdir(parents=True)
    dets = []
    for fid in FRAMES:
        rles = []
        for mask_id, (track, label, _, _) in MASKS.items():
            if not mask_id.startswith(fid):
                continue
            m = np.zeros((6, 8), bool)
            m[1:4, 1:5] = True
            rles.append({"mask_id": mask_id, "rle": rle_encode(m)})
            dets.append({
                "mask_id": mask_id, "frame_id": fid, "label": label, "label_scores": {},
                "det_score": 0.9, "mask_score": 0.9, "bbox_xyxy": [1, 1, 5, 4], "area_px": 12,
                "track_id": track, "touches_border": False,
            })  # fmt: skip
        (seg / "masks" / f"{fid}.json").write_text(json.dumps(rles), encoding="utf-8")
    doc = {"contract_version": 1, "has_tracks": True, "frames": FRAMES, "detections": dets}
    (seg / "detections.json").write_text(json.dumps(doc), encoding="utf-8")

    emb_dir = ws.root / "06_mask_embedding"
    emb_dir.mkdir()
    ids = list(MASKS)
    np.save(emb_dir / "embeddings.npy", np.eye(4, dtype=np.float16)[[MASKS[m][3] for m in ids]])
    index = {"contract_version": 1, "mask_ids": ids, "skipped": []}
    (emb_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    meta = {"model": "fake", "dim": 4, "normalized": True}
    (emb_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    scores = {m: {"best": "couch", "probs": {"couch": 1.0}} for m in ids if MASKS[m][1] == "sofa"}
    doc = {"contract_version": 1, "labels": ["couch"], "text_template": "{}", "scores": scores}
    (emb_dir / "label_scores.json").write_text(json.dumps(doc), encoding="utf-8")
    return ws


CFG = {"num_debug": 0, "min_gaussians": 50}


def test_contract(tmp_path):
    ws = make_ws(tmp_path)
    LiftingFake(ws, CFG).execute()
    out = ws.root / "07_lifting"
    npz = np.load(out / "gaussian_instances.npz")
    inst_id = npz["instance_id"]
    assert inst_id.dtype == np.int32 and npz["confidence"].dtype == np.float16
    np.testing.assert_allclose(npz["visibility"], 4.0)
    doc = json.loads((out / "instances.json").read_text("utf-8"))
    assert doc["contract_version"] == 2 and doc["embedding_model"] == "fake"
    insts = doc["instances"]
    assert [i["label"] for i in insts] == ["pillow", "sofa"]  # the two broken tracks merged
    pillow, sofa = insts
    assert pillow["track_ids"] == ["t0001", "t0002"] and pillow["num_views"] == 4
    assert sofa["label_votes"]["sofa"] > sofa["label_votes"]["couch"] > 0  # + zero-shot
    assert (inst_id[PILLOW] == 0).all()  # the smaller instance wins on overlap
    assert (inst_id[SOFA[1:]] == 1).all() and inst_id[OUTLIER] == -1  # outlier removed
    assert (inst_id[BACKGROUND] == -1).all()
    assert pillow["num_gaussians"] == 300 and sofa["num_gaussians"] == 299
    feat = np.load(out / "instance_features.npy")
    assert feat.dtype == np.float16 and feat.shape == (2, 4)
    assert feat[0].argmax() == 1 and feat[1].argmax() == 0
    report = json.loads((out / "report.json").read_text("utf-8"))
    assert report["num_candidates"] == 4 and report["num_merged"] == 3


def test_larger_wins_without_preference(tmp_path):
    ws = make_ws(tmp_path)
    cfg = {**CFG, "prefer_smaller_on_overlap": False, "min_confidence": 0.0}
    LiftingFake(ws, cfg).execute()
    inst_id = np.load(ws.root / "07_lifting" / "gaussian_instances.npz")["instance_id"]
    insts = json.loads((ws.root / "07_lifting" / "instances.json").read_text("utf-8"))
    # equal coverage on the pillow: the sofa takes it and the pillow instance is left empty
    assert [i["label"] for i in insts["instances"]] == ["sofa"]
    assert (inst_id[SOFA[1:]] == 0).all() and (inst_id[PILLOW] == 0).all()


def test_validation_rejects_bad_output(tmp_path):
    ws = make_ws(tmp_path)
    phase = LiftingFake(ws, CFG)
    outs = phase.execute()
    npz = dict(np.load(outs["gaussian_instances"]))
    npz["instance_id"] = npz["instance_id"].copy()
    npz["instance_id"][-1] = 7  # no such instance
    np.savez(outs["gaussian_instances"], **npz)
    with pytest.raises(ContractError, match="instance_id"):
        phase.validate_outputs(outs)


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (gsplat)")
def test_gradient_trick_matches_alpha():
    """Weights of all gaussians in a mask sum to the rendered alpha inside it, and a gaussian
    behind an opaque one gets (almost) nothing."""
    pytest.importorskip("gsplat")
    from scenepipe.core.cameras import Camera
    from scenepipe.phases.p07_lifting.gsplat_vote import LiftingGsplatVote

    w, h = 64, 48
    cam = Camera("f", "x", w, h, 60.0, 60.0, w / 2, h / 2, np.eye(4))
    g = Gaussians(
        means=np.array([[-0.3, 0, 2], [0.3, 0, 2], [0.3, 0, 3]], np.float32),
        sh_dc=np.zeros((3, 3), np.float32), sh_rest=np.zeros((3, 0, 3), np.float32),
        opacity=np.array([0.0, 6.0, 0.0], np.float32),  # 0.5, ~1, 0.5; third is behind the second
        scales=np.log([[0.08] * 3, [0.2] * 3, [0.08] * 3]).astype(np.float32),
        quats=np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32),
    )  # fmt: skip
    phase = LiftingGsplatVote.__new__(LiftingGsplatVote)
    phase.config = {"device": "cuda"}
    phase._cache = None
    phase._antialiased_02 = lambda: True
    params = phase._params(g)
    left = np.zeros((h, w), np.float32)
    left[:, : w // 2] = 1
    masks = torch.from_numpy(np.stack([np.ones((h, w), np.float32), left], -1)).cuda()
    grad = phase._weights(params, cam, masks).cpu().numpy()
    with torch.no_grad():
        alpha = phase._rasterize(params, cam, torch.ones((3, 1), device="cuda"))[..., 0]
    alpha = alpha.cpu().numpy()
    assert grad[:, 0].sum() == pytest.approx(alpha.sum(), rel=1e-3)
    assert grad[:, 1].sum() == pytest.approx((alpha * left).sum(), rel=1e-3)
    assert grad[0, 1] > 0.9 * grad[0, 0]  # the left gaussian is in the left half
    phase._cache = None  # the occluded gaussian alone
    alone = Gaussians(
        *(a[[2]] for a in (g.means, g.sh_dc, g.sh_rest, g.opacity, g.scales, g.quats))
    )
    ones = torch.ones((h, w, 1), device="cuda")
    unoccluded = phase._weights(phase._params(alone), cam, ones).cpu().numpy()[0, 0]
    assert grad[2, 0] < 0.2 * unoccluded
