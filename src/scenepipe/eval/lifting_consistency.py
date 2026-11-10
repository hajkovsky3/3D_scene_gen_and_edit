"""2D consistency of the 3D instances of phase 07 (docs/evaluation.md).

Without 3D ground truth, the instances are rendered back into the frames of 05 and compared
with the 2D masks: every pixel gets the instance with the highest rendered weight (none if
the assigned instances cover less than half of the pixel), and for every mask that is a
source of an instance the IoU between the mask and that instance's pixels is computed.
The masks were inputs of 07, so this measures how faithfully the lifting reproduces them
(and how consistent they are across views), not segmentation quality against ground truth.

Needs CUDA (gsplat). Environment `sp-gs`.

    python -m scenepipe.eval.lifting_consistency --scene m360_counter
    python -m scenepipe.eval.lifting_consistency --scene m360_counter --lifting 07_lifting/_cmp/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenepipe.core.cameras import Camera, load_cameras
from scenepipe.core.gaussians import Gaussians, read_ply
from scenepipe.core.masks import rle_decode
from scenepipe.core.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[3]
MIN_COVER = 0.5


def instance_map(g: Gaussians, inst_id: np.ndarray, k: int, cam: Camera) -> np.ndarray:
    """int[H, W] instance with the highest rendered weight per pixel, -1 = none."""
    import torch
    from gsplat import rasterization

    dev = "cuda"
    onehot = torch.zeros((len(g), k), device=dev)
    ids = torch.from_numpy(inst_id.astype(np.int64)).to(dev)
    sel = ids >= 0
    onehot[sel, ids[sel]] = 1.0
    with torch.no_grad():
        out, _, _ = rasterization(
            means=torch.from_numpy(g.means).to(dev),
            quats=torch.from_numpy(g.quats).to(dev),
            scales=torch.exp(torch.from_numpy(g.scales).to(dev)),
            opacities=torch.sigmoid(torch.from_numpy(g.opacity).to(dev)),
            colors=onehot,
            viewmats=torch.tensor(cam.world_to_camera, dtype=torch.float32, device=dev)[None],
            Ks=torch.tensor(cam.K, dtype=torch.float32, device=dev)[None],
            width=cam.width,
            height=cam.height,
            sh_degree=None,
            packed=False,
        )
        w = out[0]
        best = w.argmax(-1)
        best[w.sum(-1) < MIN_COVER] = -1
    return best.cpu().numpy()


def evaluate(ws: Workspace, lifting: str, scale: float) -> dict:
    g = read_ply(ws.abs("03_refinement/gaussians.ply"))
    inst_id = np.load(ws.abs(f"{lifting}/gaussian_instances.npz"))["instance_id"]
    insts = json.loads(ws.abs(f"{lifting}/instances.json").read_text("utf-8"))["instances"]
    owner = {m: i["instance_id"] for i in insts for m in i["source_masks"]}
    doc = json.loads(ws.abs("05_segmentation/detections.json").read_text("utf-8"))
    cams = {c.frame_id: c for c in load_cameras(ws.abs("01_sfm/cameras.json"))}
    ious, in_inst, prec, rec, labels = [], [], [], [], []
    det_label = {d["mask_id"]: d["label"] for d in doc["detections"]}
    for fid in doc["frames"]:
        cam = cams[fid]
        small = cam.scaled(round(cam.width * scale), round(cam.height * scale))
        imap = instance_map(g, inst_id, max(len(insts), 1), small)
        path = ws.abs(f"05_segmentation/masks/{fid}.json")
        for r in json.loads(path.read_text("utf-8")):
            m = rle_decode(r["rle"])
            ys = (np.arange(small.height) / scale).astype(int).clip(0, cam.height - 1)
            xs = (np.arange(small.width) / scale).astype(int).clip(0, cam.width - 1)
            m = m[ys[:, None], xs[None, :]]
            k = owner.get(r["mask_id"])
            in_inst.append(k is not None)
            labels.append(det_label[r["mask_id"]])
            if k is None:
                ious.append(0.0)  # mask not in any kept instance
                prec.append(0.0)
                rec.append(0.0)
                continue
            p = imap == k
            inter = np.logical_and(p, m).sum()
            union = np.logical_or(p, m).sum()
            ious.append(float(inter / union) if union else 0.0)
            prec.append(float(inter / p.sum()) if p.any() else 0.0)
            rec.append(float(inter / m.sum()) if m.any() else 0.0)
    ious = np.array(ious)
    covered = np.array(in_inst, bool)
    prec, rec, labels = np.array(prec), np.array(rec), np.array(labels, dtype=object)
    per_label = {
        str(lab): [int((labels == lab).sum()), round(float(ious[labels == lab].mean()), 3)]
        for lab in sorted(set(labels), key=lambda x: -(labels == x).sum())
    }
    return {
        "lifting": lifting,
        "num_masks": len(ious),
        "masks_in_instances": int(covered.sum()),
        "mean_iou_all": round(float(ious.mean()), 4),
        "mean_iou_covered": round(float(ious[covered].mean()) if covered.any() else 0.0, 4),
        "median_iou_all": round(float(np.median(ious)), 4),
        "share_iou_above_0.5": round(float((ious > 0.5).mean()), 4),
        # of the masks in instances: share of the instance pixels inside the mask, and of the
        # mask covered by the instance (low precision = instance too large, e.g. merged)
        "mean_precision_covered": round(float(prec[covered].mean()), 4) if covered.any() else 0,
        "mean_recall_covered": round(float(rec[covered].mean()), 4) if covered.any() else 0,
        "per_label_count_iou": per_label,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", default=str(REPO_ROOT / "workspace"))
    ap.add_argument("--lifting", default="07_lifting", help="directory with the 07 outputs")
    ap.add_argument("--scale", type=float, default=0.5, help="render resolution share")
    args = ap.parse_args()
    ws = Workspace.open(Path(args.workspace), args.scene)
    print(json.dumps(evaluate(ws, args.lifting, args.scale), indent=2))


if __name__ == "__main__":
    main()
