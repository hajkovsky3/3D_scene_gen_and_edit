"""3D instances of phase 07 against the manual annotation (scenepipe.annotate).

The instances are rendered into the annotated frames (every pixel gets the instance with the
highest rendered weight, as in lifting_consistency) and compared with the ground-truth objects.
The same instance name in several frames is one 3D object, so besides the per-frame matching
a consistent one is reported: every GT object gets the single 07 instance with the highest
IoU summed over its frames.
- miou_frame: per-frame one-to-one matching (greedy by IoU), mean over GT occurrences,
- miou_3d: with the consistent assignment, mean over GT occurrences (0 where not visible),
- recall_3d_at_0.5: share of GT objects whose mean IoU with their instance is >= 0.5,
- fragments: mean number of distinct 07 instances that are the per-frame best match
  (IoU >= 0.25) of one GT object (1 = not split),
- merged: GT objects that share their assigned 07 instance with another GT object,
- label_accuracy_3d: share of GT objects whose assigned instance has the GT class as label.
Stuff classes (configs/tag_stuff.txt) are left out unless --with-stuff.

Needs CUDA (gsplat). Environment `sp-gs`.

    python -m scenepipe.eval.lifting_gt --scene m360_counter
    python -m scenepipe.eval.lifting_gt --scene m360_counter --lifting 07_lifting/_cmp/x
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scenepipe.annotate.store import AnnotationStore
from scenepipe.core.cameras import load_cameras
from scenepipe.core.gaussians import read_ply
from scenepipe.core.workspace import Workspace
from scenepipe.eval.gt import REPO_ROOT, LabelMatcher, greedy_match, iou_matrix, resize_nearest
from scenepipe.eval.lifting_consistency import instance_map
from scenepipe.eval.segmentation_gt import stuff_classes


def evaluate(
    ws: Workspace,
    lifting: str = "07_lifting",
    scale: float = 0.5,
    with_stuff: bool = False,
    gaussians: str = "03_refinement/gaussians.ply",
) -> dict:
    store = AnnotationStore(ws)
    match = LabelMatcher(store)
    stuff = set() if with_stuff else stuff_classes(match)
    g = read_ply(ws.abs(gaussians))
    inst_id = np.load(ws.abs(f"{lifting}/gaussian_instances.npz"))["instance_id"]
    insts = json.loads(ws.abs(f"{lifting}/instances.json").read_text("utf-8"))["instances"]
    label = {i["instance_id"]: i["label"] for i in insts}
    k = max([i["instance_id"] for i in insts], default=-1) + 1
    cams = {c.frame_id: c for c in load_cameras(ws.abs("01_sfm/cameras.json"))}
    frames = [f for f in store.annotated_frames() if f in cams]
    if not frames:
        raise SystemExit("no annotated frame has a camera in 01_sfm")

    # iou[gt name][frame] = vector of IoU with every 07 instance
    per_obj: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    obj_label: dict[str, str] = {}
    frame_ious: list[float] = []
    for fid in frames:
        cam = cams[fid]
        small = cam.scaled(round(cam.width * scale), round(cam.height * scale))
        imap = instance_map(g, inst_id, max(k, 1), small)
        pred = [imap == j for j in range(k)]
        gt = [o for o in store.load_frame(fid) if match.canon(o.label) not in stuff]
        gt_masks = [resize_nearest(o.mask, small.height, small.width) for o in gt]
        iou = iou_matrix(gt_masks, pred)
        pairs = {gi: v for gi, _, v in greedy_match(iou, min_iou=1e-9)}
        for gi, o in enumerate(gt):
            frame_ious.append(pairs.get(gi, 0.0))
            per_obj[o.instance][fid] = iou[gi] if k else np.zeros(0)
            obj_label[o.instance] = o.label

    assigned: dict[str, int] = {}
    obj_miou: dict[str, float] = {}
    fragments = []
    occ_ious = []
    for name, by_frame in per_obj.items():
        stack = np.stack(list(by_frame.values())) if k else np.zeros((len(by_frame), 0))
        if k == 0:
            obj_miou[name] = 0.0
            occ_ious.extend([0.0] * len(by_frame))
            continue
        best = int(np.argmax(stack.sum(0)))
        assigned[name] = best
        obj_miou[name] = float(stack[:, best].mean())
        occ_ious.extend(stack[:, best].tolist())
        fragments.append(len({int(np.argmax(r)) for r in stack if r.max() >= 0.25}))
    owners = defaultdict(list)
    for name, j in assigned.items():
        owners[j].append(name)
    merged = sorted(n for names in owners.values() if len(names) > 1 for n in names)
    lab_ok = [match.same(obj_label[n], label[j]) for n, j in assigned.items()]
    n_obj = len(per_obj)
    thing_insts = [i for i in insts if match.canon(i["label"]) not in stuff]
    return {
        "lifting": lifting,
        "frames": frames,
        "num_gt_objects": n_obj,
        "num_gt_occurrences": len(occ_ious),
        "num_instances": len(insts),
        "num_instances_without_stuff": len(thing_insts),
        "miou_frame": round(float(np.mean(frame_ious)), 4) if frame_ious else 0.0,
        "miou_3d": round(float(np.mean(occ_ious)), 4) if occ_ious else 0.0,
        "recall_3d_at_0.5": round(float(np.mean([v >= 0.5 for v in obj_miou.values()])), 4)
        if n_obj
        else 0.0,
        "fragments_mean": round(float(np.mean(fragments)), 3) if fragments else 0.0,
        "merged": merged,
        "label_accuracy_3d": round(float(np.mean(lab_ok)), 4) if lab_ok else 0.0,
        "per_object": {
            n: {
                "frames": len(per_obj[n]),
                "miou": round(obj_miou[n], 3),
                "instance": assigned.get(n),
                "instance_label": label.get(assigned.get(n)),
            }
            for n in sorted(per_obj, key=lambda n: obj_miou[n])
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", default=str(REPO_ROOT / "workspace"))
    ap.add_argument("--lifting", default="07_lifting", help="directory with the 07 outputs")
    ap.add_argument("--scale", type=float, default=0.5, help="render resolution share")
    ap.add_argument("--with-stuff", action="store_true")
    args = ap.parse_args()
    ws = Workspace.open(Path(args.workspace), args.scene)
    result = evaluate(ws, args.lifting, args.scale, args.with_stuff)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
