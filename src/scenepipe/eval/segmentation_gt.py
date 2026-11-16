"""2D masks of phase 05 against the manual annotation (scenepipe.annotate).

On every annotated frame that 05 processed, ground-truth objects and 05 masks are matched
one-to-one by IoU (greedy, highest first). Reported:
- mIoU: mean over GT objects of the IoU of their match (0 if unmatched, match needs IoU > 0),
- recall@0.5 / precision@0.5: share of GT objects / 05 masks with a match of IoU >= 0.5,
- label accuracy: share of matched (IoU >= 0.5) objects whose 05 label equals the GT class
  (synonyms + annotations/aliases.yaml).
GT objects of `stuff` classes (configs/tag_stuff.txt, e.g. floor, wall) are left out unless
--with-stuff. Runs anywhere (numpy).

    python -m scenepipe.eval.segmentation_gt --scene m360_counter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenepipe.annotate.store import AnnotationStore
from scenepipe.core.masks import rle_decode
from scenepipe.core.workspace import Workspace
from scenepipe.eval.gt import REPO_ROOT, LabelMatcher, greedy_match, iou_matrix

STUFF = REPO_ROOT / "configs" / "tag_stuff.txt"


def stuff_classes(match: LabelMatcher) -> set[str]:
    if not STUFF.is_file():
        return set()
    lines = (s.split("#", 1)[0].strip() for s in STUFF.read_text("utf-8").splitlines())
    return {match.canon(s) for s in lines if s}


def evaluate(ws: Workspace, seg: str = "05_segmentation", with_stuff: bool = False) -> dict:
    store = AnnotationStore(ws)
    match = LabelMatcher(store)
    stuff = set() if with_stuff else stuff_classes(match)
    doc = json.loads(ws.abs(f"{seg}/detections.json").read_text("utf-8"))
    label = {d["mask_id"]: d["label"] for d in doc["detections"]}
    frames = [f for f in store.annotated_frames() if f in set(doc["frames"])]
    if not frames:
        raise SystemExit("no annotated frame was processed by 05")
    best_ious, n_pred, matched_pred, lab_ok, lab_n = [], 0, 0, 0, 0
    per_class: dict[str, list[float]] = {}
    for fid in frames:
        gt = [o for o in store.load_frame(fid) if match.canon(o.label) not in stuff]
        recs = json.loads(ws.abs(f"{seg}/masks/{fid}.json").read_text("utf-8"))
        pred = [rle_decode(r["rle"]).astype(bool) for r in recs]
        iou = iou_matrix([o.mask for o in gt], pred)
        pairs = {g: (p, v) for g, p, v in greedy_match(iou, min_iou=1e-9)}
        n_pred += len(pred)
        for gi, o in enumerate(gt):
            p, v = pairs.get(gi, (None, 0.0))
            best_ious.append(v)
            per_class.setdefault(match.canon(o.label), []).append(v)
            if p is not None and v >= 0.5:
                matched_pred += 1
                lab_n += 1
                lab_ok += match.same(o.label, label[recs[p]["mask_id"]])
    ious = np.array(best_ious)
    return {
        "segmentation": seg,
        "frames": frames,
        "num_gt_objects": len(ious),
        "num_pred_masks": n_pred,
        "miou": round(float(ious.mean()), 4) if len(ious) else 0.0,
        "recall_at_0.5": round(float((ious >= 0.5).mean()), 4) if len(ious) else 0.0,
        "precision_at_0.5": round(matched_pred / n_pred, 4) if n_pred else 0.0,
        "label_accuracy": round(lab_ok / lab_n, 4) if lab_n else 0.0,
        "per_class_count_iou": {
            k: [len(v), round(float(np.mean(v)), 3)]
            for k, v in sorted(per_class.items(), key=lambda kv: -len(kv[1]))
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", default=str(REPO_ROOT / "workspace"))
    ap.add_argument("--segmentation", default="05_segmentation", help="directory with 05 outputs")
    ap.add_argument("--with-stuff", action="store_true")
    args = ap.parse_args()
    ws = Workspace.open(Path(args.workspace), args.scene)
    result = evaluate(ws, args.segmentation, args.with_stuff)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
