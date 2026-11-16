"""Recall and precision of the vocabulary of phase 04 against the manual scene vocabulary.

Ground truth: workspace/<scene>/annotations/vocabulary.txt (scenepipe.annotate). Labels are
compared after synonyms (configs/tag_synonyms.yaml) and annotations/aliases.yaml.
Runs anywhere (numpy + yaml).

    python -m scenepipe.eval.tagging_recall --scene m360_counter
    python -m scenepipe.eval.tagging_recall --scene m360_counter --tags 04_tagging/_cmp/x/tags.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenepipe.annotate.store import AnnotationStore
from scenepipe.core.workspace import Workspace
from scenepipe.eval.gt import REPO_ROOT, LabelMatcher


def evaluate(ws: Workspace, tags_rel: str = "04_tagging/tags.json") -> dict:
    store = AnnotationStore(ws)
    gt = store.vocabulary()
    if not gt:
        raise SystemExit(f"{store.root / 'vocabulary.txt'} is missing or empty")
    match = LabelMatcher(store)
    tags = json.loads(ws.abs(tags_rel).read_text("utf-8"))["vocabulary"]
    pred = [t["tag"] for t in tags]
    found = [g for g in gt if any(match.same(g, p) for p in pred)]
    correct = [p for p in pred if any(match.same(g, p) for g in gt)]
    return {
        "tags": tags_rel,
        "num_gt": len(gt),
        "num_pred": len(pred),
        "recall": round(len(found) / len(gt), 4),
        "precision": round(len(correct) / len(pred), 4) if pred else 0.0,
        "missed": sorted(set(gt) - set(found)),
        "extra": sorted(set(pred) - set(correct)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", default=str(REPO_ROOT / "workspace"))
    ap.add_argument("--tags", default="04_tagging/tags.json")
    args = ap.parse_args()
    ws = Workspace.open(Path(args.workspace), args.scene)
    print(json.dumps(evaluate(ws, args.tags), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
