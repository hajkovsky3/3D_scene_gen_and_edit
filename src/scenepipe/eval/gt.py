"""Shared helpers of the evaluations against the manual annotation (scenepipe.annotate).

Labels are compared after canonicalization with configs/tag_synonyms.yaml (as in 04) and with
the optional per-scene file annotations/aliases.yaml (`gt label: [accepted predicted labels]`),
which records judgements such as "oven mitt" == "glove" made by the annotator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from scenepipe.annotate.store import AnnotationStore

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNONYMS = REPO_ROOT / "configs" / "tag_synonyms.yaml"


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


class LabelMatcher:
    def __init__(self, store: AnnotationStore | None = None) -> None:
        self.syn: dict[str, str] = {}
        data = yaml.safe_load(SYNONYMS.read_text("utf-8")) if SYNONYMS.is_file() else {}
        for canon, aliases in (data or {}).items():
            for a in [canon, *(aliases or [])]:
                self.syn[_norm(a)] = _norm(canon)
        self.aliases: dict[str, set[str]] = {}
        path = store.root / "aliases.yaml" if store else None
        if path is not None and path.is_file():
            for gt, preds in (yaml.safe_load(path.read_text("utf-8")) or {}).items():
                self.aliases[self.canon(gt)] = {self.canon(p) for p in preds or []}

    def canon(self, label: str) -> str:
        n = _norm(label)
        return self.syn.get(n, n)

    def same(self, gt: str, pred: str) -> bool:
        g, p = self.canon(gt), self.canon(pred)
        return g == p or p in self.aliases.get(g, set())


def iou_matrix(gt: list[np.ndarray], pred: list[np.ndarray]) -> np.ndarray:
    """[len(gt), len(pred)] IoU of boolean masks of the same size."""
    out = np.zeros((len(gt), len(pred)), dtype=np.float64)
    if not gt or not pred:
        return out
    g = np.stack([m.ravel() for m in gt]).astype(np.float32)
    p = np.stack([m.ravel() for m in pred]).astype(np.float32)
    inter = g @ p.T
    union = g.sum(1)[:, None] + p.sum(1)[None, :] - inter
    np.divide(inter, union, out=out, where=union > 0)
    return out


def greedy_match(iou: np.ndarray, min_iou: float = 0.5) -> list[tuple[int, int, float]]:
    """One-to-one (gt, pred, iou) pairs, highest IoU first, IoU >= min_iou."""
    pairs = []
    used_g, used_p = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for gi, pi in order:
        v = float(iou[gi, pi])
        if v < min_iou:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((int(gi), int(pi), v))
    return pairs


def resize_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = (np.arange(height) * mask.shape[0] / height).astype(int)
    xs = (np.arange(width) * mask.shape[1] / width).astype(int)
    return mask[ys[:, None], xs[None, :]]
