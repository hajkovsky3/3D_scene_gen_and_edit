"""Scene graph (`scene_graph.json`): format check shared by phase 08 (creates it) and the edit
phases that rewrite it. Format: phases/08_scene_graph.md, section Kontrakt.

Geometric relations (`origin: "geometric"`) are computed by 08; semantic ones come from an
LLM. `near` and `next_to` are symmetric and stored once. View-dependent relations (left_of,
above, ...) are never stored, phase 10 resolves them against a camera.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

GEOMETRIC_RELATIONS = ("on", "inside", "near")
SEMANTIC_RELATIONS = ("part_of", "supports", "belongs_to_set", "next_to")
RELATIONS = GEOMETRIC_RELATIONS + SEMANTIC_RELATIONS
ORIGINS = ("geometric", "semantic")


def load_scene_graph(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_scene_graph(doc: dict[str, Any], frame_ids: set[str] | None = None) -> None:
    """Raise ValueError if the graph breaks the format: node ids, geometry, edges, `on` cycles.
    `frame_ids` (registered frames), if given, must contain every best view."""
    if doc.get("up_axis") != "+z":
        raise ValueError(f"up_axis {doc.get('up_axis')!r}, expected '+z'")
    nodes = doc["nodes"]
    ids = [n["node_id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node_id")
    if len({n["instance_id"] for n in nodes}) != len(nodes):
        raise ValueError("duplicate instance_id")
    for n in nodes:
        nid = n["node_id"]
        if not (isinstance(nid, str) and nid.startswith("n") and nid[1:].isdigit()):
            raise ValueError(f"node_id {nid!r}, expected n0000")
        if n["label"] is not None and not isinstance(n["label"], str):
            raise ValueError(f"{nid}: label {n['label']!r}")
        obb = n["obb"]
        rot = np.asarray(obb["rotation"], np.float64)
        ext = np.asarray(obb["extent"], np.float64)
        if rot.shape != (3, 3) or not np.allclose(rot @ rot.T, np.eye(3), atol=1e-4):
            raise ValueError(f"{nid}: obb.rotation is not orthonormal")
        if np.linalg.det(rot) < 0:
            raise ValueError(f"{nid}: obb.rotation is a reflection")
        if ext.shape != (3,) or not (ext > 0).all() or len(obb["center"]) != 3:
            raise ValueError(f"{nid}: obb extent {ext.tolist()} must be 3 positive values")
        lo, hi = np.asarray(n["aabb"]["min"]), np.asarray(n["aabb"]["max"])
        if lo.shape != (3,) or hi.shape != (3,) or (lo > hi).any():
            raise ValueError(f"{nid}: aabb min > max")
        if n["num_gaussians"] <= 0 or n["feature_row"] < 0:
            raise ValueError(f"{nid}: num_gaussians / feature_row")
        if frame_ids is not None and not set(n["best_views"]) <= frame_ids:
            raise ValueError(f"{nid}: best_views not among the registered frames")
    known = set(ids)
    on: dict[str, list[str]] = {}
    seen = set()
    for e in doc["edges"]:
        s, t, rel = e["source"], e["target"], e["relation"]
        if s not in known or t not in known:
            raise ValueError(f"edge {s} -> {t}: unknown node")
        if s == t:
            raise ValueError(f"edge {s} -> {t}: self-loop")
        if rel not in RELATIONS:
            raise ValueError(f"edge {s} -> {t}: relation {rel!r} not in {RELATIONS}")
        if e["origin"] not in ORIGINS:
            raise ValueError(f"edge {s} -> {t}: origin {e['origin']!r}")
        if not 0 <= e["confidence"] <= 1:
            raise ValueError(f"edge {s} -> {t}: confidence {e['confidence']}")
        key = (s, t, rel) if rel not in ("near", "next_to") else (*sorted((s, t)), rel)
        if key in seen:
            raise ValueError(f"edge {s} -> {t} ({rel}) twice")
        seen.add(key)
        if rel == "on":
            on.setdefault(s, []).append(t)
    if _has_cycle(on):
        raise ValueError("relation 'on' has a cycle")


def _has_cycle(adj: dict[str, list[str]]) -> bool:
    state: dict[str, int] = {}  # 1 = on the stack, 2 = done

    def visit(u: str) -> bool:
        state[u] = 1
        for v in adj.get(u, []):
            if state.get(v) == 1 or (v not in state and visit(v)):
                return True
        state[u] = 2
        return False

    return any(u not in state and visit(u) for u in list(adj))
