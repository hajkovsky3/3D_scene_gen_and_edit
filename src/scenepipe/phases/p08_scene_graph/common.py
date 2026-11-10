"""Phase 08_scene_graph: shared mechanics of all implementations (node geometry, best views,
geometric relations, output, visualization, validation). Spec: phases/08_scene_graph.md

An implementation provides `describe()`: captions, attributes and label corrections of the
nodes (a VLM, or nothing in the baseline). Everything here is NumPy, runs on the CPU.

Conventions (also in the spec):
- obb: world point = center + rotation @ local, local in [-extent/2, extent/2]; the columns of
  `rotation` are the box axes in world coordinates. In gravity_aligned mode the third column
  is +Z and the first is the major horizontal direction.
- footprint: convex XY projection of the OBB (CCW polygon), used by `on`.
- distances are relative to the scene scale (core.cameras.scene_scale, ~ camera radius).
"""

from __future__ import annotations

import html
import json
import logging
import time
import zlib
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from scenepipe.core.cameras import load_cameras, scene_scale
from scenepipe.core.gaussians import read_ply
from scenepipe.core.phase import ContractError, Phase
from scenepipe.core.scene_graph import validate_scene_graph

log = logging.getLogger(__name__)


@dataclass
class NodeGeometry:
    points: np.ndarray  # float32[P, 3] subsample of the reliable gaussian centers
    center: np.ndarray  # float64[3]
    extent: np.ndarray  # float64[3] full side lengths
    rotation: np.ndarray  # float64[3, 3] columns = box axes
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    footprint: np.ndarray  # float64[k, 2] CCW
    bottom: float  # low percentile of the heights (thin legs survive, unlike in the aabb)
    num_gaussians: int

    @property
    def area(self) -> float:
        return polygon_area(self.footprint)

    @property
    def volume(self) -> float:
        return float(np.prod(self.extent))


class SceneGraphBase(Phase):
    phase = "08_scene_graph"
    contract_version = 2
    describe_stage: ClassVar[str | None] = None  # progress stage of describe(), if any
    describe_report: dict[str, Any]  # extra report.json fields set by describe()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.describe_report = {}

    defaults: ClassVar[dict[str, Any]] = {
        "obb_mode": "gravity_aligned",  # gravity_aligned | full
        "obb_trim_percent": 1.5,  # drop this percentile of points at both ends of every axis
        "min_opacity": 0.1,  # geometry only from gaussians at least this opaque
        "min_extent_rel": 0.002,  # lower bound of an OBB side (flat objects), x scene scale
        "max_points": 5000,  # subsample of gaussian centers per node for relations
        "num_best_views": 3,
        "best_view_min_gap": 5,  # best views at least this many processed frames apart
        "border_penalty": 0.5,  # view score factor of masks touching the image border
        "structure_labels": ["floor", "wall", "ceiling"],  # + `stuff` tags of 04
        "on_eps_rel": 0.02,  # |bottom of A - support height of B| < eps x scene scale
        "bottom_percentile": 0.5,  # bottom of an object (percentile of point heights)
        "on_overlap": 0.3,  # share of A's footprint over B's footprint
        "support_scale": 1.5,  # B's support surface is sought under A's footprint x this
        "inside_share": 0.6,  # share of A's points inside B's OBB
        "near_delta_rel": 0.05,  # point distance < delta x scene scale
        "near_points": 1000,  # subsample per node for the distance of `near`
        "semantic_relations": False,
    }

    def inputs(self) -> dict[str, Path]:
        return {
            "gaussians": self.ws.abs("03_refinement/gaussians.ply"),
            "gaussian_instances": self.ws.abs("07_lifting/gaussian_instances.npz"),
            "instances": self.ws.abs("07_lifting/instances.json"),
            "instance_features": self.ws.abs("07_lifting/instance_features.npy"),
            "cameras": self.ws.abs("01_sfm/cameras.json"),
            "detections": self.ws.abs("05_segmentation/detections.json"),
            "tags": self.ws.abs("04_tagging/tags.json"),
        }

    @abstractmethod
    def describe(self, nodes: list[dict[str, Any]], ctx: dict[str, Any]) -> None:
        """Fill `caption`, `attributes` of every node in place; may replace `label` (then the
        previous one goes to `label_original`). ctx: cameras, detections, geometry."""

    def steps(self) -> list[str]:
        desc = [self.describe_stage] if self.describe_stage else []
        return ["geometria", "najlepšie pohľady", *desc, "vzťahy", "zápis"]

    # --- run -------------------------------------------------------------------------------

    def run(self) -> dict[str, Path]:
        cfg = self.config
        t0 = time.perf_counter()
        g = read_ply(self.ws.abs("03_refinement/gaussians.ply"))
        inst_id = np.load(self.ws.abs("07_lifting/gaussian_instances.npz"))["instance_id"]
        doc07 = _read_json(self.ws.abs("07_lifting/instances.json"))
        feats = np.load(self.ws.abs("07_lifting/instance_features.npy"))
        cams = {c.frame_id: c for c in load_cameras(self.ws.abs("01_sfm/cameras.json"))}
        dets = _read_json(self.ws.abs("05_segmentation/detections.json"))
        structure = self._structure_labels()
        scale = scene_scale(list(cams.values()))
        insts = doc07["instances"]
        log.info("%d instances, %d gaussians, scene scale %.3f", len(insts), len(g), scale)

        # 1. geometry
        self.progress.update(0.0, "geometria", "")
        opacity = 1 / (1 + np.exp(-g.opacity.astype(np.float64)))
        order = np.argsort(inst_id, kind="stable")
        bounds = np.searchsorted(inst_id[order], np.arange(len(insts) + 1))
        rng = np.random.default_rng(0)
        geoms = []
        for k in range(len(insts)):
            idx = order[bounds[k] : bounds[k + 1]]
            pts = reliable_points(g.means[idx], opacity[idx], float(cfg["min_opacity"]))
            geoms.append(node_geometry(pts, len(idx), scale, cfg, rng))
        geometry_s = time.perf_counter() - t0

        # 2. best views
        self.progress.update(0.2, "najlepšie pohľady", "")
        by_mask = {d["mask_id"]: d for d in dets["detections"]}
        frame_pos = {f: i for i, f in enumerate(dets["frames"])}
        nodes = []
        for inst, geo in zip(insts, geoms, strict=True):
            masks = [by_mask[m] for m in inst["source_masks"] if m in by_mask]
            views = best_views(masks, frame_pos, cams, cfg)
            nodes.append(self._node(inst, geo, views, structure))

        # 3. captions, attributes
        t1 = time.perf_counter()
        if self.describe_stage:
            self.progress.update(0.3, self.describe_stage, "")
        ctx = {
            "cameras": cams,
            "detections": by_mask,
            "geometry": geoms,
            "instances": insts,
            "vocabulary": self._vocabulary(),
        }
        self.describe(nodes, ctx)
        for n in nodes:  # describe() may have corrected the label
            n["is_structure"] = n["label"] is not None and n["label"] in structure
        describe_s = time.perf_counter() - t1

        # 4. relations
        self.progress.update(0.8, "vzťahy", "")
        t2 = time.perf_counter()
        is_struct = [n["is_structure"] for n in nodes]
        edges = geometric_relations(geoms, is_struct, scale, cfg)
        if cfg["semantic_relations"]:
            edges += self.semantic_relations(nodes, edges)
        relations_s = time.perf_counter() - t2
        for e in edges:
            e["source"], e["target"] = nodes[e["source"]]["node_id"], nodes[e["target"]]["node_id"]

        self.progress.update(0.95, "zápis", "")
        outs = {
            "scene_graph": self.out_dir / "scene_graph.json",
            "node_features": self.out_dir / "node_features.npy",
            "graph": self.out_dir / "graph.html",
            "report": self.out_dir / "report.json",
        }
        doc = {
            "contract_version": self.contract_version,
            "up_axis": "+z",
            "embedding_model": doc07["embedding_model"],
            "scene_scale": round(scale, 6),
            "nodes": nodes,
            "edges": edges,
        }
        _write_json(outs["scene_graph"], doc)
        np.save(outs["node_features"], feats.astype(np.float16))
        outs["graph"].write_text(render_html(doc, geoms), encoding="utf-8")
        rel_counts = {r: sum(e["relation"] == r for e in edges) for r in sorted({
            e["relation"] for e in edges})}  # fmt: skip
        supported = {e["source"] for e in edges if e["relation"] in ("on", "inside")}
        _write_json(outs["report"], {
            "num_nodes": len(nodes),
            "num_structure": sum(is_struct),
            "num_unlabeled": sum(n["label"] is None for n in nodes),
            "num_edges": len(edges),
            "edges_by_relation": rel_counts,
            "unsupported_nodes": [
                n["node_id"] for n in nodes if not n["is_structure"] and n["node_id"] not in supported
            ],
            "scene_scale": scale,
            "on_eps": float(cfg["on_eps_rel"]) * scale,
            "near_delta": float(cfg["near_delta_rel"]) * scale,
            "geometry_s": round(geometry_s, 2),
            "describe_s": round(describe_s, 2),
            "relations_s": round(relations_s, 2),
            **self.describe_report,
        })  # fmt: skip
        log.info("%d nodes, %d edges %s", len(nodes), len(edges), rel_counts)
        return outs

    def semantic_relations(self, nodes: list[dict], edges: list[dict]) -> list[dict]:
        """Optional LLM relations (spec, step 5); not provided by every implementation."""
        raise NotImplementedError(f"{self.impl}: semantic_relations not supported")

    def _node(self, inst: dict, geo: NodeGeometry, views: list[str], structure: set[str]) -> dict:
        label = inst["label"]
        return {
            "node_id": f"n{inst['instance_id']:04d}",
            "instance_id": inst["instance_id"],
            "label": label,
            "label_original": None,
            "label_votes": inst["label_votes"],
            "caption": label or "",
            "attributes": {},
            "obb": {
                "center": _r(geo.center),
                "extent": _r(geo.extent),
                "rotation": [_r(row) for row in geo.rotation],
            },
            "aabb": {"min": _r(geo.aabb_min), "max": _r(geo.aabb_max)},
            "num_gaussians": geo.num_gaussians,
            "feature_row": inst["feature_row"],
            "best_views": views,
            "is_structure": label is not None and label in structure,
        }

    def _vocabulary(self) -> list[str]:
        tags = _read_json(self.ws.abs("04_tagging/tags.json"))
        return [v["tag"] for v in tags["vocabulary"]]

    def _structure_labels(self) -> set[str]:
        tags = _read_json(self.ws.abs("04_tagging/tags.json"))
        stuff = {v["tag"] for v in tags["vocabulary"] if v.get("kind") == "stuff"}
        return stuff | set(self.config["structure_labels"])

    # --- validation ------------------------------------------------------------------------

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        doc = _read_json(outputs["scene_graph"])
        if doc.get("contract_version") != self.contract_version:
            raise ContractError(f"scene_graph.json contract_version {doc.get('contract_version')}")
        frames = {c.frame_id for c in load_cameras(self.ws.abs("01_sfm/cameras.json"))}
        try:
            validate_scene_graph(doc, frames)
        except ValueError as e:
            raise ContractError(str(e)) from e
        doc07 = _read_json(self.ws.abs("07_lifting/instances.json"))
        if [n["instance_id"] for n in doc["nodes"]] != [
            i["instance_id"] for i in doc07["instances"]
        ]:
            raise ContractError("nodes must map 1:1 to the instances of 07, in order")
        if doc["embedding_model"] != doc07["embedding_model"]:
            raise ContractError("embedding_model differs from 07_lifting/instances.json")
        feat = np.load(outputs["node_features"])
        k = len(doc["nodes"])
        if feat.dtype != np.float16 or feat.ndim != 2:
            raise ContractError(f"node_features {feat.dtype}{list(feat.shape)}")
        if any(not 0 <= n["feature_row"] < feat.shape[0] for n in doc["nodes"]) or (
            feat.shape[0] != k
        ):
            raise ContractError(f"node_features has {feat.shape[0]} rows for {k} nodes")
        norms = np.linalg.norm(feat.astype(np.float32), axis=1)
        if norms.size and (np.abs(norms - 1) > 0.01).any():
            raise ContractError("node_features are not L2-normalized")


# --- geometry (unit-tested) -------------------------------------------------------------------


def reliable_points(means: np.ndarray, opacity: np.ndarray, min_opacity: float) -> np.ndarray:
    """Centers of gaussians at least min_opacity opaque; all of them if that leaves too few
    (under 20 points or 10 %)."""
    keep = opacity >= min_opacity
    if keep.sum() < max(20, 0.1 * len(means)):
        return means
    return means[keep]


def node_geometry(
    pts: np.ndarray, num_gaussians: int, scale: float, cfg: dict, rng: np.random.Generator
) -> NodeGeometry:
    pts = np.asarray(pts, np.float64)
    trim = float(cfg["obb_trim_percent"])
    min_ext = float(cfg["min_extent_rel"]) * scale
    center, extent, rot = oriented_box(pts, str(cfg["obb_mode"]), trim, min_ext)
    lo = np.percentile(pts, trim, axis=0)
    hi = np.percentile(pts, 100 - trim, axis=0)
    m = int(cfg["max_points"])
    sub = pts[rng.choice(len(pts), m, replace=False)] if len(pts) > m else pts
    return NodeGeometry(
        points=sub.astype(np.float32),
        center=center,
        extent=extent,
        rotation=rot,
        aabb_min=lo,
        aabb_max=hi,
        footprint=obb_footprint(center, extent, rot),
        bottom=float(np.percentile(pts[:, 2], float(cfg["bottom_percentile"]))),
        num_gaussians=num_gaussians,
    )


def oriented_box(
    pts: np.ndarray, mode: str, trim: float, min_extent: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(center, extent, rotation) of the box over the trimmed points. gravity_aligned: PCA in
    the XY plane for the heading, Z stays up; full: 3D PCA."""
    if mode == "gravity_aligned":
        rot = np.eye(3)
        if len(pts) >= 3:
            xy = pts[:, :2] - pts[:, :2].mean(0)
            _, vec = np.linalg.eigh(xy.T @ xy)
            theta = float(np.arctan2(vec[1, 1], vec[0, 1]))  # major axis (largest eigenvalue)
            if theta <= -np.pi / 2:  # the axis has no sign: keep the heading in (-90°, 90°]
                theta += np.pi
            elif theta > np.pi / 2:
                theta -= np.pi
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    elif mode == "full":
        rot = np.eye(3)
        if len(pts) >= 4:
            p = pts - pts.mean(0)
            _, vec = np.linalg.eigh(p.T @ p)
            rot = vec[:, ::-1].copy()  # major axis first
            for j in range(3):  # deterministic signs: largest component positive
                if rot[np.abs(rot[:, j]).argmax(), j] < 0:
                    rot[:, j] *= -1
            if np.linalg.det(rot) < 0:
                rot[:, 2] *= -1
    else:
        raise ValueError(f"obb_mode {mode!r}, expected gravity_aligned | full")
    local = pts @ rot
    lo = np.percentile(local, trim, axis=0)
    hi = np.percentile(local, 100 - trim, axis=0)
    return rot @ ((lo + hi) / 2), np.maximum(hi - lo, min_extent), rot


def obb_corners(center: np.ndarray, extent: np.ndarray, rot: np.ndarray) -> np.ndarray:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64)
    return center + (signs * extent / 2) @ rot.T


def obb_footprint(center: np.ndarray, extent: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return convex_hull(obb_corners(center, extent, rot)[:, :2])


def points_in_obb(pts: np.ndarray, center, extent, rot) -> np.ndarray:
    local = (pts - center) @ rot
    return (np.abs(local) <= np.asarray(extent) / 2).all(axis=1)


# --- 2D polygons ------------------------------------------------------------------------------


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """CCW convex hull (Andrew's monotone chain), without repeated points."""
    p = np.unique(np.round(np.asarray(pts, np.float64), 12), axis=0)
    if len(p) < 3:
        return p

    def half(points):
        out: list[np.ndarray] = []
        for q in points:
            while len(out) >= 2 and _cross(out[-2], out[-1], q) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Intersection of a polygon with a convex CCW polygon (Sutherland-Hodgman)."""
    out = [np.asarray(v, np.float64) for v in subject]
    n = len(clipper)
    for i in range(n):
        a, b = clipper[i], clipper[(i + 1) % n]
        inp, out = out, []
        if not inp:
            break
        prev = inp[-1]
        for cur in inp:
            cin, pin = _cross(a, b, cur) >= 0, _cross(a, b, prev) >= 0
            if cin:
                if not pin:
                    out.append(_intersect(prev, cur, a, b))
                out.append(cur)
            elif pin:
                out.append(_intersect(prev, cur, a, b))
            prev = cur
    return np.array(out).reshape(-1, 2)


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Points (XY) inside a convex CCW polygon (boundary included)."""
    inside = np.ones(len(pts), bool)
    if len(poly) < 3:
        return ~inside
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        cross = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
        inside &= cross >= -1e-12
    return inside


def _cross(o, a, b) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _intersect(p, q, a, b) -> np.ndarray:
    """Intersection of segment p-q with the line a-b."""
    d1, d2 = q - p, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-15:
        return np.asarray(q, np.float64)
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
    return p + t * d1


# --- best views -------------------------------------------------------------------------------


def best_views(masks: list[dict], frame_pos: dict[str, int], cams: dict, cfg: dict) -> list[str]:
    """Frames where the object is best visible: mask area share x mask_score x border penalty,
    at least best_view_min_gap processed frames apart (gap relaxed if too few frames)."""
    best: dict[str, float] = {}
    for d in masks:
        cam = cams.get(d["frame_id"])
        if cam is None:
            continue
        q = 1.0 if d["mask_score"] is None else float(d["mask_score"])
        if d["touches_border"]:
            q *= float(cfg["border_penalty"])
        s = d["area_px"] / (cam.width * cam.height) * q
        best[d["frame_id"]] = max(best.get(d["frame_id"], 0.0), s)
    ranked = sorted(best, key=lambda f: (-best[f], f))
    k, gap = int(cfg["num_best_views"]), int(cfg["best_view_min_gap"])
    picked: list[str] = []
    for f in ranked:
        if len(picked) < k and all(abs(frame_pos[f] - frame_pos[p]) >= gap for p in picked):
            picked.append(f)
    for f in ranked:
        if len(picked) < k and f not in picked:
            picked.append(f)
    return picked


# --- relations (unit-tested) ------------------------------------------------------------------


def geometric_relations(
    geoms: list[NodeGeometry], structure: list[bool], scale: float, cfg: dict
) -> list[dict[str, Any]]:
    """Edges on / inside / near between node indices (spec, step 4).

    - support(A, B), the condition of on: A is not structure. On a horizontal structure node
      (floor) only the height counts: B is a plane, its instance often covers only a part of
      the floor. Otherwise A has the smaller footprint or
      its bottom is above B's by more than eps (a plant wider than its pot), >= on_overlap of
      the smaller footprint over the other, and the bottom of A is within eps of B's support
      height: the 95th percentile of the heights of B's points under A's footprint (enlarged
      support_scale x, as the surface right under A is hidden) and below A's middle (the
      backrest of a sofa is not the seat a pillow lies on).
    - inside(A, B): >= inside_share of A's points in B's OBB, A smaller in volume, B is not
      structure, and B encloses A: B has points above A over part of its footprint (a box in a
      cabinet, a shelf) or in the upper half of A's heights (and above eps) on all four sides
      of it (a bowl, a sink, a drawer).
      Lying in B's box is not enough: a pillow on a sofa is on it, and sparse background
      instances have huge boxes. Only the smallest container is kept.
    - on(A, B): the highest support of A (a cup on a tray on a table is on the tray). A inside
      a container is on nothing but other nodes in the same container (a box in a cabinet is
      not on the floor below it). Cycles (possible only through the bottom rule) are broken
      at their weakest edge.
    - near(A, B): neither is structure, no on/inside between them, nearest point distance
      < delta; symmetric, stored once with source < target.
    """
    eps = float(cfg["on_eps_rel"]) * scale
    delta = float(cfg["near_delta_rel"]) * scale
    k = len(geoms)
    edges: list[dict[str, Any]] = []
    related: set[tuple[int, int]] = set()
    container: dict[int, int] = {}
    for a in range(k):
        if structure[a]:
            continue
        ga = geoms[a]
        best = None
        for b in range(k):
            gb = geoms[b]
            if b == a or structure[b] or gb.volume <= ga.volume:
                continue
            share = float(points_in_obb(ga.points, gb.center, gb.extent, gb.rotation).mean())
            if share < float(cfg["inside_share"]):
                continue
            if not (_covers(gb, ga, eps) or _surrounds(gb, ga, eps, delta)):
                continue
            if best is None or gb.volume < best[1]:
                best = (b, gb.volume, share)
        if best is not None:
            container[a] = best[0]
            edges.append(_edge(a, best[0], "inside", best[2]))
            related.add((a, best[0]))

    on: dict[int, tuple[int, float]] = {}
    for a in range(k):
        sup = _supports(a, geoms, structure, eps, cfg)
        # inside a container, A stands only on things inside the same container
        cands = {b: hc for b, hc in sup.items()
                 if a not in container or container.get(b) == container[a]}  # fmt: skip
        if cands:
            b = max(cands, key=lambda j: (round(cands[j][0], 9), cands[j][1]))
            on[a] = (b, cands[b][1])
    for a, (b, conf) in sorted(_break_cycles(on).items()):
        edges.append(_edge(a, b, "on", conf))
        related.add((a, b))

    m = int(cfg["near_points"])
    near_pts = [g.points[:: max(1, len(g.points) // m)][:m] for g in geoms]
    for a in range(k):
        for b in range(a + 1, k):
            if structure[a] or structure[b] or (a, b) in related or (b, a) in related:
                continue
            gap = np.maximum(geoms[a].aabb_min - geoms[b].aabb_max,
                             geoms[b].aabb_min - geoms[a].aabb_max).max()  # fmt: skip
            if gap >= delta:
                continue
            d = _min_distance(near_pts[a], near_pts[b])
            if d < delta:
                edges.append(_edge(a, b, "near", 1 - d / delta))
    return edges


def _supports(
    a: int, geoms: list[NodeGeometry], structure: list[bool], eps: float, cfg: dict
) -> dict[int, tuple[float, float]]:
    """{b: (support height, confidence)} of every node B that A stands on (see on)."""
    out: dict[int, tuple[float, float]] = {}
    ga = geoms[a]
    area_a = ga.area
    if structure[a] or area_a <= 0:
        return out
    support = _scale_polygon(ga.footprint, float(cfg["support_scale"]))
    zmid = (ga.bottom + ga.aabb_max[2]) / 2
    for b, gb in enumerate(geoms):
        if b == a or gb.area <= 0:
            continue
        if structure[b] and _horizontal(gb):  # floor: a plane, its instance is often partial
            h = float(np.quantile(gb.points[:, 2], 0.95))
            dz = abs(ga.bottom - h)
            if dz < eps:
                out[b] = (h, 1 - 0.5 * dz / eps)
            continue
        if not _boxes_overlap_xy(ga, gb):
            continue
        if not (gb.area > area_a or ga.bottom > gb.bottom + eps):
            continue
        overlap = polygon_area(clip_polygon(ga.footprint, gb.footprint)) / min(area_a, gb.area)
        if overlap < float(cfg["on_overlap"]):
            continue
        pts = gb.points
        under = points_in_polygon(pts[:, :2].astype(np.float64), support) & (pts[:, 2] < zmid)
        if not under.any():
            continue
        h = float(np.quantile(pts[under, 2], 0.95))
        dz = abs(ga.bottom - h)
        if dz < eps:
            out[b] = (h, min(overlap, 1.0) * (1 - 0.5 * dz / eps))
    return out


def _horizontal(g: NodeGeometry) -> bool:
    """Flat horizontal node (floor, carpet): height under a quarter of its XY size."""
    size = g.aabb_max - g.aabb_min
    return bool(size[2] < 0.25 * min(size[0], size[1]))


def _break_cycles(on: dict[int, tuple[int, float]]) -> dict[int, tuple[int, float]]:
    """Every node has at most one support; drop the weakest edge of each cycle."""
    on = dict(on)
    while True:
        cycle = None
        done: set[int] = set()
        for start in on:
            path: list[int] = []
            u = start
            while u in on and u not in done and u not in path:
                path.append(u)
                u = on[u][0]
            if u in path:
                cycle = path[path.index(u) :]
                break
            done.update(path)
        if cycle is None:
            return on
        del on[min(cycle, key=lambda v: (on[v][1], v))]


def _covers(b: NodeGeometry, a: NodeGeometry, eps: float, min_share: float = 0.25) -> bool:
    """B has points more than eps above A's top over at least min_share of A's footprint
    (4x4 grid); eps keeps the noise of a surface at the same height from covering A."""
    above = b.points[b.points[:, 2] > a.aabb_max[2] + eps]
    if not len(above):
        return False
    inside = points_in_polygon(above[:, :2].astype(np.float64), a.footprint)
    if not inside.any():
        return False
    lo, hi = a.footprint.min(0), a.footprint.max(0)
    cell = np.floor((above[inside, :2] - lo) / np.maximum(hi - lo, 1e-12) * 4).clip(0, 3)
    return len(np.unique(cell[:, 0] * 4 + cell[:, 1])) >= min_share * 16


def _surrounds(
    b: NodeGeometry, a: NodeGeometry, eps: float, delta: float, min_points: int = 3
) -> bool:
    """B has at least min_points points in the upper half of A's heights, and more than eps
    above A's bottom, on each of the four sides of A's footprint (axis-aligned box), within
    max(half its size, delta) of it: walls rising around A (a bowl), not the low front of a
    seat a pillow lies on, nor a table around a mat lying on it."""
    lo_z = max((a.bottom + a.aabb_max[2]) / 2, a.bottom + eps)
    p = b.points[(b.points[:, 2] >= lo_z) & (b.points[:, 2] <= a.aabb_max[2])]
    lo, hi = a.footprint.min(0), a.footprint.max(0)
    m = max(0.5 * float((hi - lo).max()), delta)
    x, y = p[:, 0], p[:, 1]
    in_x = (x >= lo[0]) & (x <= hi[0])
    in_y = (y >= lo[1]) & (y <= hi[1])
    sides = (
        in_y & (x > hi[0]) & (x <= hi[0] + m),
        in_y & (x < lo[0]) & (x >= lo[0] - m),
        in_x & (y > hi[1]) & (y <= hi[1] + m),
        in_x & (y < lo[1]) & (y >= lo[1] - m),
    )
    return all(int(s.sum()) >= min_points for s in sides)


def _edge(source: int, target: int, relation: str, confidence: float) -> dict[str, Any]:
    conf = round(float(np.clip(confidence, 0.0, 1.0)), 3)
    return {"source": source, "target": target, "relation": relation, "confidence": conf,
            "origin": "geometric"}  # fmt: skip


def _boxes_overlap_xy(a: NodeGeometry, b: NodeGeometry) -> bool:
    lo = np.maximum(a.footprint.min(0), b.footprint.min(0))
    hi = np.minimum(a.footprint.max(0), b.footprint.max(0))
    return bool((lo < hi).all())


def _scale_polygon(poly: np.ndarray, factor: float) -> np.ndarray:
    c = poly.mean(0)
    return c + factor * (poly - c)


def _min_distance(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> float:
    if not len(a) or not len(b):
        return float("inf")
    a, b = a.astype(np.float32), b.astype(np.float32)
    best = np.inf
    bb = (b**2).sum(1)
    for i in range(0, len(a), chunk):
        x = a[i : i + chunk]
        d2 = (x**2).sum(1)[:, None] + bb[None] - 2 * x @ b.T
        best = min(best, float(d2.min()))
    return float(np.sqrt(max(best, 0.0)))


# --- visualization ----------------------------------------------------------------------------


def render_html(doc: dict, geoms: list[NodeGeometry], size: int = 900) -> str:
    """Self-contained page: top view (XY) of the node footprints with the edges, and tables."""
    nodes = doc["nodes"]
    pad = 30
    if geoms:
        allp = np.concatenate([g.footprint for g in geoms])
        lo, hi = allp.min(0), allp.max(0)
    else:
        lo, hi = np.zeros(2), np.ones(2)
    s = (size - 2 * pad) / max(float((hi - lo).max()), 1e-9)

    def xy(p) -> tuple[float, float]:
        return pad + (p[0] - lo[0]) * s, size - pad - (p[1] - lo[1]) * s  # +Y up

    order = sorted(range(len(nodes)), key=lambda i: (not nodes[i]["is_structure"], -geoms[i].area))
    parts = []
    for i in order:
        n, g = nodes[i], geoms[i]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(p) for p in g.footprint))
        cls = "structure" if n["is_structure"] else "node"
        color = "none" if n["is_structure"] else _color(n["label"] or n["node_id"])
        title = html.escape(f"{n['node_id']} {n['label']} ({n['num_gaussians']} gaussiánov)")
        parts.append(f'<polygon class="{cls}" points="{pts}" style="--c:{color}">'
                     f"<title>{title}</title></polygon>")  # fmt: skip
    centers = {n["node_id"]: xy(g.footprint.mean(0)) for n, g in zip(nodes, geoms, strict=True)}
    for e in doc["edges"]:
        (x0, y0), (x1, y1) = centers[e["source"]], centers[e["target"]]
        parts.append(f'<line class="e-{e["relation"]}" x1="{x0:.1f}" y1="{y0:.1f}" '
                     f'x2="{x1:.1f}" y2="{y1:.1f}"/>')  # fmt: skip
    for n in nodes:
        if n["is_structure"]:
            continue
        x, y = centers[n["node_id"]]
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}">{html.escape(n["label"] or "?")}</text>')
    labels = {n["node_id"]: n["label"] for n in nodes}
    node_rows = "".join(
        f"<tr><td>{n['node_id']}</td><td>{html.escape(str(n['label']))}"
        f"{_was(n['label_original'])}</td>"
        f"<td>{html.escape(n['caption'])}</td>"
        f"<td>{html.escape(', '.join(f'{k}: {v}' for k, v in n['attributes'].items()))}</td>"
        f"<td>{n['num_gaussians']}</td>"
        f"<td>{' × '.join(f'{v:.2f}' for v in n['obb']['extent'])}</td>"
        f"<td>{'áno' if n['is_structure'] else ''}</td><td>{' '.join(n['best_views'])}</td></tr>"
        for n in nodes
    )
    edge_rows = "".join(
        f"<tr><td>{e['source']} {html.escape(str(labels[e['source']]))}</td>"
        f"<td>{e['relation']}</td><td>{e['target']} {html.escape(str(labels[e['target']]))}</td>"
        f"<td>{e['confidence']:.2f}</td><td>{e['origin']}</td></tr>"
        for e in doc["edges"]
    )
    return _PAGE.format(
        size=size,
        svg="\n".join(parts),
        nodes=node_rows,
        edges=edge_rows,
        num_nodes=len(nodes),
        num_edges=len(doc["edges"]),
    )


_PAGE = """<!doctype html>
<html lang="sk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Graf scény</title>
<style>
:root {{ --bg: #fbfbfa; --fg: #1d1d1b; --muted: #8a8a85; --line: #d9d9d4; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #1b1b1a; --fg: #ecece8; --muted: #8f8f8a; --line: #3a3a37; }}
}}
body {{ background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif;
  margin: 0 auto; max-width: {size}px; padding: 16px; }}
svg {{ width: 100%; height: auto; border: 1px solid var(--line); }}
polygon.node {{ fill: var(--c); fill-opacity: 0.35; stroke: var(--c); stroke-width: 1.5; }}
polygon.structure {{ fill: none; stroke: var(--muted); stroke-dasharray: 4 3; }}
line {{ stroke: var(--fg); stroke-width: 1.2; }}
line.e-on {{ stroke-width: 2; }}
line.e-inside {{ stroke-dasharray: 6 3; }}
line.e-near {{ stroke: var(--muted); stroke-dasharray: 2 3; }}
text {{ fill: var(--fg); font-size: 11px; text-anchor: middle; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
td, th {{ border-bottom: 1px solid var(--line); padding: 3px 6px; text-align: left; }}
.legend, .was {{ color: var(--muted); }}
</style></head><body>
<h1>Graf scény</h1>
<p class="legend">Pohľad zhora (rovina XY), obdĺžniky sú pôdorysy OBB. Hrany: plná hrubá = on,
čiarkovaná = inside, bodkovaná = near. Prerušovaný sivý obrys = štruktúra (podlaha, stena).</p>
<svg viewBox="0 0 {size} {size}">
{svg}
</svg>
<h2>Uzly ({num_nodes})</h2>
<table><tr><th>uzol</th><th>label</th><th>popis</th><th>atribúty</th><th>gaussiány</th><th>rozmer OBB</th>
<th>štruktúra</th><th>najlepšie pohľady</th></tr>{nodes}</table>
<h2>Hrany ({num_edges})</h2>
<table><tr><th>zdroj</th><th>vzťah</th><th>cieľ</th><th>istota</th><th>pôvod</th></tr>{edges}</table>
</body></html>
"""


def _was(label_original: str | None) -> str:
    """Note of a label corrected by the VLM."""
    if label_original is None:
        return ""
    return f' <span class="was">(07: {html.escape(label_original)})</span>'


def _color(key: str) -> str:
    h = zlib.crc32(key.encode()) % 360
    return f"hsl({h} 60% 50%)"


def _r(v) -> list[float]:
    return [round(float(x), 6) for x in v]


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
