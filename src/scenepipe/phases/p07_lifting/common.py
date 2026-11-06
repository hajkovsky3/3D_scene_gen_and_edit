"""Phase 07_lifting: shared mechanics of all implementations (candidates from tracks or masks,
merging, per-gaussian voting, 3D cleanup, labels and features, output, validation).
Spec: phases/07_lifting.md

An implementation provides `frame_weights()`: for every frame the total weight with which
each gaussian contributes to the image (visibility) and to each mask (W_i(m)). Everything
else here is NumPy/SciPy, so the phase logic is testable without a GPU.

Terms used below:
- candidate: a group of masks believed to show one object (a track, or a single mask without
  tracks); merging joins candidates into instances,
- frac_i(m) = W_i(m) / vis_i(frame): share of gaussian i's visible weight in the frame that
  falls into mask m (1 = fully inside),
- coverage_i(c) = sum_m W_i(m) / sum_f vis_i(f) over the frames where c has a mask: how
  consistently gaussian i lies inside c's masks when c is observed.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import zlib
from abc import abstractmethod
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from scenepipe.core.cameras import Camera, load_cameras
from scenepipe.core.gaussians import Gaussians, read_ply
from scenepipe.core.masks import rle_decode
from scenepipe.core.phase import ContractError, Phase
from scenepipe.core.progress import StageProgress

log = logging.getLogger(__name__)

DEBUG_MAX_SIDE = 960


@dataclass(frozen=True)
class FrameMasks:
    """Input of frame_weights(): one frame with its masks (decoded, at camera resolution)."""

    camera: Camera
    mask_ids: list[str]
    masks: list[np.ndarray]  # bool[H, W] each


@dataclass
class FrameWeights:
    """Output of frame_weights() for one frame: sparse (gaussian index, weight) pairs."""

    vis_idx: np.ndarray  # int64[V] gaussians visible in the frame
    vis_w: np.ndarray  # float32[V] sum over all pixels of alpha * transmittance
    mask_idx: list[np.ndarray]  # per mask int64[...]
    mask_w: list[np.ndarray]  # per mask float32[...] W_i(m)


@dataclass
class Candidate:
    """Accumulated votes of a group of masks (sparse over gaussians)."""

    masks: list[str]
    frames: set[str]
    tracks: set[str]
    idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    w: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # sum W
    wq: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # sum W * q
    frac: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # max frac

    def add(self, idx, w, wq, frac) -> None:
        self.idx, (self.w, self.wq), self.frac = merge_sparse(
            [self.idx, idx], [[self.w, w], [self.wq, wq]], [self.frac, frac]
        )

    def absorb(self, other: Candidate) -> None:
        self.masks += other.masks
        self.frames |= other.frames
        self.tracks |= other.tracks
        self.add(other.idx, other.w, other.wq, other.frac)


class LiftingBase(Phase):
    phase = "07_lifting"
    contract_version = 2
    defaults: ClassVar[dict[str, Any]] = {
        "min_weight": 0.05,  # keep W_i(m) only where frac_i(m) >= this (sparsity)
        "support_fraction": 0.5,  # frac at which a gaussian counts as inside a mask (merging)
        "border_penalty": 0.5,  # quality factor of masks touching the image border
        "merge_overlap": 0.3,
        "merge_sim": 0.75,
        "merge_rounds": 5,
        "duplicate_overlap": 0.7,  # merge even candidates seen in one frame (duplicate tracks)
        "min_confidence": 0.5,
        "min_coverage": 0.2,
        "prefer_smaller_on_overlap": True,
        "outlier": "components",  # components | none
        "outlier_k": 3.0,  # link radius = k x the outlier_quantile of nearest-neighbor distances
        "outlier_quantile": 0.9,  # high, so sparse parts of an object stay linked
        "outlier_min_share": 0.05,  # drop components smaller than this share of the instance
        "min_gaussians": 200,
        "min_views": 3,  # instances seen in fewer frames do not vote (fragments)
        "fragment_attach_share": 0.5,  # a fragment's masks join the instance with this vote share
        "zero_shot_weight": 0.5,  # weight of the 06 zero-shot label scores in label_votes
        "feature_min_sim": 0.6,  # drop mask embeddings less similar to the first mean
        "num_debug": 6,
    }

    def inputs(self) -> dict[str, Path]:
        return {
            "gaussians": self.ws.abs("03_refinement/gaussians.ply"),
            "cameras": self.ws.abs("01_sfm/cameras.json"),
            "detections": self.ws.abs("05_segmentation/detections.json"),
            "masks": self.ws.abs("05_segmentation/masks"),
            "embeddings": self.ws.abs("06_mask_embedding/embeddings.npy"),
            "embedding_index": self.ws.abs("06_mask_embedding/index.json"),
            "embedding_meta": self.ws.abs("06_mask_embedding/meta.json"),
        }

    @abstractmethod
    def frame_weights(
        self, g: Gaussians, frames: list[FrameMasks], stage: StageProgress
    ) -> Iterator[FrameWeights]:
        """Yield the weights of every frame, in the order of `frames`."""

    def render_instances(self, g: Gaussians, colors: np.ndarray, cam: Camera) -> np.ndarray:
        """uint8[H, W, 3] render of per-gaussian colors (debug only); optional."""
        raise NotImplementedError

    def steps(self) -> list[str]:
        debug = ["debug rendery"] if int(self.config["num_debug"]) > 0 else []
        return ["váhy masiek", "spájanie", "hlasovanie", *debug, "zápis"]

    # --- run -------------------------------------------------------------------------------

    def run(self) -> dict[str, Path]:
        cfg = self.config
        g = read_ply(self.ws.abs("03_refinement/gaussians.ply"))
        n = len(g)
        doc = self._detections()
        dets = {d["mask_id"]: d for d in doc["detections"]}
        cams = {c.frame_id: c for c in load_cameras(self.ws.abs("01_sfm/cameras.json"))}
        emb, emb_row, meta = self._embeddings()
        log.info("%d gaussians, %d frames, %d masks", n, len(doc["frames"]), len(dets))

        # 1. weights: candidates (tracks or single masks) and per-frame visibility
        t0 = time.perf_counter()
        stage = self.progress.stage(0.0, 0.7, "váhy masiek")
        cands: dict[str, Candidate] = {}
        mask_mass: dict[str, float] = {}
        frame_vis: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        visibility = np.zeros(n, np.float32)
        frames = list(self._frame_masks(doc, cams))
        min_w = float(cfg["min_weight"])
        for fm, fw in zip(frames, self.frame_weights(g, frames, stage), strict=True):
            fid = fm.camera.frame_id
            np.add.at(visibility, fw.vis_idx, fw.vis_w)
            frame_vis[fid] = (fw.vis_idx.astype(np.int32), fw.vis_w.astype(np.float16))
            vis = np.zeros(n, np.float32)
            vis[fw.vis_idx] = fw.vis_w
            for mask_id, idx, w in zip(fm.mask_ids, fw.mask_idx, fw.mask_w, strict=True):
                d = dets[mask_id]
                mask_mass[mask_id] = float(w.sum())
                frac = w / np.maximum(vis[idx], 1e-12)
                keep = frac >= min_w
                key = d["track_id"] or mask_id
                c = cands.setdefault(key, Candidate([], set(), set()))
                c.masks.append(mask_id)
                c.frames.add(fid)
                if d["track_id"]:
                    c.tracks.add(d["track_id"])
                q = self._quality(d)
                c.add(idx[keep], w[keep], w[keep] * q, frac[keep].astype(np.float32))
        weights_s = time.perf_counter() - t0
        log.info("%d candidates from %d masks", len(cands), len(dets))

        # 2. merging candidates into instances
        self.progress.update(0.7, "spájanie", "")
        feats = {k: self._feature(c.masks, mask_mass, dets, emb, emb_row) for k, c in cands.items()}
        insts = merge_candidates(list(cands.values()), [feats[k] for k in cands], n, cfg)
        log.info("%d instances after merging", len(insts))

        # 3. voting, 3D cleanup, filtering. Instances seen in fewer than min_views frames do
        # not vote: their coverage comes from 1-2 frames (trivially high) and as small ones they
        # would win over the real object, then be dropped and leave its gaussians unassigned.
        self.progress.update(0.8, "hlasovanie", "")
        min_views = int(cfg["min_views"])
        fragments = [c for c in insts if len(c.frames) < min_views]
        insts = [c for c in insts if len(c.frames) >= min_views]
        inst_id, conf = vote(insts, frame_vis, n, cfg)
        if cfg["outlier"] == "components":
            inst_id = remove_outliers(inst_id, g.means, cfg)
        elif cfg["outlier"] != "none":
            raise ValueError(f"outlier {cfg['outlier']!r}, expected components | none")
        counts = np.bincount(inst_id[inst_id >= 0], minlength=len(insts))
        keep = [k for k in range(len(insts)) if counts[k] >= int(cfg["min_gaussians"])]
        keep.sort(key=lambda k: -counts[k])  # instance 0 = the largest
        remap = np.full(len(insts) + 1, -1, np.int32)  # last entry maps -1
        remap[keep] = np.arange(len(keep), dtype=np.int32)
        inst_id = remap[inst_id]
        conf = np.where(inst_id >= 0, conf, 0).astype(np.float16)
        absorbed = 0
        for frag, new in zip(fragments, attach_fragments(fragments, inst_id, cfg), strict=True):
            if new >= 0:  # its masks describe that instance (labels, features, sources)
                c = insts[keep[new]]
                c.masks += frag.masks
                c.frames |= frag.frames
                c.tracks |= frag.tracks
                absorbed += 1
        log.info("%d fragments below min_views, %d attached to instances", len(fragments), absorbed)

        # 4. labels and features
        label_scores = self._label_scores()
        instances, features = [], []
        for new, k in enumerate(keep):
            c = insts[k]
            votes = label_votes(c.masks, mask_mass, dets, label_scores, cfg)
            instances.append({
                "instance_id": new,
                "label": max(votes, key=votes.get) if votes else None,
                "label_votes": votes,
                "num_gaussians": int(counts[k]),
                "num_views": len(c.frames),
                "source_masks": sorted(c.masks),
                "track_ids": sorted(c.tracks),
                "feature_row": new,
            })  # fmt: skip
            features.append(self._feature(c.masks, mask_mass, dets, emb, emb_row, robust=True))
        dim = emb.shape[1]
        feat = np.stack(features) if features else np.zeros((0, dim), np.float32)

        if int(cfg["num_debug"]) > 0:
            self.progress.update(0.9, "debug rendery", "")
            self._write_debug(g, inst_id, frames)

        self.progress.update(0.97, "zápis", "")
        outs = {
            "gaussian_instances": self.out_dir / "gaussian_instances.npz",
            "instances": self.out_dir / "instances.json",
            "instance_features": self.out_dir / "instance_features.npy",
            "report": self.out_dir / "report.json",
        }
        np.savez_compressed(
            outs["gaussian_instances"], instance_id=inst_id, confidence=conf, visibility=visibility
        )
        _write_json(outs["instances"], {
            "contract_version": self.contract_version,
            "embedding_model": meta["model"],
            "instances": instances,
        })  # fmt: skip
        np.save(outs["instance_features"], feat.astype(np.float16))
        assigned = int((inst_id >= 0).sum())
        _write_json(outs["report"], {
            "num_gaussians": n, "num_frames": len(frames), "num_masks": len(dets),
            "num_candidates": len(cands), "num_merged": len(insts) + len(fragments),
            "num_fragments": len(fragments), "num_fragments_attached": absorbed,
            "num_instances": len(instances),
            "assigned_share": round(assigned / max(n, 1), 4),
            "unobserved_share": round(float((visibility <= 0).mean()) if n else 0.0, 4),
            "labels": dict(Counter(i["label"] for i in instances).most_common()),
            "weights_s": round(weights_s, 1),
        })  # fmt: skip
        if int(cfg["num_debug"]) > 0:
            outs["debug"] = self.out_dir / "debug"
        return outs

    def _frame_masks(self, doc: dict, cams: dict[str, Camera]) -> Iterator[FrameMasks]:
        by_frame: dict[str, list[str]] = {}
        for d in doc["detections"]:
            by_frame.setdefault(d["frame_id"], []).append(d["mask_id"])
        for fid in doc["frames"]:
            ids = by_frame.get(fid, [])
            path = self.ws.abs(f"05_segmentation/masks/{fid}.json")
            rles = {r["mask_id"]: r["rle"] for r in json.loads(path.read_text("utf-8"))}
            yield FrameMasks(cams[fid], ids, [rle_decode(rles[m]) for m in ids])

    def _quality(self, d: dict) -> float:
        q = 1.0 if d["mask_score"] is None else float(d["mask_score"])
        return q * (float(self.config["border_penalty"]) if d["touches_border"] else 1.0)

    def _feature(self, masks, mass, dets, emb, emb_row, robust: bool = False) -> np.ndarray:
        """Weighted mean of the mask embeddings (weight = mask mass x quality), L2-normalized;
        robust: a second pass without embeddings far from the first mean."""
        rows = [(emb_row[m], mass[m] * self._quality(dets[m])) for m in masks if m in emb_row]
        if not rows:
            return np.zeros(emb.shape[1], np.float32)
        e = emb[[r for r, _ in rows]].astype(np.float32)
        w = np.array([max(x, 1e-6) for _, x in rows], np.float32)
        mean = _normalize(w @ e)
        if robust:
            near = e @ mean >= float(self.config["feature_min_sim"])
            if near.any():
                mean = _normalize(w[near] @ e[near])
        return mean

    def _embeddings(self) -> tuple[np.ndarray, dict[str, int], dict]:
        emb = np.load(self.ws.abs("06_mask_embedding/embeddings.npy"))
        index = json.loads(self.ws.abs("06_mask_embedding/index.json").read_text("utf-8"))
        meta = json.loads(self.ws.abs("06_mask_embedding/meta.json").read_text("utf-8"))
        return emb, {m: i for i, m in enumerate(index["mask_ids"])}, meta

    def _label_scores(self) -> dict[str, dict[str, float]]:
        path = self.ws.abs("06_mask_embedding/label_scores.json")  # optional output of 06
        if not path.is_file():
            return {}
        return {k: v["probs"] for k, v in json.loads(path.read_text("utf-8"))["scores"].items()}

    def _detections(self) -> dict:
        return json.loads(self.ws.abs("05_segmentation/detections.json").read_text("utf-8"))

    def _write_debug(self, g: Gaussians, inst_id: np.ndarray, frames: list[FrameMasks]) -> None:
        """Side by side: frame | its 2D masks | rendered instance ids (same color per instance
        in all views)."""
        out = self.out_dir / "debug"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        colors = np.zeros((len(inst_id), 3), np.uint8)
        ids = np.unique(inst_id[inst_id >= 0])
        palette = np.array([_color(f"i{i}") for i in ids], np.uint8).reshape(-1, 3)
        lut = np.zeros(int(inst_id.max(initial=-1)) + 2, np.int64)
        lut[ids] = np.arange(len(ids))
        assigned = inst_id >= 0
        colors[assigned] = palette[lut[inst_id[assigned]]]
        k = int(self.config["num_debug"])
        picks = np.unique(np.round(np.linspace(0, len(frames) - 1, min(k, len(frames)))))
        for i in picks.astype(int):
            fm = frames[i]
            with Image.open(self.ws.abs(fm.camera.image)) as im:
                img = np.asarray(im.convert("RGB"))
            masks = np.zeros_like(img)
            for mask_id, m in sorted(zip(fm.mask_ids, fm.masks), key=lambda t: -t[1].sum()):
                masks[m] = _color(mask_id)
            ren = self.render_instances(g, colors, fm.camera)
            tile = Image.fromarray(np.concatenate([img, masks, ren], axis=1))
            s = DEBUG_MAX_SIDE / max(img.shape[:2])
            if s < 1:
                tile = tile.resize((round(tile.width * s), round(tile.height * s)))
            tile.save(out / f"{fm.camera.frame_id}.jpg", quality=88)

    # --- validation ------------------------------------------------------------------------

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        n = len(read_ply(self.ws.abs("03_refinement/gaussians.ply")))
        npz = np.load(outputs["gaussian_instances"])
        inst_id, conf, vis = npz["instance_id"], npz["confidence"], npz["visibility"]
        if inst_id.dtype != np.int32 or inst_id.shape != (n,):
            raise ContractError(f"instance_id {inst_id.dtype}{list(inst_id.shape)}, N = {n}")
        if conf.dtype != np.float16 or conf.shape != (n,):
            raise ContractError(f"confidence {conf.dtype}{list(conf.shape)}")
        if vis.dtype != np.float32 or vis.shape != (n,) or (vis < 0).any():
            raise ContractError(f"visibility {vis.dtype}{list(vis.shape)} or negative")
        doc = json.loads(outputs["instances"].read_text("utf-8"))
        if doc.get("contract_version") != self.contract_version:
            raise ContractError(f"instances.json contract_version {doc.get('contract_version')}")
        insts = doc["instances"]
        k = len(insts)
        if inst_id.min(initial=-1) < -1 or inst_id.max(initial=-1) >= k:
            raise ContractError(f"instance_id outside {{-1}} u [0, {k})")
        counts = np.bincount(inst_id[inst_id >= 0], minlength=k)
        mask_ids = {d["mask_id"] for d in self._detections()["detections"]}
        for i, inst in enumerate(insts):
            if inst["instance_id"] != i or inst["feature_row"] != i:
                raise ContractError(f"instance {i}: ids must be 0..K-1 in order")
            if inst["num_gaussians"] != counts[i] or counts[i] < int(self.config["min_gaussians"]):
                raise ContractError(f"instance {i}: {counts[i]} gaussians")
            if not set(inst["source_masks"]) <= mask_ids:
                raise ContractError(f"instance {i}: unknown source masks")
        feat = np.load(outputs["instance_features"])
        if feat.dtype != np.float16 or feat.shape[0] != k:
            raise ContractError(f"instance_features {feat.dtype}{list(feat.shape)}, K = {k}")
        norms = np.linalg.norm(feat.astype(np.float32), axis=1)
        if norms.size and (np.abs(norms - 1) > 0.01).any():
            raise ContractError("instance_features are not L2-normalized")
        meta = json.loads(self.ws.abs("06_mask_embedding/meta.json").read_text("utf-8"))
        if doc["embedding_model"] != meta["model"] or feat.shape[1] != meta["dim"]:
            raise ContractError("embedding_model/dim differ from 06_mask_embedding/meta.json")


# --- algorithm (unit-tested) ------------------------------------------------------------------


def merge_sparse(idx_list, sums_list, max_list):
    """Union of sparse vectors over idx_list: the parts of each quantity in sums_list are added,
    the parts in max_list combined by maximum."""
    idx = np.concatenate(idx_list)
    uniq, inv = np.unique(idx, return_inverse=True)
    sums = [
        np.bincount(inv, np.concatenate(parts), len(uniq)).astype(np.float32) for parts in sums_list
    ]
    mx = np.zeros(len(uniq), np.float32)
    np.maximum.at(mx, inv, np.concatenate(max_list))
    return uniq, sums, mx


def merge_candidates(
    cands: list[Candidate], feats: list[np.ndarray], n: int, cfg: dict
) -> list[Candidate]:
    """Greedy rounds of merging: two candidates join if their gaussian supports overlap both
    ways (weighted share of each one's votes on the other's support > merge_overlap), their
    features are similar (> merge_sim) and they never appear as different masks in one frame,
    unless the overlap exceeds duplicate_overlap: the video tracker sometimes starts a second
    track on an object it already tracks, and such duplicates lie on the same gaussians."""
    from scipy import sparse

    cands = list(cands)
    feats = list(feats)
    for _ in range(int(cfg["merge_rounds"])):
        k = len(cands)
        if k < 2:
            break
        rows = np.concatenate([np.full(len(c.idx), i) for i, c in enumerate(cands)])
        cols = np.concatenate([c.idx for c in cands])
        share = np.concatenate([c.wq / max(c.wq.sum(), 1e-12) for c in cands])
        inside = np.concatenate([c.frac >= float(cfg["support_fraction"]) for c in cands])
        a = sparse.csr_matrix((share, (rows, cols)), shape=(k, n))
        s = sparse.csr_matrix((inside.astype(np.float32), (rows, cols)), shape=(k, n))
        on_other = (a @ s.T).toarray()  # [a, b]: share of a's votes on b's support
        overlap = np.minimum(on_other, on_other.T)
        sim = np.stack(feats) @ np.stack(feats).T
        frame_ids = {f: j for j, f in enumerate(sorted(set().union(*(c.frames for c in cands))))}
        fr = sparse.csr_matrix(
            (
                np.ones(sum(len(c.frames) for c in cands)),
                (
                    np.concatenate([np.full(len(c.frames), i) for i, c in enumerate(cands)]),
                    [frame_ids[f] for c in cands for f in sorted(c.frames)],
                ),
            ),
            shape=(k, len(frame_ids)),
        )
        together = (fr @ fr.T).toarray() > 0
        ok = (overlap > float(cfg["merge_overlap"])) & (sim > float(cfg["merge_sim"]))
        dup = overlap > float(cfg["duplicate_overlap"])
        ok &= ~together | dup
        np.fill_diagonal(ok, False)
        if not ok.any():
            break
        # union-find over all qualifying pairs, strongest first
        parent = list(range(k))
        members: dict[int, set[str]] = {i: set(c.frames) for i, c in enumerate(cands)}
        pa, pb = np.nonzero(np.triu(ok))
        for i in np.argsort(-overlap[pa, pb]):
            ra, rb = _root(parent, pa[i]), _root(parent, pb[i])
            if ra != rb and (dup[pa[i], pb[i]] or not members[ra] & members[rb]):
                parent[rb] = ra
                members[ra] |= members.pop(rb)
        groups: dict[int, list[int]] = {}
        for i in range(k):
            groups.setdefault(_root(parent, i), []).append(i)
        merged, mfeats = [], []
        for idxs in groups.values():
            c = cands[idxs[0]]
            for j in idxs[1:]:
                c.absorb(cands[j])
            w = np.array([max(cands[j].wq.sum(), 1e-6) for j in idxs], np.float32)
            merged.append(c)
            mfeats.append(_normalize(w @ np.stack([feats[j] for j in idxs])))
        cands, feats = merged, mfeats
    return cands


def _root(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def vote(
    insts: list[Candidate],
    frame_vis: dict[str, tuple[np.ndarray, np.ndarray]],
    n: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Instance id (-1 = none) and confidence of every gaussian.

    A gaussian is eligible for an instance if its coverage >= min_coverage. The winner is the
    eligible instance with the highest coverage, or with prefer_smaller_on_overlap the smallest
    eligible one (a pillow beats the sofa whose masks include it); larger eligible instances
    are then containers, not competitors. confidence = winner score / sum of the scores of the
    competitors (incl. the winner); it must reach min_confidence.
    """
    rows, cols, score, cov = [], [], [], []
    for k, c in enumerate(insts):
        denom = np.zeros(n, np.float32)
        for f in c.frames:
            idx, w = frame_vis[f]
            denom[idx] += w.astype(np.float32)
        rows.append(np.full(len(c.idx), k))
        cols.append(c.idx)
        score.append(c.wq)
        cov.append(c.w / np.maximum(denom[c.idx], 1e-12))
    inst_id = np.full(n, -1, np.int32)
    conf = np.zeros(n, np.float32)
    if not insts or not sum(len(r) for r in rows):
        return inst_id, conf
    inst = np.concatenate(rows)
    gid = np.concatenate(cols)
    score = np.concatenate(score)
    cov = np.minimum(np.concatenate(cov), 1.0)
    size = np.array([np.count_nonzero(c.frac >= float(cfg["support_fraction"])) for c in insts])
    eligible = cov >= float(cfg["min_coverage"])
    smaller = bool(cfg["prefer_smaller_on_overlap"])
    # winner per gaussian: first eligible entry in the order (gaussian, preference)
    pref = size[inst].astype(np.float64) if smaller else -cov
    order = np.lexsort((pref, ~eligible, gid))
    first = np.ones(len(order), bool)
    first[1:] = gid[order][1:] != gid[order][:-1]
    win = order[first & eligible[order]]
    win_size = np.full(n, np.inf)
    win_size[gid[win]] = size[inst[win]]
    competitor = ~(eligible & (size[inst] > win_size[gid])) if smaller else np.ones_like(eligible)
    total = np.bincount(gid, score * competitor, minlength=n)
    c = score[win] / np.maximum(total[gid[win]], 1e-12)
    ok = c >= float(cfg["min_confidence"])
    inst_id[gid[win][ok]] = inst[win][ok]
    conf[gid[win][ok]] = c[ok]
    return inst_id, conf


def attach_fragments(fragments: list[Candidate], inst_id: np.ndarray, cfg: dict) -> list[int]:
    """For every fragment the final instance holding at least fragment_attach_share of its
    votes, or -1."""
    out = []
    for c in fragments:
        ids = inst_id[c.idx]
        on = ids >= 0
        total = float(c.wq.sum())
        if not on.any() or total <= 0:
            out.append(-1)
            continue
        mass = np.bincount(ids[on], c.wq[on])
        best = int(mass.argmax())
        out.append(best if mass[best] / total >= float(cfg["fragment_attach_share"]) else -1)
    return out


def remove_outliers(inst_id: np.ndarray, means: np.ndarray, cfg: dict) -> np.ndarray:
    """Per instance: connected components of gaussian centers linked within outlier_k x the
    outlier_quantile of nearest-neighbor distances (gaussian density varies a lot, the median
    would split sparse parts); components below outlier_min_share of the instance (except the
    largest) are unassigned."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    out = inst_id.copy()
    for k in np.unique(inst_id[inst_id >= 0]):
        idx = np.flatnonzero(inst_id == k)
        if len(idx) < 3:
            continue
        pts = means[idx]
        tree = cKDTree(pts)
        nn = tree.query(pts, k=2)[0][:, 1]
        radius = float(cfg["outlier_k"]) * float(np.quantile(nn, float(cfg["outlier_quantile"])))
        if radius <= 0:
            continue
        pairs = tree.query_pairs(radius, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), (len(idx),) * 2)
        _, lab = connected_components(graph, directed=False)
        sizes = np.bincount(lab)
        small = (sizes < float(cfg["outlier_min_share"]) * len(idx)) & (
            np.arange(len(sizes)) != sizes.argmax()
        )
        out[idx[small[lab]]] = -1
    return out


def label_votes(
    masks: list[str],
    mass: dict[str, float],
    dets: dict[str, dict],
    zero_shot: dict[str, dict[str, float]],
    cfg: dict,
) -> dict[str, float]:
    """Labels of the source masks weighted by mask mass x quality, plus the zero-shot label
    distribution of 06 with weight zero_shot_weight; normalized, top 5."""
    votes: Counter[str] = Counter()
    zw = float(cfg["zero_shot_weight"])
    border = float(cfg["border_penalty"])
    for m in masks:
        d = dets[m]
        q = (1.0 if d["mask_score"] is None else d["mask_score"]) * (
            border if d["touches_border"] else 1.0
        )
        w = mass[m] * q
        if d["label"] is not None:
            votes[d["label"]] += w
        for label, p in zero_shot.get(m, {}).items():
            votes[label] += zw * w * p
    total = sum(votes.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 3) for k, v in votes.most_common(5)}


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / max(float(np.linalg.norm(x)), 1e-12)).astype(np.float32)


def _color(key: str) -> tuple[int, int, int]:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    c = rng.integers(60, 256, 3)
    return int(c[0]), int(c[1]), int(c[2])


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
