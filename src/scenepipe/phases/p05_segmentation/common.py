"""Phase 05_segmentation: shared mechanics of all implementations (frame order, mask cleanup,
ids, RLE output, previews, report, validation). Spec: phases/05_segmentation.md

An implementation provides `segment()`, which yields the raw masks of each frame in order;
everything else here is NumPy/OpenCV, so the phase logic is testable without a model.
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

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scenepipe.core.cameras import load_cameras
from scenepipe.core.masks import mask_bbox, rle_decode, rle_encode
from scenepipe.core.phase import ContractError, Phase
from scenepipe.core.progress import StageProgress

log = logging.getLogger(__name__)

PREVIEW_ALPHA = 0.45
PREVIEW_MAX_SIDE = 1280
BORDER_PX = 2  # a mask within this many pixels of the image edge "touches the border"


@dataclass(frozen=True)
class FrameRef:
    frame_id: str
    path: Path
    width: int
    height: int


@dataclass
class RawMask:
    """One object mask of a frame as produced by an implementation."""

    mask: np.ndarray  # bool[H, W]
    label: str | None
    det_score: float | None
    mask_score: float | None
    label_scores: dict[str, float] = field(default_factory=dict)
    track: int | None = None  # implementation-local track number; None without tracking


class SegmentationBase(Phase):
    phase = "05_segmentation"
    contract_version = 1
    defaults: ClassVar[dict[str, Any]] = {
        "frame_stride": 1,  # process every k-th registered frame
        "min_area_frac": 0.001,
        "max_masks_per_frame": 100,
        "hole_area_frac": 0.0005,  # fill holes smaller than this share of the image
        "component_area_frac": 0.0005,  # drop parts smaller than this share (largest stays)
        "num_previews": 8,
    }
    has_tracks: bool = False  # set by the implementation in segment()

    def inputs(self) -> dict[str, Path]:
        return {
            "cameras": self.ws.abs("01_sfm/cameras.json"),
            "images": self.ws.abs("01_sfm/images_undistorted"),
            "tags": self.ws.abs("04_tagging/tags.json"),
        }

    @abstractmethod
    def segment(
        self, frames: list[FrameRef], vocabulary: list[str], stage: StageProgress
    ) -> Iterator[tuple[str, list[RawMask]]]:
        """Yield (frame_id, masks) for every frame, in the order of `frames`."""

    def model_steps(self) -> list[str]:
        """Progress stages reported by segment(), in order."""
        return ["segmentácia"]

    def steps(self) -> list[str]:
        return [*self.model_steps(), "zápis"]

    # --- run -------------------------------------------------------------------------------

    def run(self) -> dict[str, Path]:
        cfg = self.config
        frames = self.frames()[:: max(int(cfg["frame_stride"]), 1)]
        vocab = [v["tag"] for v in self._tags()["vocabulary"]]
        log.info("%d frames, vocabulary of %d tags", len(frames), len(vocab))
        masks_dir = self.out_dir / "masks"
        previews_dir = self.out_dir / "previews"
        for d in (masks_dir, previews_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        preview_ids = {f.frame_id for f in _spread(frames, int(cfg["num_previews"]))}
        by_id = {f.frame_id: f for f in frames}
        detections: list[dict] = []
        t0 = time.perf_counter()
        stage = self.progress.stage(0.0, 0.97, self.model_steps()[0])
        done = 0
        for frame_id, raw in self.segment(frames, vocab, stage):
            frame = by_id[frame_id]
            kept = self.clean(raw, frame.width, frame.height)
            entries, rles = [], []
            for i, (m, area, bbox) in enumerate(kept):
                mask_id = f"{frame_id}_m{i:03d}"
                entries.append(self._entry(mask_id, frame, m, area, bbox))
                rles.append({"mask_id": mask_id, "rle": rle_encode(m.mask)})
            detections.extend(entries)
            (masks_dir / f"{frame_id}.json").write_text(json.dumps(rles), encoding="utf-8")
            if frame_id in preview_ids:
                write_preview(frame.path, kept, entries, previews_dir / f"{frame_id}.jpg")
            done += 1
        if done != len(frames):
            raise RuntimeError(f"segment() returned {done} of {len(frames)} frames")
        seg_s = time.perf_counter() - t0

        self.progress.update(0.97, "zápis", "")
        out = self.out_dir / "detections.json"
        doc = {
            "contract_version": self.contract_version,
            "has_tracks": self.has_tracks,
            "frames": [f.frame_id for f in frames],
            "detections": detections,
        }
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        report = self.out_dir / "report.json"
        report.write_text(
            json.dumps(self._report(frames, detections, seg_s), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "detections": out,
            "masks": masks_dir,
            "previews": previews_dir,
            "report": report,
        }

    def frames(self) -> list[FrameRef]:
        """Registered frames in sequence order (frame_id = ingest order: time for video,
        file order for photos)."""
        cams = sorted(load_cameras(self.ws.abs("01_sfm/cameras.json")), key=lambda c: c.frame_id)
        return [FrameRef(c.frame_id, self.ws.abs(c.image), c.width, c.height) for c in cams]

    def clean(
        self, raw: list[RawMask], width: int, height: int
    ) -> list[tuple[RawMask, int, list[int]]]:
        """Cleaned masks with area and bbox: holes filled, small parts dropped, too small masks
        removed, at most max_masks_per_frame (best det_score, then largest)."""
        cfg = self.config
        img_area = width * height
        hole_px = float(cfg["hole_area_frac"]) * img_area
        part_px = float(cfg["component_area_frac"]) * img_area
        min_px = float(cfg["min_area_frac"]) * img_area
        out = []
        for m in raw:
            if m.mask.shape != (height, width):
                raise ValueError(f"mask shape {m.mask.shape}, frame {height}x{width}")
            m.mask = clean_mask(m.mask, hole_px, part_px)
            area = int(m.mask.sum())
            if area >= min_px:
                out.append((m, area, mask_bbox(m.mask)))
        out.sort(key=lambda t: (-(t[0].det_score or 0.0), -t[1]))
        return out[: int(cfg["max_masks_per_frame"])]

    def _entry(self, mask_id: str, frame: FrameRef, m: RawMask, area: int, bbox: list[int]):
        x0, y0, x1, y1 = bbox
        touches = (
            x0 < BORDER_PX or y0 < BORDER_PX or x1 > frame.width - BORDER_PX
            or y1 > frame.height - BORDER_PX
        )  # fmt: skip
        return {
            "mask_id": mask_id,
            "frame_id": frame.frame_id,
            "label": m.label,
            "label_scores": {k: round(v, 4) for k, v in m.label_scores.items()},
            "det_score": None if m.det_score is None else round(m.det_score, 4),
            "mask_score": None if m.mask_score is None else round(m.mask_score, 4),
            "bbox_xyxy": bbox,
            "area_px": area,
            "track_id": f"t{m.track:04d}" if self.has_tracks and m.track is not None else None,
            "touches_border": bool(touches),
        }

    def _report(self, frames: list[FrameRef], detections: list[dict], seg_s: float) -> dict:
        per_frame = Counter(d["frame_id"] for d in detections)
        tracks = Counter(d["track_id"] for d in detections if d["track_id"])
        return {
            "num_frames": len(frames),
            "num_masks": len(detections),
            "masks_per_frame_mean": round(len(detections) / max(len(frames), 1), 2),
            "frames_without_masks": sum(1 for f in frames if not per_frame[f.frame_id]),
            "num_tracks": len(tracks),
            "track_length_median": int(np.median(list(tracks.values()))) if tracks else 0,
            "labels": dict(Counter(d["label"] for d in detections).most_common()),
            "segmentation_s": round(seg_s, 1),
        }

    def _tags(self) -> dict:
        return json.loads(self.ws.abs("04_tagging/tags.json").read_text("utf-8"))

    # --- validation ------------------------------------------------------------------------

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        doc = json.loads(outputs["detections"].read_text("utf-8"))
        if doc.get("contract_version") != self.contract_version:
            raise ContractError(f"contract_version {doc.get('contract_version')!r}")
        has_tracks = doc["has_tracks"]
        registered = {f.frame_id: f for f in self.frames()}
        if len(set(doc["frames"])) != len(doc["frames"]) or not set(doc["frames"]) <= set(
            registered
        ):
            raise ContractError("`frames` must be unique registered frame ids")
        frames = {fid: registered[fid] for fid in doc["frames"]}
        vocab = {v["tag"] for v in self._tags()["vocabulary"]}
        by_frame: dict[str, list[dict]] = {}
        seen: set[str] = set()
        for d in doc["detections"]:
            if d["mask_id"] in seen:
                raise ContractError(f"duplicate mask_id {d['mask_id']}")
            seen.add(d["mask_id"])
            if d["frame_id"] not in frames:
                raise ContractError(f"{d['mask_id']}: unknown frame_id {d['frame_id']}")
            if d["label"] is not None and d["label"] not in vocab:
                raise ContractError(f"{d['mask_id']}: label {d['label']!r} not in vocabulary")
            if has_tracks != (d["track_id"] is not None):
                raise ContractError(f"{d['mask_id']}: track_id {d['track_id']!r} vs has_tracks")
            by_frame.setdefault(d["frame_id"], []).append(d)
        for fid, dets in by_frame.items():
            tracks = [d["track_id"] for d in dets if d["track_id"]]
            if len(tracks) != len(set(tracks)):
                raise ContractError(f"{fid}: a track_id occurs twice")
        for fid, frame in frames.items():
            path = outputs["masks"] / f"{fid}.json"
            if not path.is_file():
                raise ContractError(f"missing masks/{fid}.json")
            rles = {r["mask_id"]: r["rle"] for r in json.loads(path.read_text("utf-8"))}
            dets = by_frame.get(fid, [])
            if set(rles) != {d["mask_id"] for d in dets}:
                raise ContractError(f"masks/{fid}.json does not match detections.json")
            for d in dets:
                rle = rles[d["mask_id"]]
                if rle["size"] != [frame.height, frame.width]:
                    raise ContractError(f"{d['mask_id']}: RLE size {rle['size']}")
                m = rle_decode(rle)
                if int(m.sum()) != d["area_px"] or mask_bbox(m) != d["bbox_xyxy"]:
                    raise ContractError(f"{d['mask_id']}: area/bbox do not match the mask")


# --- helpers (unit-tested) --------------------------------------------------------------------


def clean_mask(mask: np.ndarray, max_hole_px: float, min_part_px: float) -> np.ndarray:
    """Fill holes smaller than max_hole_px, drop connected parts smaller than min_part_px
    (the largest part is always kept)."""
    m = np.ascontiguousarray(mask, np.uint8)
    if max_hole_px > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=4)
        # background components not touching the border are holes
        h, w = m.shape
        x, y, bw, bh, area = (stats[1:, j] for j in range(5))
        inner = (x > 0) & (y > 0) & (x + bw < w) & (y + bh < h)
        holes = np.flatnonzero(inner & (area < max_hole_px)) + 1
        if holes.size:
            m[np.isin(lab, holes)] = 1
    if min_part_px > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.flatnonzero((areas >= min_part_px) | (areas == areas.max())) + 1
            m = np.isin(lab, keep).astype(np.uint8)
    return m.astype(bool)


def _spread(items: list, k: int) -> list:
    if k <= 0 or not items:
        return []
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, min(k, len(items)))).astype(int))
    return [items[i] for i in idx]


def _color(key: str) -> tuple[int, int, int]:
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    c = rng.integers(60, 256, 3)
    return int(c[0]), int(c[1]), int(c[2])


def write_preview(
    image: Path, kept: list[tuple[RawMask, int, list[int]]], entries: list[dict], out: Path
) -> None:
    """Frame with colored masks, outlines and `label track` captions."""
    with Image.open(image) as im:
        rgb = np.asarray(im.convert("RGB"), np.float32)
    over = rgb.copy()
    order = sorted(range(len(kept)), key=lambda i: -kept[i][1])  # small masks drawn last
    for i in order:
        m = kept[i][0].mask
        color = np.array(_color(entries[i]["track_id"] or entries[i]["mask_id"]), np.float32)
        over[m] = (1 - PREVIEW_ALPHA) * over[m] + PREVIEW_ALPHA * color
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(over, contours, -1, color.tolist(), 2)
    img = Image.fromarray(over.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for i in order:
        e = entries[i]
        x0, y0 = e["bbox_xyxy"][:2]
        text = f"{e['label']} {e['track_id'] or ''}".strip()
        draw.rectangle([x0, y0, x0 + 7 * len(text) + 6, y0 + 14], fill=(0, 0, 0))
        draw.text((x0 + 3, y0 + 1), text, fill=_color(e["track_id"] or e["mask_id"]))
    scale = PREVIEW_MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    img.save(out, quality=88)
