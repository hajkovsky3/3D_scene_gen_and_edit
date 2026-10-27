"""Phase 05_segmentation, implementation `gdino_sam2`: Grounding DINO boxes from the
vocabulary of 04, SAM 2 masks from the boxes and SAM 2 video tracking between re-detections.
Both models come from Hugging Face transformers (Apache 2.0). Spec: phases/05_segmentation.md

Tracking works in windows of `redetect_every` frames (as the "continuous id" demo of
Grounded-SAM-2): the detections on the first frame of a window are matched by box IoU to
the masks tracked into that frame by the previous window (same track, otherwise a new one);
tracks left without a detection continue from their tracked mask's box for at most
`carry_windows` windows. SAM 2 then segments all of them and propagates them to the rest of
the window. Its cost grows with the number of objects (~0.35 s per object and frame on a
GTX 1650), so `frame_stride` and `max_masks_per_frame` are the main speed knobs.
Detection runs first on all window starts and Grounding DINO is freed before SAM 2 loads,
so the two models never share the GPU memory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from scenepipe.core.masks import mask_bbox
from scenepipe.core.phase import register
from scenepipe.core.progress import StageProgress
from scenepipe.phases.p05_segmentation.common import FrameRef, RawMask, SegmentationBase

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
DETECT_SHARE = 0.3  # share of segment() progress taken by detection when tracking


@dataclass
class Detection:
    box: list[float]  # x0, y0, x1, y1 in pixels
    label: str
    score: float
    label_scores: dict[str, float]


@register
class SegmentationGdinoSam2(SegmentationBase):
    impl = "gdino_sam2"
    impl_version = "1"
    defaults: ClassVar[dict[str, Any]] = {
        **SegmentationBase.defaults,
        "gdino_model": "IDEA-Research/grounding-dino-tiny",
        "sam2_model": "facebook/sam2.1-hiera-small",
        "model_dir": "checkpoints/huggingface",
        "device": "auto",
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "vocab_batch": 20,
        "nms_iou": 0.7,
        "tracking": True,
        "redetect_every": 10,
        "track_iou": 0.5,
        "carry_windows": 2,
    }

    def model_steps(self) -> list[str]:
        return ["detekcia", "segmentácia"]

    def segment(
        self, frames: list[FrameRef], vocabulary: list[str], stage: StageProgress
    ) -> Iterator[tuple[str, list[RawMask]]]:
        import torch

        cfg = self.config
        dev = cfg["device"]
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        tracking = bool(cfg["tracking"])
        if tracking and len({(f.width, f.height) for f in frames}) > 1:
            log.warning("frames differ in size, tracking disabled")
            tracking = False
        self.has_tracks = tracking
        step = max(int(cfg["redetect_every"]), 1) if tracking else 1
        starts = list(range(0, len(frames), step))

        split = stage.lo + DETECT_SHARE * (stage.hi - stage.lo)
        det_stage = StageProgress(self.progress, stage.lo, split, "detekcia")
        dets = self._detect_all([frames[i] for i in starts], vocabulary, dev, det_stage)
        seg_stage = StageProgress(self.progress, split, stage.hi, "segmentácia")
        yield from self._segment_windows(frames, starts, dets, dev, seg_stage)

    # --- detection -------------------------------------------------------------------------

    def _detect_all(
        self, frames: list[FrameRef], vocabulary: list[str], dev: str, stage: StageProgress
    ) -> dict[str, list[Detection]]:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        cfg = self.config
        cache = REPO_ROOT / cfg["model_dir"]
        self.progress.update(stage.lo, "detekcia", "Grounding DINO")
        proc = AutoProcessor.from_pretrained(cfg["gdino_model"], cache_dir=cache)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg["gdino_model"], cache_dir=cache
        )
        model = model.to(dev).eval()
        n = max(int(cfg["vocab_batch"]), 1)
        batches = [vocabulary[i : i + n] for i in range(0, len(vocabulary), n)]
        out: dict[str, list[Detection]] = {}
        for i, f in enumerate(frames):
            with Image.open(f.path) as im:
                image = im.convert("RGB")
            found: list[Detection] = []
            for phrases in batches:
                found += detect(model, proc, image, phrases, cfg, dev)
            out[f.frame_id] = nms(found, float(cfg["nms_iou"]))[: int(cfg["max_masks_per_frame"])]
            stage.update((i + 1) / len(frames), f"{i + 1}/{len(frames)} snímok")
        n_dets = sum(len(d) for d in out.values())
        log.info("Grounding DINO: %d detections on %d frames", n_dets, len(frames))
        del model
        if dev == "cuda":
            log.info("detection peak VRAM %.0f MB", torch.cuda.max_memory_allocated() / 2**20)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return out

    # --- masks and tracking ----------------------------------------------------------------

    def _segment_windows(
        self,
        frames: list[FrameRef],
        starts: list[int],
        dets: dict[str, list[Detection]],
        dev: str,
        stage: StageProgress,
    ) -> Iterator[tuple[str, list[RawMask]]]:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        cfg = self.config
        cache = REPO_ROOT / cfg["model_dir"]
        self.progress.update(stage.lo, "segmentácia", "SAM 2")
        proc = Sam2VideoProcessor.from_pretrained(cfg["sam2_model"], cache_dir=cache)
        model = Sam2VideoModel.from_pretrained(cfg["sam2_model"], cache_dir=cache)
        model = model.to(dev).eval()

        next_track = 0
        prev: dict[int, np.ndarray] = {}  # track -> mask tracked into the current window start
        last_det: dict[int, Detection] = {}  # track -> its latest detection (label, scores)
        missed: dict[int, int] = {}  # track -> windows in a row without a matching detection
        carry = int(cfg["carry_windows"])
        emitted = 0
        for w, k in enumerate(starts):
            last = w + 1 == len(starts)
            end = k if not self.has_tracks else len(frames) - 1 if last else starts[w + 1]
            window = frames[k : end + 1]
            # with tracking, the last frame of a window (= start of the next) is only tracked
            # into, for matching; it is emitted by the next window
            n_emit = len(window) if last or not self.has_tracks else len(window) - 1
            objects = list(dets[frames[k].frame_id])
            tracks: list[int | None] = [None] * len(objects)
            if self.has_tracks:
                prev_ids = list(prev)
                prev_boxes = [[float(v) for v in mask_bbox(prev[t])] for t in prev_ids]
                match = match_boxes([d.box for d in objects], prev_boxes, float(cfg["track_iou"]))
                for i, j in enumerate(match):
                    if j is None:
                        tracks[i] = next_track
                        next_track += 1
                    else:
                        tracks[i] = prev_ids[j]
                    last_det[tracks[i]] = objects[i]
                    missed[tracks[i]] = 0
                # tracks without a matching detection continue from their tracked mask
                matched = set(tracks)
                for t, box in zip(prev_ids, prev_boxes, strict=True):
                    if t in matched or missed[t] >= carry:
                        continue
                    if len(objects) >= int(cfg["max_masks_per_frame"]):
                        break
                    d = last_det[t]
                    objects.append(Detection(box, d.label, d.score, d.label_scores))
                    tracks.append(t)
                    missed[t] += 1
            if not objects:
                for f in window[:n_emit]:
                    yield f.frame_id, []
                emitted += n_emit
                prev = {}
                stage.update(emitted / len(frames), f"{emitted}/{len(frames)} snímok")
                continue

            images = []
            for f in window:
                with Image.open(f.path) as im:
                    images.append(im.convert("RGB"))
            session = proc.init_video_session(
                video=images, inference_device=dev, video_storage_device="cpu"
            )
            ids = list(range(1, len(objects) + 1))
            boxes = [[d.box for d in objects]]
            proc.add_inputs_to_inference_session(
                session, frame_idx=0, obj_ids=ids, input_boxes=boxes
            )
            size = [[session.video_height, session.video_width]]
            next_prev: dict[int, np.ndarray] = {}
            with torch.inference_mode():
                model(inference_session=session, frame_idx=0)
                for out in model.propagate_in_video_iterator(session):
                    fi = out.frame_idx
                    masks, scores = _frame_masks(proc, out, size)
                    if fi < n_emit:
                        yield window[fi].frame_id, _raw(masks, scores, objects, tracks)
                        emitted += 1
                    else:
                        next_prev = {
                            t: m
                            for t, m, sc in zip(tracks, masks, scores, strict=True)
                            if sc > 0.5 and m.any()
                        }
            prev = next_prev
            stage.update(emitted / len(frames), f"{emitted}/{len(frames)} snímok")
            del session
        del model
        if dev == "cuda":
            log.info("SAM 2 peak VRAM %.0f MB", torch.cuda.max_memory_allocated() / 2**20)
            torch.cuda.empty_cache()


def _frame_masks(
    proc: Any, out: Any, size: list[list[int]]
) -> tuple[list[np.ndarray], list[float]]:
    """Full-resolution bool masks and presence probabilities, in object id order."""
    import torch

    masks = proc.post_process_masks([out.pred_masks], original_sizes=size, binarize=True)[0]
    masks = masks[:, 0].cpu().numpy().astype(bool)
    scores = torch.sigmoid(out.object_score_logits.float().flatten()).cpu().tolist()
    return list(masks), scores


def _raw(
    masks: list[np.ndarray], scores: list[float], found: list[Detection], tracks: list[int | None]
) -> list[RawMask]:
    """Masks of present objects (presence > 0.5, non-empty) with their detection's label."""
    out = []
    for m, s, d, t in zip(masks, scores, found, tracks, strict=True):
        if s > 0.5 and m.any():
            out.append(RawMask(m, d.label, d.score, s, dict(d.label_scores), t))
    return out


# --- helpers (unit-tested) --------------------------------------------------------------------


def phrase_token_spans(offsets: list[tuple[int, int]], char_spans: list[tuple[int, int]]):
    """Token indices of each phrase, from tokenizer offsets and the phrases' character spans."""
    spans = []
    for a, b in char_spans:
        spans.append([i for i, (s, e) in enumerate(offsets) if a <= s < e <= b])
    return spans


def build_prompt(phrases: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Grounding DINO text "a . b . c ." and the character span of every phrase."""
    text, spans = "", []
    for p in phrases:
        spans.append((len(text), len(text) + len(p)))
        text += p + " . "
    return text.strip(), spans


def detect(
    model: Any, proc: Any, image: Image.Image, phrases: list[str], cfg: dict, dev: str
) -> list[Detection]:
    """Boxes for one batch of phrases. The score of a phrase is the max over its tokens; a box
    is kept if its best phrase scores >= box_threshold, `label_scores` lists the phrases with
    score >= text_threshold."""
    import torch

    text, char_spans = build_prompt(phrases)
    enc = proc(images=image, text=text, return_tensors="pt").to(dev)
    offsets = proc.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    if len(offsets) != enc["input_ids"].shape[1]:
        raise RuntimeError("Grounding DINO tokenization does not match the prompt")
    spans = phrase_token_spans(offsets, char_spans)
    with torch.inference_mode():
        out = model(**enc)
    probs = out.logits[0].sigmoid().float()  # [queries, max_text_len]
    phrase_scores = torch.stack([probs[:, s].max(dim=1).values for s in spans], dim=1)
    best, best_idx = phrase_scores.max(dim=1)
    keep = torch.nonzero(best >= float(cfg["box_threshold"])).flatten().tolist()
    w, h = image.size
    cxcywh = out.pred_boxes[0].float().cpu().numpy()
    ps = phrase_scores.cpu().numpy()
    dets = []
    for q in keep:
        cx, cy, bw, bh = cxcywh[q]
        box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
        box = [float(np.clip(v, 0, lim)) for v, lim in zip(box, (w, h, w, h), strict=True)]
        scores = {
            phrases[j]: float(ps[q, j])
            for j in np.argsort(-ps[q])
            if ps[q, j] >= float(cfg["text_threshold"])
        }
        label = phrases[int(best_idx[q])]
        dets.append(Detection(box, label, float(best[q]), scores))
    return dets


def box_iou(a: list[float], b: list[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[Detection], iou: float) -> list[Detection]:
    """Greedy NMS across labels, best score first; a suppressed box adds its label scores to
    the kept one (max per label)."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        for k in kept:
            if box_iou(k.box, d.box) > iou:
                for lab, s in d.label_scores.items():
                    k.label_scores[lab] = max(k.label_scores.get(lab, 0.0), s)
                break
        else:
            kept.append(d)
    for k in kept:
        k.label_scores = dict(sorted(k.label_scores.items(), key=lambda kv: -kv[1]))
    return kept


def match_boxes(
    boxes: list[list[float]], prev: list[list[float]], min_iou: float
) -> list[int | None]:
    """Greedy one-to-one matching of new boxes to previous ones (index into `prev`) by IoU."""
    pairs = sorted(
        ((box_iou(a, b), i, j) for i, a in enumerate(boxes) for j, b in enumerate(prev)),
        reverse=True,
    )
    out: list[int | None] = [None] * len(boxes)
    used: set[int] = set()
    for iou, i, j in pairs:
        if iou < min_iou:
            break
        if out[i] is None and j not in used:
            out[i] = j
            used.add(j)
    return out
