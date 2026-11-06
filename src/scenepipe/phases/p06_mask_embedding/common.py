"""Phase 06_mask_embedding: shared mechanics of all implementations (crops, batching, crop
averaging, zero-shot label scores, output files, validation). Spec: phases/06_mask_embedding.md

An implementation provides the encoder: `load()`, `encode_images()`, `encode_texts()` and a
few properties; everything else here is NumPy/OpenCV, so the phase is testable without a model.
"""

from __future__ import annotations

import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image

from scenepipe.core.cameras import load_cameras
from scenepipe.core.masks import rle_decode
from scenepipe.core.phase import ContractError, Phase

log = logging.getLogger(__name__)

FILL_GRAY = 127  # neutral color of out-of-image areas and of the background of masked crops
CROP_STRATEGIES = ("bbox", "masked", "multi")


class MaskEmbeddingBase(Phase):
    phase = "06_mask_embedding"
    contract_version = 1
    defaults: ClassVar[dict[str, Any]] = {
        "crop_strategy": "multi",  # bbox | masked | multi (mean of both)
        "bbox_padding": 0.15,  # context around the bbox, share of its longer side per edge
        "masked_padding": 0.05,
        "mask_background": "gray",  # gray | blur: what replaces pixels outside the mask
        "min_crop_px": 32,  # masks whose bbox is smaller on both sides are skipped
        "batch_size": 64,
        "compute_label_scores": True,
        "text_template": "a photo of a {}",
    }

    # --- encoder (implementation) ----------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load the model (called once before encoding)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier written to meta.json, e.g. 'hf:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'."""

    @property
    @abstractmethod
    def image_size(self) -> int:
        """Side of the square model input; crops are resized to it."""

    @property
    def logit_scale(self) -> float:
        """Temperature of the zero-shot softmax (CLIP: its learned logit scale, ~100)."""
        return 100.0

    @abstractmethod
    def encode_images(self, crops: np.ndarray) -> np.ndarray:
        """uint8[N, S, S, 3] RGB crops -> float32[N, D] (normalization is done here)."""

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Texts -> float32[T, D] in the same space as the images."""

    def steps(self) -> list[str]:
        zero_shot = ["zero-shot"] if self.config["compute_label_scores"] else []
        return ["načítanie modelu", "embeddingy", *zero_shot, "zápis"]

    # --- inputs ----------------------------------------------------------------------------

    def inputs(self) -> dict[str, Path]:
        ins = {
            "cameras": self.ws.abs("01_sfm/cameras.json"),
            "images": self.ws.abs("01_sfm/images_undistorted"),
            "detections": self.ws.abs("05_segmentation/detections.json"),
            "masks": self.ws.abs("05_segmentation/masks"),
        }
        if self.config["compute_label_scores"]:
            ins["tags"] = self.ws.abs("04_tagging/tags.json")
        return ins

    def _detections(self) -> dict:
        return json.loads(self.ws.abs("05_segmentation/detections.json").read_text("utf-8"))

    # --- run -------------------------------------------------------------------------------

    def run(self) -> dict[str, Path]:
        cfg = self.config
        strategy = cfg["crop_strategy"]
        if strategy not in CROP_STRATEGIES:
            raise ValueError(f"crop_strategy {strategy!r}, expected one of {CROP_STRATEGIES}")
        kinds = ["bbox", "masked"] if strategy == "multi" else [strategy]
        doc = self._detections()
        images = {
            c.frame_id: self.ws.abs(c.image)
            for c in load_cameras(self.ws.abs("01_sfm/cameras.json"))
        }
        by_frame: dict[str, list[dict]] = {}
        for d in doc["detections"]:
            by_frame.setdefault(d["frame_id"], []).append(d)
        min_px = int(cfg["min_crop_px"])
        mask_ids, skipped = [], []
        for d in doc["detections"]:
            x0, y0, x1, y1 = d["bbox_xyxy"]
            (mask_ids if max(x1 - x0, y1 - y0) >= min_px else skipped).append(d["mask_id"])
        row_of = {m: i for i, m in enumerate(mask_ids)}
        log.info("%d masks (%d skipped as smaller than %d px), crops: %s",
                 len(mask_ids), len(skipped), min_px, kinds)  # fmt: skip

        self.progress.update(0.0, "načítanie modelu", "")
        t0 = time.perf_counter()
        self.load()
        load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        stage = self.progress.stage(0.02, 0.9, "embeddingy")
        sums: np.ndarray | None = None
        batch: list[np.ndarray] = []
        rows: list[int] = []

        def flush() -> None:
            nonlocal sums
            if not batch:
                return
            emb = l2_normalize(self.encode_images(np.stack(batch)))
            if sums is None:
                sums = np.zeros((len(mask_ids), emb.shape[1]), np.float32)
            np.add.at(sums, rows, emb)  # crops of one mask are averaged with equal weight
            batch.clear()
            rows.clear()

        size = self.image_size
        frames = [f for f in doc["frames"] if f in by_frame]
        for i, fid in enumerate(frames):
            todo = [d for d in by_frame[fid] if d["mask_id"] in row_of]
            if todo:
                with Image.open(images[fid]) as im:
                    img = np.asarray(im.convert("RGB"))
                masks_path = self.ws.abs(f"05_segmentation/masks/{fid}.json")
                rles = {r["mask_id"]: r["rle"] for r in json.loads(masks_path.read_text("utf-8"))}
                for d in todo:
                    mask = rle_decode(rles[d["mask_id"]])
                    for kind in kinds:
                        batch.append(self._crop(img, mask, d["bbox_xyxy"], kind, size))
                        rows.append(row_of[d["mask_id"]])
                    if len(batch) >= int(cfg["batch_size"]):
                        flush()
            stage.update((i + 1) / len(frames), f"{i + 1}/{len(frames)} snímok")
        flush()
        embed_s = time.perf_counter() - t0
        if sums is None:  # no masks at all: dimension from a text embedding
            sums = np.zeros((0, self.encode_texts(["object"]).shape[1]), np.float32)
        emb = l2_normalize(sums).astype(np.float16)

        dets = {d["mask_id"]: d for d in doc["detections"]}
        outs: dict[str, Path] = {}
        label_report: dict[str, Any] = {}
        if cfg["compute_label_scores"]:
            self.progress.update(0.9, "zero-shot", "")
            vocab = [v["tag"] for v in self._tags()["vocabulary"]]
            outs["label_scores"] = self.out_dir / "label_scores.json"
            label_report = self._write_label_scores(
                emb, mask_ids, vocab, dets, outs["label_scores"]
            )

        self.progress.update(0.97, "zápis", "")
        outs["embeddings"] = self.out_dir / "embeddings.npy"
        np.save(outs["embeddings"], emb)
        outs["index"] = self.out_dir / "index.json"
        _write_json(outs["index"], {
            "contract_version": self.contract_version, "mask_ids": mask_ids, "skipped": skipped,
        })  # fmt: skip
        outs["meta"] = self.out_dir / "meta.json"
        _write_json(outs["meta"], {
            "model": self.model_name, "dim": int(emb.shape[1]), "normalized": True,
            "crop_strategy": strategy, "text_template": cfg["text_template"],
        })  # fmt: skip
        outs["report"] = self.out_dir / "report.json"
        _write_json(outs["report"], {
            "num_masks": len(doc["detections"]), "num_embedded": len(mask_ids),
            "num_skipped": len(skipped), "num_frames": len(frames),
            "crops_per_mask": len(kinds), "load_s": round(load_s, 1),
            "embedding_s": round(embed_s, 1), **label_report,
        })  # fmt: skip
        return outs

    def _crop(self, img: np.ndarray, mask: np.ndarray, bbox: list[int], kind: str, size: int):
        cfg = self.config
        if kind == "bbox":
            return resize_square(crop_square(img, bbox, float(cfg["bbox_padding"])), size)
        crop = crop_square(img, bbox, float(cfg["masked_padding"]))
        m = crop_square(mask.astype(np.uint8), bbox, float(cfg["masked_padding"]), fill=0)
        return resize_square(mask_background(crop, m.astype(bool), cfg["mask_background"]), size)

    def _write_label_scores(
        self, emb: np.ndarray, mask_ids: list[str], vocab: list[str], dets: dict, out: Path
    ) -> dict[str, Any]:
        """Zero-shot softmax over the vocabulary of 04 for every mask; returns report fields."""
        scores: dict[str, dict] = {}
        agree = labeled = 0
        if vocab and mask_ids:
            texts = [self.config["text_template"].format(t) for t in vocab]
            txt = l2_normalize(self.encode_texts(texts))
            probs = softmax(self.logit_scale * emb.astype(np.float32) @ txt.T)
            for mask_id, p in zip(mask_ids, probs, strict=True):
                best = vocab[int(p.argmax())]
                scores[mask_id] = {
                    "best": best,
                    "probs": {t: round(float(v), 4) for t, v in zip(vocab, p, strict=True)},
                }
                label = dets[mask_id]["label"]
                if label is not None:
                    labeled += 1
                    agree += label == best
        _write_json(out, {
            "contract_version": self.contract_version, "labels": vocab,
            "text_template": self.config["text_template"], "scores": scores,
        })  # fmt: skip
        return {
            "vocabulary_size": len(vocab),
            "zero_shot_agrees_with_detector": round(agree / labeled, 3) if labeled else None,
        }

    def _tags(self) -> dict:
        return json.loads(self.ws.abs("04_tagging/tags.json").read_text("utf-8"))

    # --- validation ------------------------------------------------------------------------

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        index = json.loads(outputs["index"].read_text("utf-8"))
        meta = json.loads(outputs["meta"].read_text("utf-8"))
        emb = np.load(outputs["embeddings"])
        if index.get("contract_version") != self.contract_version:
            raise ContractError(f"index.json contract_version {index.get('contract_version')!r}")
        ids, skipped = index["mask_ids"], index["skipped"]
        all_ids = [d["mask_id"] for d in self._detections()["detections"]]
        if len(set(ids)) != len(ids) or set(ids) & set(skipped):
            raise ContractError("mask_ids must be unique and disjoint from skipped")
        if set(ids) | set(skipped) != set(all_ids):
            raise ContractError("mask_ids + skipped must be exactly the masks of 05")
        if emb.dtype != np.float16 or emb.ndim != 2 or emb.shape[0] != len(ids):
            raise ContractError(f"embeddings {emb.dtype}{list(emb.shape)}, {len(ids)} mask_ids")
        if meta.get("dim") != emb.shape[1] or meta.get("normalized") is not True:
            raise ContractError(f"meta.json dim {meta.get('dim')} vs width {emb.shape[1]}")
        e = emb.astype(np.float32)
        if not np.isfinite(e).all():
            raise ContractError("embeddings contain NaN/inf")
        norms = np.linalg.norm(e, axis=1)
        if norms.size and (norms.min() < 0.99 or norms.max() > 1.01):
            raise ContractError(f"norms in [{norms.min():.4f}, {norms.max():.4f}]")
        if "label_scores" in outputs:
            doc = json.loads(outputs["label_scores"].read_text("utf-8"))
            if doc["scores"] and set(doc["scores"]) != set(ids):
                raise ContractError("label_scores.json does not cover mask_ids")
            labels = set(doc["labels"])
            for mask_id, s in doc["scores"].items():
                if set(s["probs"]) != labels or s["best"] not in labels:
                    raise ContractError(f"label_scores.json: {mask_id} has other labels")


# --- helpers (unit-tested) --------------------------------------------------------------------


def square_box(bbox: list[int], padding: float) -> tuple[int, int, int]:
    """Square around the bbox center: side = longer bbox side * (1 + 2 * padding).
    Returns (x0, y0, side); it may reach outside the image."""
    x0, y0, x1, y1 = bbox
    side = max(round(max(x1 - x0, y1 - y0) * (1 + 2 * padding)), 1)
    return round((x0 + x1 - side) / 2), round((y0 + y1 - side) / 2), side


def crop_square(img: np.ndarray, bbox: list[int], padding: float, fill: int = FILL_GRAY):
    """Square crop of square_box(); the part outside the image is `fill`."""
    x0, y0, side = square_box(bbox, padding)
    h, w = img.shape[:2]
    out = np.full((side, side, *img.shape[2:]), fill, img.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def mask_background(crop: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Pixels outside the mask -> neutral gray or a strong blur of the crop."""
    if mode == "gray":
        bg = np.full_like(crop, FILL_GRAY)
    elif mode == "blur":
        bg = cv2.GaussianBlur(crop, (0, 0), sigmaX=max(crop.shape[0] / 20, 1.0))
    else:
        raise ValueError(f"mask_background {mode!r}, expected gray | blur")
    return np.where(mask[..., None], crop, bg)


def resize_square(crop: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interp)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
