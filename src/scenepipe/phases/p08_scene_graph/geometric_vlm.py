"""Phase 08_scene_graph, implementation `geometric_vlm` (primary): the geometry and relations of
`common.py`, captions, attributes and label corrections from a vision-language model.

Every node is shown to the VLM in its best views (`views_per_caption`): a square crop around
the mask with some context, the object outlined in red (the model sees what surrounds the
object, but knows which one is meant). The model answers with JSON: a name, whether the label
of 07 fits, a short caption, color and material. The label of 07 has priority; it is replaced
only when the model says it is wrong and its name is in the vocabulary of 04 or among the
votes of 07 (`label_correction: vocabulary`), so the graph keeps the labels that phase 10
can ground with the same text encoder, and (`relabel_check`) when the model then picks the
new name over the old one in a forced choice, asked separately for every view.

Default model: Qwen3-VL-4B-Instruct (Apache 2.0), 4-bit NF4 through bitsandbytes: ~2.9 GB of
VRAM, fits a 4 GB GPU. Spec: phases/08_scene_graph.md
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image

from scenepipe.core.masks import rle_decode
from scenepipe.core.phase import register
from scenepipe.phases.p08_scene_graph.common import SceneGraphBase

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
FILL_GRAY = 127
OUTLINE_RGB = (255, 0, 0)
LABEL_CORRECTION = ("off", "vocabulary", "any")
ATTRIBUTES = ("color", "material")
_EMPTY = {"", "none", "unknown", "n/a", "null", "various", "multiple", "mixed"}

PROMPT = (
    "{detected} The red outline marks one region of the image (the outline itself is not "
    "part of the object, do not mention it). The region may be a whole object, a part of a "
    "larger object, or an object lying on another one: describe only what is inside the "
    "outline, not what it rests on.\n"
    "Reply with one line of compact JSON with exactly these keys:\n"
    '{{"name": "<common noun>", "label_correct": <true|false>, '
    '"caption": "<at most 12 words>", "color": "<main color>", "material": "<main material>"}}\n'
    "{correct_hint} If you give another name, prefer one of: {vocabulary}."
)


@register
class SceneGraphGeometricVlm(SceneGraphBase):
    impl = "geometric_vlm"
    impl_version = "1"
    describe_stage = "popisy"
    defaults: ClassVar[dict[str, Any]] = {
        **SceneGraphBase.defaults,
        "vlm": "Qwen/Qwen3-VL-4B-Instruct",
        "model_dir": "checkpoints/huggingface",
        "vlm_quantization": "4bit",  # none | 8bit | 4bit (bitsandbytes, CUDA only)
        "vlm_compute_dtype": "auto",  # auto | float16 | float32: matmuls of 4-bit layers
        "device": "auto",  # auto | cuda | cpu
        "views_per_caption": 2,
        "crop_size": 448,  # side of a square crop sent to the VLM, px
        "crop_context": 1.6,  # crop side = longer bbox side * crop_context
        # nodes per generate() call; bitsandbytes 4-bit gains nothing from batches on a GTX 1650
        "vlm_batch_size": 1,
        "max_new_tokens": 80,
        "label_correction": "vocabulary",  # off | vocabulary | any
        "relabel_check": True,  # confirm a correction by a forced choice old/new in each view
        "describe_structure": False,  # captions also for floor, walls (usually useless)
    }

    def inputs(self) -> dict[str, Path]:
        return {
            **super().inputs(),
            "images": self.ws.abs("01_sfm/images_undistorted"),
            "masks": self.ws.abs("05_segmentation/masks"),
        }

    # --- model ------------------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        cfg = self.config
        dev = cfg["device"]
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        quant = cfg["vlm_quantization"]
        if quant not in ("none", "8bit", "4bit"):
            raise ValueError(f"vlm_quantization {quant!r}, expected none | 8bit | 4bit")
        if dev == "cpu" and quant != "none":
            log.warning("bitsandbytes quantization needs CUDA, loading %s unquantized", cfg["vlm"])
            quant = "none"
        # weights fp16 on CUDA (GTX 16xx has no bf16), fp32 on the CPU
        dtype = torch.float16 if dev == "cuda" else torch.float32
        compute = cfg["vlm_compute_dtype"]
        if compute == "auto":
            # GTX 16xx (Turing without tensor cores) runs fp16 matmuls ~4x slower than fp32:
            # the prefill of ~400 tokens takes 14 s in fp16, 6 s in fp32 (GTX 1650)
            slow_fp16 = dev == "cuda" and "GTX 16" in torch.cuda.get_device_name()
            compute = "float32" if slow_fp16 or dev == "cpu" else "float16"
        if compute not in ("float16", "float32"):
            raise ValueError(f"vlm_compute_dtype {compute!r}, expected auto | float16 | float32")
        kwargs: dict[str, Any] = {"dtype": dtype, "device_map": dev}
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, compute), bnb_4bit_use_double_quant=True,
            )  # fmt: skip
        elif quant == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        cache = REPO_ROOT / cfg["model_dir"]
        self._proc = AutoProcessor.from_pretrained(cfg["vlm"], cache_dir=cache)
        self._proc.tokenizer.padding_side = "left"  # batched generation
        self._model = AutoModelForImageTextToText.from_pretrained(
            cfg["vlm"], cache_dir=cache, **kwargs
        ).eval()
        self._dev = dev
        if dev == "cuda":
            log.info("VLM on GPU (%s), %.0f MB", quant, torch.cuda.memory_allocated() / 2**20)

    def generate(self, requests: list[tuple[list[Image.Image], str]]) -> list[str]:
        """One answer per (images, prompt) request, greedy decoding."""
        import torch

        convs = [
            [{"role": "user", "content": [
                *({"type": "image", "image": im} for im in images),
                {"type": "text", "text": prompt},
            ]}]
            for images, prompt in requests
        ]  # fmt: skip
        inputs = self._proc.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True,
        ).to(self._dev)  # fmt: skip
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=int(self.config["max_new_tokens"]), do_sample=False
            )
        return self._proc.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    # --- describe ---------------------------------------------------------------------------

    def describe(self, nodes: list[dict[str, Any]], ctx: dict[str, Any]) -> None:
        cfg = self.config
        mode = cfg["label_correction"]
        if mode not in LABEL_CORRECTION:
            raise ValueError(f"label_correction {mode!r}, expected one of {LABEL_CORRECTION}")
        todo = [
            k for k, n in enumerate(nodes)
            if n["best_views"] and (cfg["describe_structure"] or not n["is_structure"])
        ]  # fmt: skip
        vocab = ctx["vocabulary"]
        stage = self.progress.stage(0.3, 0.8, self.describe_stage)
        t0 = time.perf_counter()
        self.load()
        load_s = time.perf_counter() - t0
        log.info("VLM %s loaded in %.0f s, describing %d of %d nodes",
                 cfg["vlm"], load_s, len(todo), len(nodes))  # fmt: skip

        crops = _CropSource(self.ws, ctx)
        bs = max(int(cfg["vlm_batch_size"]), 1)
        relabeled: list[dict] = []
        rejected: list[dict] = []
        failed: list[str] = []
        t0 = time.perf_counter()
        for start in range(0, len(todo), bs):
            chunk = todo[start : start + bs]
            requests = []
            for k in chunk:
                n, inst = nodes[k], ctx["instances"][k]
                images = [
                    crops.crop(inst, view, int(cfg["crop_size"]), float(cfg["crop_context"]))
                    for view in n["best_views"][: int(cfg["views_per_caption"])]
                ]
                requests.append(([Image.fromarray(im) for im in images if im is not None],
                                 build_prompt(n["label"], vocab)))  # fmt: skip
            answers = self.generate(requests)
            for k, (images, _), text in zip(chunk, requests, answers, strict=True):
                n = nodes[k]
                resp = parse_response(text)
                if resp is None:
                    log.warning("%s: unparsable VLM answer %r", n["node_id"], text[:200])
                    failed.append(n["node_id"])
                    continue
                old, new = n["label"], apply_description(n, resp, vocab, mode)
                if new is None:
                    continue
                if old is not None and cfg["relabel_check"]:
                    # forced choice old/new, every view separately: all must pick the new one
                    checks = self.generate([([im], build_check_prompt(old, new)) for im in images])
                    if not all(check_vote(a, old, new) for a in checks):
                        rejected.append({"node_id": n["node_id"], "from": old, "to": new})
                        log.info("%s: label %s -> %s rejected by the check %s",
                                 n["node_id"], old, new, checks)  # fmt: skip
                        continue
                n["label_original"], n["label"] = old, new
                relabeled.append({"node_id": n["node_id"], "from": old, "to": new})
                log.info("%s: label %s -> %s", n["node_id"], old, new)
            done = min(start + bs, len(todo))
            stage.update(done / max(len(todo), 1), f"{done}/{len(todo)} uzlov")
        self.describe_report = {
            "vlm": cfg["vlm"],
            "vlm_quantization": cfg["vlm_quantization"],
            "num_described": len(todo) - len(failed),
            "num_vlm_failed": len(failed),
            "vlm_failed": failed,
            "relabeled": relabeled,
            "relabel_rejected": rejected,
            "vlm_load_s": round(load_s, 1),
            "vlm_s_per_node": round((time.perf_counter() - t0) / max(len(todo), 1), 2),
        }

    def semantic_relations(self, nodes: list[dict], edges: list[dict]) -> list[dict]:
        log.warning("geometric_vlm: semantic_relations (LLM) not implemented yet, skipped")
        return []


class _CropSource:
    """Images and masks of 05, read lazily, the last frame cached."""

    def __init__(self, ws: Any, ctx: dict[str, Any]) -> None:
        self.ws, self.cams, self.dets = ws, ctx["cameras"], ctx["detections"]
        self._frame: str | None = None
        self._img: np.ndarray | None = None
        self._rles: dict[str, dict] = {}

    def _load(self, fid: str) -> None:
        if fid != self._frame:
            with Image.open(self.ws.abs(self.cams[fid].image)) as im:
                self._img = np.asarray(im.convert("RGB"))
            path = self.ws.abs(f"05_segmentation/masks/{fid}.json")
            self._rles = {r["mask_id"]: r["rle"] for r in json.loads(path.read_text("utf-8"))}
            self._frame = fid

    def crop(
        self, inst: dict, fid: str, size: int, context: float, style: str = "outline"
    ) -> np.ndarray | None:
        masks = [self.dets[m] for m in inst["source_masks"]
                 if m in self.dets and self.dets[m]["frame_id"] == fid]  # fmt: skip
        if not masks:
            return None
        d = max(masks, key=lambda d: d["area_px"])
        self._load(fid)
        assert self._img is not None
        return outlined_crop(self._img, rle_decode(self._rles[d["mask_id"]]),
                             d["bbox_xyxy"], context, size, style)  # fmt: skip


# --- helpers (unit-tested) --------------------------------------------------------------------


def outlined_crop(
    img: np.ndarray, mask: np.ndarray, bbox: list[int], context: float, size: int,
    style: str = "outline",
) -> np.ndarray:  # fmt: skip
    """Square crop around the bbox (side = longer bbox side * context, gray outside the
    image), resized so that its side is `size`, with the mask outline drawn in red
    (`style="outline"`) or everything outside the mask gray (`style="masked"`)."""
    x0, y0, x1, y1 = bbox
    side = max(round(max(x1 - x0, y1 - y0) * context), 8)
    cx0, cy0 = round((x0 + x1 - side) / 2), round((y0 + y1 - side) / 2)
    h, w = img.shape[:2]
    out = np.full((side, side, 3), FILL_GRAY, np.uint8)
    m = np.zeros((side, side), np.uint8)
    sx0, sy0, sx1, sy1 = max(cx0, 0), max(cy0, 0), min(cx0 + side, w), min(cy0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - cy0 : sy1 - cy0, sx0 - cx0 : sx1 - cx0] = img[sy0:sy1, sx0:sx1]
        m[sy0 - cy0 : sy1 - cy0, sx0 - cx0 : sx1 - cx0] = mask[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    out = cv2.resize(out, (size, size), interpolation=interp)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    if style == "masked":
        out[m == 0] = FILL_GRAY
        return out
    if style != "outline":
        raise ValueError(f"crop style {style!r}, expected outline | masked")
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, OUTLINE_RGB, max(2, size // 150))
    return out


def build_prompt(label: str | None, vocabulary: list[str]) -> str:
    if label is None:
        detected = "The object outlined in red was not identified."
        correct_hint = "label_correct is false."
    else:
        detected = f'An object detector labeled the region outlined in red "{label}".'
        correct_hint = (
            f'label_correct: is "{label}" a reasonable name for the outlined region? '
            "Answer false only if it is clearly wrong."
        )
    vocab = ", ".join(sorted(set(vocabulary))) or "(none)"
    return PROMPT.format(detected=detected, correct_hint=correct_hint, vocabulary=vocab)


def parse_response(text: str) -> dict[str, Any] | None:
    """The first JSON object in the answer (code fences and chatter around it are ignored)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        doc = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def normalize_name(name: Any) -> str:
    """Lowercase noun phrase without articles and extra spaces."""
    if not isinstance(name, str):
        return ""
    s = re.sub(r"[^a-z0-9 \-]", " ", name.lower())
    s = re.sub(r"^(a|an|the)\s+", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def _match_vocabulary(name: str, allowed: set[str]) -> str | None:
    if name in allowed:
        return name
    for cand in (name[:-1] if name.endswith("s") else None,
                 name[:-2] if name.endswith("es") else None):  # fmt: skip
        if cand and cand in allowed:
            return cand
    return None


def apply_description(node: dict[str, Any], resp: dict[str, Any], vocab: list[str],
                      mode: str) -> str | None:  # fmt: skip
    """Fill caption and attributes of the node from a parsed VLM answer. Returns the label
    the model proposes instead of the current one (see the module docstring), or None; the
    label itself is not changed here."""
    caption = resp.get("caption")
    if isinstance(caption, str) and caption.strip():
        node["caption"] = caption.strip()
    attrs = {}
    for key in ATTRIBUTES:
        val = normalize_name(resp.get(key))
        if val and val not in _EMPTY:
            attrs[key] = val
    node["attributes"] = attrs

    name = normalize_name(resp.get("name"))
    label = node["label"]
    rejected = label is None or resp.get("label_correct") is False
    if mode == "off" or not name or not rejected or name == label:
        return None
    if mode == "vocabulary":
        match = _match_vocabulary(name, set(vocab) | set(node["label_votes"]))
        if match is None or match == label:
            return None
        name = match
    return name


def build_check_prompt(old: str, new: str) -> str:
    return (f'Which word better names the object outlined in red: "{old}" or "{new}"? '
            "Answer with only that word.")  # fmt: skip


def check_vote(answer: str, old: str, new: str) -> bool:
    """True if the answer to build_check_prompt() names the new label and not the old one."""
    a = f" {normalize_name(answer)} "
    has_new = f" {new} " in a or f" {new}s " in a
    has_old = f" {old} " in a or f" {old}s " in a
    return has_new and not has_old
