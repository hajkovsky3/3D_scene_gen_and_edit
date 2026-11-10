"""Phase 08_scene_graph, implementation `florence` (light alternative): captions from
Florence-2 (MIT, ~0.8 B parameters, native in transformers 5) on masked crops of the best
view. Florence-2 answers only fixed tasks (<CAPTION>, ...), not questions: no label
corrections, attributes are color and material words found in the caption.
Spec: phases/08_scene_graph.md
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, ClassVar

from PIL import Image

from scenepipe.core.paths import checkpoint_path
from scenepipe.core.phase import register
from scenepipe.phases.p08_scene_graph.geometric_vlm import (
    SceneGraphGeometricVlm,
    _CropSource,
)

log = logging.getLogger(__name__)

COLORS = ("black", "white", "gray", "grey", "silver", "red", "orange", "yellow", "green",
          "blue", "purple", "pink", "brown", "beige", "gold", "tan")  # fmt: skip
MATERIALS = {"wood": "wood", "wooden": "wood", "metal": "metal", "metallic": "metal",
             "steel": "metal", "glass": "glass", "plastic": "plastic", "fabric": "fabric",
             "leather": "leather", "ceramic": "ceramic", "paper": "paper",
             "cardboard": "cardboard", "stone": "stone", "marble": "stone"}  # fmt: skip


@register
class SceneGraphFlorence(SceneGraphGeometricVlm):
    impl = "florence"
    impl_version = "1"
    defaults: ClassVar[dict[str, Any]] = {
        **SceneGraphGeometricVlm.defaults,
        # own keys: the vlm*/crop* keys of default.yaml belong to geometric_vlm
        "florence_model": "florence-community/Florence-2-large",
        "florence_task": "<CAPTION>",  # <CAPTION> | <DETAILED_CAPTION>
        "florence_crop_size": 768,  # Florence-2 resizes to 768 x 768
        "florence_crop_context": 1.2,  # masked crop: little context is visible anyway
        # batches of 8 crops need 4.5 GB in fp16 (on a 4 GB GPU the answers came back empty)
        # and were not faster than single crops
        "florence_batch_size": 1,
    }

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        cfg = self.config
        dev = cfg["device"]
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        cache = checkpoint_path(cfg["model_dir"])
        self._proc = AutoProcessor.from_pretrained(cfg["florence_model"], cache_dir=cache)
        # vlm_compute_dtype auto: float32 on the CPU and on GTX 16xx, whose fp16 matmuls are
        # slow (GTX 1650: 1.9 s per crop in fp32, 6.2 s in fp16; fp32 peak 3.4 GB)
        compute = cfg["vlm_compute_dtype"]
        if compute == "auto":
            slow_fp16 = dev == "cuda" and "GTX 16" in torch.cuda.get_device_name()
            compute = "float32" if slow_fp16 or dev == "cpu" else "float16"
        self._dtype = getattr(torch, compute)
        self._model = AutoModelForImageTextToText.from_pretrained(
            cfg["florence_model"], cache_dir=cache, dtype=self._dtype
        ).to(dev).eval()  # fmt: skip
        self._dev = dev

    def generate(self, requests: list[tuple[list[Image.Image], str]]) -> list[str]:
        import torch

        images = [imgs[0] for imgs, _ in requests]
        inputs = self._proc(text=[p for _, p in requests], images=images, return_tensors="pt")
        inputs = inputs.to(self._dev)
        inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs, max_new_tokens=int(self.config["max_new_tokens"]), num_beams=3,
                do_sample=False,
            )  # fmt: skip
        return [t.strip() for t in self._proc.batch_decode(out, skip_special_tokens=True)]

    def describe(self, nodes: list[dict[str, Any]], ctx: dict[str, Any]) -> None:
        cfg = self.config
        todo = [
            k for k, n in enumerate(nodes)
            if n["best_views"] and (cfg["describe_structure"] or not n["is_structure"])
        ]  # fmt: skip
        stage = self.progress.stage(0.3, 0.8, self.describe_stage)
        t0 = time.perf_counter()
        self.load()
        load_s = time.perf_counter() - t0
        crops = _CropSource(self.ws, ctx)
        bs = max(int(cfg["florence_batch_size"]), 1)
        t0 = time.perf_counter()
        for start in range(0, len(todo), bs):
            chunk = todo[start : start + bs]
            reqs, keys = [], []
            for k in chunk:
                im = crops.crop(ctx["instances"][k], nodes[k]["best_views"][0],
                                int(cfg["florence_crop_size"]),
                                float(cfg["florence_crop_context"]), "masked")  # fmt: skip
                if im is not None:
                    reqs.append(([Image.fromarray(im)], cfg["florence_task"]))
                    keys.append(k)
            for k, text in zip(keys, self.generate(reqs) if reqs else [], strict=True):
                if text:
                    nodes[k]["caption"] = text
                    nodes[k]["attributes"] = caption_attributes(text)
            done = min(start + bs, len(todo))
            stage.update(done / max(len(todo), 1), f"{done}/{len(todo)} uzlov")
        self.describe_report = {
            "vlm": cfg["florence_model"],
            "num_described": len(todo),
            "vlm_load_s": round(load_s, 1),
            "vlm_s_per_node": round((time.perf_counter() - t0) / max(len(todo), 1), 2),
        }


def caption_attributes(caption: str) -> dict[str, str]:
    """First color and material word of a caption."""
    # the gray outside the mask is often described ("on a gray background"): not the object
    text = re.sub(r"\b(on|against|with)( a| the)? \w+ (background|surface)\b", " ",
                  caption.lower())  # fmt: skip
    words = re.findall(r"[a-z]+", text)
    attrs = {}
    color = next((w for w in words if w in COLORS), None)
    if color:
        attrs["color"] = "gray" if color == "grey" else color
    material = next((MATERIALS[w] for w in words if w in MATERIALS), None)
    if material:
        attrs["material"] = material
    return attrs
