"""Phase 06_mask_embedding, implementation `openclip`: OpenCLIP weights trained on LAION-2B
(default ViT-H-14, laion2b_s32b_b79k), loaded through Hugging Face transformers from the
`laion/CLIP-*` repositories (the same weights converted by LAION; MIT). Using transformers
avoids the extra open_clip_torch package in the environment of phases 04 and 05.
Spec: phases/06_mask_embedding.md
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from scenepipe.core.paths import checkpoint_path
from scenepipe.core.phase import register
from scenepipe.phases.p06_mask_embedding.common import MaskEmbeddingBase

log = logging.getLogger(__name__)


@register
class MaskEmbeddingOpenClip(MaskEmbeddingBase):
    impl = "openclip"
    impl_version = "1"
    defaults: ClassVar[dict[str, Any]] = {
        **MaskEmbeddingBase.defaults,
        "model": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "model_dir": "checkpoints/huggingface",
        "device": "auto",  # auto | cuda | cpu: the image tower
        # half precision on CUDA: less VRAM, faster with tensor cores; GPUs without them
        # (GTX 16xx) run fp16 matmuls ~4x slower than fp32 (GTX 1650: 0.4 vs 1.6 TFLOPS)
        "fp16": False,
        # the text tower encodes only the vocabulary (tens of phrases), so it can stay on the
        # CPU and leave the VRAM to the image tower (ViT-H fp32: ~2.5 GB instead of ~3.9 GB)
        "text_device": "cpu",
    }

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        cfg = self.config
        dev = cfg["device"]
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        text_dev = cfg["text_device"] if cfg["text_device"] != "auto" else dev
        self._dev, self._text_dev = dev, text_dev
        self._dtype = torch.float16 if cfg["fp16"] and dev == "cuda" else torch.float32
        cache = checkpoint_path(cfg["model_dir"])
        self._proc = AutoProcessor.from_pretrained(cfg["model"], cache_dir=cache)
        model = AutoModel.from_pretrained(cfg["model"], cache_dir=cache).eval()
        # the towers are independent modules: image tower on `device`, text tower (fp32
        # unless it shares the GPU) on `text_device`
        model.vision_model.to(dev, self._dtype)
        model.visual_projection.to(dev, self._dtype)
        self._text_dtype = self._dtype if text_dev == dev else torch.float32
        model.text_model.to(text_dev, self._text_dtype)
        model.text_projection.to(text_dev, self._text_dtype)
        self._model = model
        self._scale = float(model.logit_scale.detach().exp())
        size = self._proc.image_processor.crop_size
        # an int, a dict or (transformers 5) a SizeDict with attributes
        height = size.get("height") if isinstance(size, dict) else getattr(size, "height", size)
        self._size = int(height)
        if dev == "cuda":
            log.info("model on GPU, %.0f MB", torch.cuda.memory_allocated() / 2**20)

    @property
    def model_name(self) -> str:
        return f"hf:{self.config['model']}"

    @property
    def image_size(self) -> int:
        return self._size

    @property
    def logit_scale(self) -> float:
        return self._scale

    def encode_images(self, crops: np.ndarray) -> np.ndarray:
        import torch

        # crops are already square at the model size, so the processor only normalizes them
        px = self._proc(images=list(crops), return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            out = self._model.get_image_features(pixel_values=px.to(self._dev, self._dtype))
        return _features(out)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        tok = self._proc(text=texts, padding=True, return_tensors="pt")
        with torch.inference_mode():
            out = self._model.get_text_features(**{k: v.to(self._text_dev) for k, v in tok.items()})
        return _features(out)


def _features(out: Any) -> np.ndarray:
    # transformers 5 returns a model output whose pooler_output holds the projected features
    t = out if hasattr(out, "float") else out.pooler_output
    return t.float().cpu().numpy()
