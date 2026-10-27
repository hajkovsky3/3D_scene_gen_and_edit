"""Phase 04_tagging, implementation `ram_plus`: Recognize Anything Plus (RAM++, Swin-L, 4585
tags) from the `ram` package (github.com/xinyu1205/recognize-anything, Apache 2.0), called
as an external dependency. Spec: phases/04_tagging.md

`RAM_plus.generate_tag` returns only tags above its own thresholds, so `_scores` repeats its
forward pass (same submodules, same order) and returns the sigmoid scores of all tags.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from scenepipe.core.phase import register
from scenepipe.core.progress import StageProgress
from scenepipe.phases.p04_tagging.common import TaggingBase, _repo_path

log = logging.getLogger(__name__)

IMAGE_SIZE = 384  # the Swin-L checkpoint is trained at 384 x 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@register
class TaggingRamPlus(TaggingBase):
    impl = "ram_plus"
    impl_version = "1"
    defaults: ClassVar[dict[str, Any]] = {
        **TaggingBase.defaults,
        "checkpoint": "checkpoints/ram_plus/ram_plus_swin_large_14m.pth",
        "tokenizer_dir": "checkpoints/bert-base-uncased",
        "device": "auto",  # auto | cuda | cpu
        "fp16": True,  # half precision on CUDA (~1.5 GB VRAM instead of ~3 GB)
        "batch_size": 4,
        # per-tag thresholds calibrated by the RAM++ authors (0.65 for most objects, up to 1.0
        # for verbs like "see"); a tag counts if it passes both these and score_threshold
        "class_thresholds": True,
    }

    def model_steps(self) -> list[str]:
        return ["načítanie modelu", "tagovanie"]

    def tag_frames(
        self, frames: list[tuple[str, Path]], stage: StageProgress
    ) -> dict[str, list[tuple[str, float]]]:
        import torch

        cfg = self.config
        # model files live outside the workspace, so they are not manifest inputs
        for key in ("checkpoint", "tokenizer_dir"):
            if not _repo_path(cfg[key]).exists():
                raise FileNotFoundError(f"{key}: {cfg[key]} not found (phases/04_tagging.md)")
        dev = cfg["device"]
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        half = bool(cfg["fp16"]) and dev == "cuda"
        load_share = 0.3
        self.progress.update(stage.lo, "načítanie modelu", "RAM++ Swin-L")
        model = load_model(
            _repo_path(cfg["checkpoint"]), _repo_path(cfg["tokenizer_dir"]), dev, half
        )
        tag_list = [str(t) for t in model.tag_list]
        thr = np.full(len(tag_list), float(cfg["score_threshold"]), np.float32)
        if cfg["class_thresholds"]:
            thr = np.maximum(thr, model.class_threshold.cpu().numpy())
        log.info("RAM++ loaded on %s (%s), %d tags", dev, "fp16" if half else "fp32", len(tag_list))

        tagging = self.progress.stage(
            stage.lo + load_share * (stage.hi - stage.lo), stage.hi, "tagovanie"
        )
        out: dict[str, list[tuple[str, float]]] = {}
        bs = max(int(cfg["batch_size"]), 1)
        for i in range(0, len(frames), bs):
            chunk = frames[i : i + bs]
            batch = torch.from_numpy(np.stack([preprocess(p) for _, p in chunk])).to(dev)
            scores = _scores(model, batch.half() if half else batch).float().cpu().numpy()
            for (fid, _), s in zip(chunk, scores, strict=True):
                keep = np.flatnonzero(s >= thr)
                out[fid] = [(tag_list[j], float(s[j])) for j in keep]
            tagging.update((i + len(chunk)) / len(frames), f"{i + len(chunk)}/{len(frames)} snímok")
        del model
        if dev == "cuda":
            log.info("peak VRAM %.0f MB", torch.cuda.max_memory_allocated() / 2**20)
            torch.cuda.empty_cache()
        return out


def preprocess(path: Path) -> np.ndarray:
    """[3, 384, 384] float32, ImageNet-normalized (as ram.transform.get_transform)."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1).copy()


def load_model(checkpoint: Path, tokenizer_dir: Path, device: str, half: bool) -> Any:
    _patch_compat()
    import ram.models  # noqa: F401  (the package re-exports the function under the module name)

    rp = sys.modules["ram.models.ram_plus"]

    # RAM++ uses its BERT tokenizer only for the vocabulary size (embedding rows of the
    # tagging head, deleted right after); BertTokenizer of transformers 5 lacks the API that
    # ram's init_tokenizer calls, so a stand-in with the same length is passed instead
    rp.init_tokenizer = _VocabSize
    # the tagging head subclasses PreTrainedModel of transformers 4; under transformers 5 its
    # weight init and embedding resize fail. Both are moot here: every weight comes from the
    # checkpoint and the token embeddings are deleted (RAM_plus.del_selfattention)
    rp.BertModel.init_weights = lambda self: None
    rp.BertModel.resize_token_embeddings = lambda self, *a, **k: None
    for name, fn in (
        ("invert_attention_mask", _invert_attention_mask),
        ("get_head_mask", _no_head_mask),
    ):
        if not hasattr(rp.BertModel, name):  # removed from PreTrainedModel in transformers 5
            setattr(rp.BertModel, name, fn)
    model = rp.ram_plus(
        pretrained=str(checkpoint),
        image_size=IMAGE_SIZE,
        vit="swin_l",
        text_encoder_type=str(tokenizer_dir),
    )
    model.eval()
    if half:
        model.half()
    return model.to(device)


def _invert_attention_mask(self: Any, mask: Any) -> Any:
    """1 = attend, 0 = masked -> additive mask, as in transformers 4 (ModuleUtilsMixin)."""
    import torch

    ext = mask[:, None, :, :] if mask.dim() == 3 else mask[:, None, None, :]
    ext = ext.to(dtype=self.dtype)
    return (1.0 - ext) * torch.finfo(self.dtype).min


def _no_head_mask(self: Any, head_mask: Any, num_hidden_layers: int) -> list[None]:
    if head_mask is not None:
        raise NotImplementedError("head_mask is not supported")
    return [None] * num_hidden_layers


class _VocabSize:
    """len() of ram's tokenizer: the BERT vocabulary plus the added [DEC] and [ENC] tokens."""

    def __init__(self, tokenizer_dir: str):
        vocab = Path(tokenizer_dir) / "vocab.txt"
        self.size = len(vocab.read_text(encoding="utf-8").splitlines()) + 2

    def __len__(self) -> int:
        return self.size


def _scores(model: Any, image: Any) -> Any:
    """Sigmoid score of every tag, [B, num_class]; the forward of RAM_plus.generate_tag."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        embeds = model.image_proj(model.visual_encoder(image))
        atts = torch.ones(embeds.shape[:-1], dtype=torch.long, device=image.device)
        cls = embeds[:, 0, :]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        n = model.num_class
        label_embed = model.label_embed.view(n, -1, model.label_embed.shape[-1])  # [n, des, 512]
        # weights of the per-class text descriptions, then their weighted sum
        logits = model.reweight_scale.exp() * torch.einsum("bc,ndc->bnd", cls, label_embed)
        weights = F.softmax(logits, dim=2)
        reweighted = torch.einsum("bnd,ndc->bnc", weights, label_embed)
        queries = F.relu(model.wordvec_proj(reweighted))
        hidden = model.tagging_head(
            encoder_embeds=queries,
            encoder_hidden_states=embeds,
            encoder_attention_mask=atts,
            return_dict=False,
            mode="tagging",
        )
        return torch.sigmoid(model.fc(hidden[0]).squeeze(-1))


def _patch_compat() -> None:
    """Let `ram` (written for transformers 4.25 and fairscale) import with transformers 5.

    Only names that the RAM++ inference path imports but never calls are stubbed; the `ram`
    package itself is not modified.
    """
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu

    for name in ("apply_chunking_to_forward", "prune_linear_layer"):
        if not hasattr(mu, name):
            setattr(mu, name, getattr(pu, name))
    if not hasattr(mu, "find_pruneable_heads_and_indices"):  # used only by prune_heads()

        def _unsupported(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("head pruning is not supported")

        mu.find_pruneable_heads_and_indices = _unsupported
    try:
        import fairscale  # noqa: F401
    except ImportError:  # only the ViT backbone (not used by RAM++ Swin-L) imports it
        mod = types.ModuleType("fairscale.nn.checkpoint.checkpoint_activations")
        mod.checkpoint_wrapper = lambda module, *a, **k: module
        for name in ("fairscale", "fairscale.nn", "fairscale.nn.checkpoint"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules[mod.__name__] = mod
