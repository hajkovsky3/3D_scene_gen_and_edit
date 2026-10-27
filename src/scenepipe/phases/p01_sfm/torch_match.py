"""SIFT descriptor matching on the GPU with PyTorch (optional, `use_gpu: true`).

Two matchers: nearest neighbours with COLMAP's rules (`match_pairs`, below) and LightGlue
(`match_pairs_lightglue`, at the end of the file).

Reproduces COLMAP's SIFT brute-force matcher rules (SiftMatchingOptions): descriptors are
compared by the angle between the L2-normalized vectors, a match needs angle <= max_distance
and angle < max_ratio * second best angle, and with cross_check it has to be the mutual best
match. Only the nearest-neighbour search runs here; the geometric verification stays in
COLMAP (`pycolmap.verify_matches`), so the result is used exactly like COLMAP's own matches.

Pairs are processed per anchor image: the anchor's descriptors are compared with a batch
of partner images at once (one batched matmul), which keeps kernel launches few.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator

import numpy as np

BATCH_BYTES = 512 * 2**20  # similarity tensor budget per batch (GTX 1650 has 4 GB)
MASKED = -2.0  # similarity below any cosine: padding and masked best matches


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def match_pairs(
    descriptors: list[np.ndarray],
    pairs: list[tuple[int, int]],
    max_ratio: float = 0.8,
    max_distance: float = 0.7,
    cross_check: bool = True,
    device: str = "cuda",
    on_progress: Callable[[int], None] | None = None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield (i, j, matches uint32[M, 2]) for every pair; matches index rows of descriptors.

    `descriptors[i]` is uint8 or float [N_i, 128]. `on_progress(done_pairs)` is called after
    every batch.
    """
    import torch

    dev = torch.device(device)
    descs = [torch.from_numpy(np.array(d)).to(dev) for d in descriptors]  # writable copy
    by_anchor: dict[int, list[int]] = defaultdict(list)
    for i, j in pairs:
        by_anchor[i].append(j)
    done = 0
    with torch.inference_mode():
        for i in sorted(by_anchor):
            a = _normalized(descs[i])
            partners = by_anchor[i]
            for batch in _batches(partners, len(a), [len(descs[j]) for j in partners]):
                for j, m in zip(
                    batch,
                    _match_batch(
                        a, [descs[j] for j in batch], max_ratio, max_distance, cross_check
                    ),
                    strict=True,
                ):
                    yield i, j, m
                done += len(batch)
                if on_progress is not None:
                    on_progress(done)


def _normalized(d):
    import torch

    x = d.float()
    return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)


def _batches(partners: list[int], n: int, sizes: list[int]) -> Iterator[list[int]]:
    """Split partners so that the [k, n, max_m] float32 similarity tensor fits the budget."""
    batch: list[int] = []
    max_m = 0
    for j, m in zip(partners, sizes, strict=True):
        new_max = max(max_m, m)
        if batch and (len(batch) + 1) * n * new_max * 4 > BATCH_BYTES:
            yield batch
            batch, new_max = [], m
        batch.append(j)
        max_m = new_max
    if batch:
        yield batch


def _match_batch(a, others, max_ratio: float, max_distance: float, cross_check: bool):
    import torch

    n = len(a)
    k = len(others)
    sizes = [len(o) for o in others]
    max_m = max(max(sizes), 2)
    if n < 1 or max_m < 1:
        return [np.zeros((0, 2), np.uint32) for _ in others]
    b = torch.zeros((k, max_m, a.shape[1]), dtype=a.dtype, device=a.device)
    valid = torch.zeros((k, max_m), dtype=torch.bool, device=a.device)
    for t, o in enumerate(others):
        if len(o):
            b[t, : len(o)] = _normalized(o)
            valid[t, : len(o)] = True
    sim = torch.einsum("nd,kmd->knm", a, b)  # cosine similarities [k, n, m]
    sim.masked_fill_(~valid[:, None, :], MASKED)

    # Best and second best along both axes. topk is several times slower than max on the
    # GPU, so the second best is the max after temporarily masking the best (in place).
    v12, j12 = sim.max(dim=2)  # a -> b, [k, n]
    v21, i21 = sim.max(dim=1)  # b -> a, [k, m]
    sim.scatter_(2, j12[..., None], MASKED)
    second12 = sim.max(dim=2).values
    sim.scatter_(2, j12[..., None], v12[..., None])  # restore
    ok12 = _passes(v12, second12, max_ratio, max_distance)
    ok12 &= valid.gather(1, j12)  # padding can win only if the partner has no features
    if cross_check:
        sim.scatter_(1, i21[:, None, :], MASKED)
        second21 = sim.max(dim=1).values
        ok21 = _passes(v21, second21, max_ratio, max_distance) & valid
        back = i21.gather(1, j12)  # [k, n]: best anchor row of the matched column
        ok12 &= ok21.gather(1, j12)
        ok12 &= back == torch.arange(n, device=a.device)[None]
    out = []
    ok_cpu = ok12.cpu().numpy()
    j_cpu = j12.cpu().numpy()
    for t in range(k):
        rows = np.nonzero(ok_cpu[t])[0]
        out.append(np.stack([rows, j_cpu[t, rows]], axis=1).astype(np.uint32))
    return out


def _passes(best, second, max_ratio: float, max_distance: float):
    import torch

    d1 = torch.arccos(best.clamp(-1.0, 1.0))
    d2 = torch.arccos(second.clamp(-1.0, 1.0))
    return (d1 <= max_distance) & (d1 < max_ratio * d2)


# --- SIFT + LightGlue (`matcher_backend: lightglue`) ---
#
# LightGlue (Lindenberger et al., ICCV 2023) with the SIFT weights of cvg/LightGlue (Apache 2.0),
# network from kornia (Apache 2.0). Inputs are prepared as COLMAP's own SIFT_LIGHTGLUE matcher
# does (src/colmap/feature/lightglue.cc): keypoints shifted from COLMAP's pixel-corner origin
# by -0.5, the stored uint8 RootSIFT descriptors L2-normalized, scale and orientation (radians)
# from the keypoint's affine shape. Matches with a score below `min_score` are dropped.

LIGHTGLUE_SIFT_CONF = {"input_dim": 128, "add_scale_ori": True}
CACHE_IMAGES = 64  # images whose tensors are kept on the device


def load_lightglue(weights: str, device: str = "cuda"):
    """LightGlue for SIFT from a local `sift_lightglue.pth` (no download at run time)."""
    import torch
    from kornia.feature import LightGlue

    model = LightGlue(None, **LIGHTGLUE_SIFT_CONF)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    for i in range(model.conf.n_layers):  # old checkpoint layout, renamed like kornia does
        for old, new in ((f"self_attn.{i}", f"transformers.{i}.self_attn"),
                         (f"cross_attn.{i}", f"transformers.{i}.cross_attn")):  # fmt: skip
            state = {k.replace(old, new): v for k, v in state.items()}
    missing = model.load_state_dict(state, strict=False).missing_keys
    params = dict(model.named_parameters())
    if any(name in params for name in missing):
        raise RuntimeError(f"{weights}: LightGlue weights incomplete, missing {missing}")
    return model.eval().to(torch.device(device))


def sift_lightglue_inputs(keypoints: np.ndarray, descriptors: np.ndarray, width: int, height: int):
    """COLMAP keypoints [N, 6] (x, y, a11, a12, a21, a22) + uint8 descriptors -> LightGlue dict.

    Arrays stay on the CPU (numpy); `match_pairs_lightglue` moves them to the device.
    """
    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 6)
    a11, a12, a21, a22 = kp[:, 2], kp[:, 3], kp[:, 4], kp[:, 5]
    desc = np.asarray(descriptors, dtype=np.float32)
    desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
    return {
        "keypoints": kp[:, :2] - 0.5,
        "scales": (np.hypot(a11, a21) + np.hypot(a12, a22)) / 2,
        "oris": np.arctan2(a21, a11),
        "descriptors": desc,
        "image_size": np.array([width, height], dtype=np.float32),
    }


def match_pairs_lightglue(
    features: list[dict],
    pairs: list[tuple[int, int]],
    model,
    min_score: float = 0.1,
    on_progress: Callable[[int], None] | None = None,
    progress_every: int = 20,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield (i, j, matches uint32[M, 2]) for every pair; `features` from `sift_lightglue_inputs`.

    Pairs run one at a time (LightGlue's point pruning keeps a different number of points per
    pair, so batching would not help). Tensors of an image are moved to the device once.
    """
    import torch

    dev = next(model.parameters()).device
    cache: dict[int, dict] = {}

    def tensors(i: int) -> dict:
        if i not in cache:
            if len(cache) >= CACHE_IMAGES:  # pairs are sorted, so the anchor is reloaded rarely
                cache.clear()
            cache[i] = {k: torch.from_numpy(v).to(dev)[None] for k, v in features[i].items()}
        return cache[i]

    done = 0
    with torch.inference_mode():
        for i, j in sorted(pairs):
            a, b = tensors(i), tensors(j)
            if a["keypoints"].shape[1] < 1 or b["keypoints"].shape[1] < 1:
                m = np.zeros((0, 2), np.uint32)
            else:
                out = model({"image0": a, "image1": b})
                m0 = out["matches0"][0]
                ok = (m0 >= 0) & (out["matching_scores0"][0] >= min_score)
                rows = torch.nonzero(ok)[:, 0]
                m = torch.stack([rows, m0[rows]], 1).cpu().numpy().astype(np.uint32)
            yield i, j, m
            done += 1
            if on_progress is not None and (done % progress_every == 0 or done == len(pairs)):
                on_progress(done)
