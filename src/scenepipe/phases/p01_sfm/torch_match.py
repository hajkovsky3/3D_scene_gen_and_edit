"""SIFT descriptor matching on the GPU with PyTorch (optional, `use_gpu: true`).

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
