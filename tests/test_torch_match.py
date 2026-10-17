"""PyTorch SIFT matcher (01_sfm, use_gpu) against a plain numpy version of COLMAP's rules."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # environment sp-gs

from scenepipe.phases.p01_sfm.torch_match import match_pairs


def reference(a: np.ndarray, b: np.ndarray, ratio=0.8, max_dist=0.7) -> set[tuple[int, int]]:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    ang = np.arccos(np.clip(a @ b.T, -1, 1))

    def one_way(d):
        d = np.concatenate([d, np.full((len(d), 1), np.pi)], axis=1)  # no 2nd neighbour: pi
        order = np.argsort(d, axis=1)
        best, second = order[:, 0], order[:, 1]
        r = np.arange(len(d))
        ok = (d[r, best] <= max_dist) & (d[r, best] < ratio * d[r, second])
        return {int(i): int(best[i]) for i in np.nonzero(ok)[0]}

    m12, m21 = one_way(ang), one_way(ang.T)
    return {(i, j) for i, j in m12.items() if m21.get(j) == i}


def descriptors(rng, n: int, base: np.ndarray | None = None) -> np.ndarray:
    if base is None:
        return rng.integers(0, 80, (n, 128)).astype(np.uint8)
    noisy = base[:n].astype(int) + rng.integers(-6, 7, (n, 128))
    return np.clip(noisy, 0, 255).astype(np.uint8)


def test_matches_equal_reference():
    rng = np.random.default_rng(0)
    base = descriptors(rng, 400)
    descs = [base, descriptors(rng, 300, base), descriptors(rng, 250), descriptors(rng, 1)]
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    got = {(i, j): m for i, j, m in match_pairs(descs, pairs, device=device)}
    assert set(got) == set(pairs)
    for i, j in pairs:
        assert {tuple(x) for x in got[i, j].tolist()} == reference(
            descs[i].astype(np.float64), descs[j].astype(np.float64)
        )
    assert len(got[0, 1]) > 250  # the noisy copy matches, random descriptors hardly


def test_batches_split_by_memory(monkeypatch):
    """Small batch budget: same result as one batch."""
    from scenepipe.phases.p01_sfm import torch_match

    rng = np.random.default_rng(1)
    base = descriptors(rng, 200)
    descs = [base] + [descriptors(rng, 150 + 10 * k, base) for k in range(5)]
    pairs = [(0, j) for j in range(1, 6)]
    one = {(i, j): m.tolist() for i, j, m in match_pairs(descs, pairs, device="cpu")}
    monkeypatch.setattr(torch_match, "BATCH_BYTES", 200 * 160 * 4 * 2)
    split = {(i, j): m.tolist() for i, j, m in match_pairs(descs, pairs, device="cpu")}
    assert one == split
