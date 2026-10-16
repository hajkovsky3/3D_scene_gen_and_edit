"""Pair selection for the `retrieval` matcher: temporal neighbours + bag-of-words retrieval.

The visual vocabulary is built on the fly from the SIFT descriptors of the scene itself
(k-means in numpy), so no pretrained vocabulary tree has to be downloaded.
"""

from __future__ import annotations

import numpy as np

KMEANS_ITERS = 10
ASSIGN_CHUNK = 65_536


def root_sift(desc: np.ndarray) -> np.ndarray:
    """RootSIFT: L1-normalize, sqrt (Arandjelovic & Zisserman 2012); rows have unit L2 norm."""
    d = desc.astype(np.float32)
    d /= np.abs(d).sum(axis=1, keepdims=True) + 1e-7
    return np.sqrt(d)


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = KMEANS_ITERS) -> np.ndarray:
    """Plain Lloyd k-means, initialized with random distinct samples."""
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        labels = assign(x, centers)
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, x)
        counts = np.bincount(labels, minlength=k)
        empty = counts == 0
        centers[~empty] = sums[~empty] / counts[~empty, None]
        if empty.any():  # re-seed empty clusters
            centers[empty] = x[rng.choice(len(x), int(empty.sum()), replace=False)]
    return centers


def assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest center for every row (squared L2, chunked)."""
    c2 = (centers**2).sum(axis=1)
    out = np.empty(len(x), dtype=np.int64)
    for i in range(0, len(x), ASSIGN_CHUNK):
        xi = x[i : i + ASSIGN_CHUNK]
        out[i : i + ASSIGN_CHUNK] = np.argmin(c2[None] - 2 * xi @ centers.T, axis=1)
    return out


def bow_similarity(
    descriptors: list[np.ndarray], num_words: int, sample: int, seed: int = 0
) -> np.ndarray:
    """[n, n] cosine similarity of tf-idf bag-of-words vectors (diagonal set to -1)."""
    rng = np.random.default_rng(seed)
    feats = [root_sift(d) if len(d) else np.zeros((0, 128), np.float32) for d in descriptors]
    pool = np.concatenate([f for f in feats if len(f)], axis=0)
    if len(pool) > sample:
        pool = pool[rng.choice(len(pool), sample, replace=False)]
    centers = kmeans(pool, num_words, seed)
    k = len(centers)
    hist = np.zeros((len(feats), k), dtype=np.float32)
    for i, f in enumerate(feats):
        if len(f):
            hist[i] = np.bincount(assign(f, centers), minlength=k)
    df = (hist > 0).sum(axis=0)
    idf = np.log(len(feats) / np.maximum(df, 1)).astype(np.float32)
    v = (hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)) * idf
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    sim = v @ v.T
    np.fill_diagonal(sim, -1.0)
    return sim


def select_pairs(sim: np.ndarray | None, n: int, overlap: int, num_neighbors: int) -> list[tuple]:
    """Sorted unique (i, j), i < j: temporal window + quadratic offsets + top-k retrieval.

    `overlap`: every frame is paired with the next `overlap` frames and with frames at
    offsets 2^k (k >= log2(overlap)), like COLMAP's quadratic sequential overlap.
    `num_neighbors`: the most similar frames by `sim`, outside the temporal window.
    """
    pairs: set[tuple[int, int]] = set()
    offsets = set(range(1, overlap + 1))
    step = 1
    while step < n:
        if step > overlap:
            offsets.add(step)
        step *= 2
    for i in range(n):
        for o in offsets:
            if i + o < n:
                pairs.add((i, i + o))
    if sim is not None and num_neighbors > 0:
        idx = np.arange(n)
        for i in range(n):
            far = idx[np.abs(idx - i) > overlap]
            best = far[np.argsort(-sim[i, far], kind="stable")[:num_neighbors]]
            pairs.update((min(i, int(j)), max(i, int(j))) for j in best)
    return sorted(pairs)
