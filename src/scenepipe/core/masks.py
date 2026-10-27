"""Binary masks as COCO RLE (docs/data-contracts.md, section 7), without pycocotools.

The format is that of the COCO API: {"size": [h, w], "counts": str}. Run lengths go over the
mask in column-major (Fortran) order and start with a run of zeros; the string stores each
count (from the fourth on as the difference to the count two positions back) as 5-bit groups,
lowest first, with 0x20 as the continuation bit and 0x10 as the sign bit of the last group,
offset by 48 to printable ASCII.
"""

from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray) -> dict:
    """bool[H, W] -> COCO compressed RLE."""
    h, w = mask.shape
    flat = np.asarray(mask, bool).ravel(order="F")
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)  # runs start with zeros
    return {"size": [int(h), int(w)], "counts": _counts_to_string(counts)}


def rle_decode(rle: dict) -> np.ndarray:
    """COCO compressed (string) or uncompressed (list) RLE -> bool[H, W]."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = _string_to_counts(counts)
    counts = np.asarray(counts, np.int64)
    if counts.sum() != h * w:
        raise ValueError(f"RLE counts sum to {counts.sum()}, expected {h * w}")
    values = np.arange(len(counts)) % 2 == 1
    flat = np.repeat(values, counts)
    return flat.reshape((w, h)).T


def rle_area(rle: dict) -> int:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = _string_to_counts(counts)
    return int(sum(counts[1::2]))


def mask_bbox(mask: np.ndarray) -> list[int]:
    """[x0, y0, x1, y1] in pixels, x1/y1 exclusive; [0, 0, 0, 0] for an empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return [0, 0, 0, 0]
    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]


def _counts_to_string(counts: list[int]) -> str:
    out = []
    for i, c in enumerate(counts):
        x = c - counts[i - 2] if i > 2 else c
        more = True
        while more:
            g = x & 0x1F
            x >>= 5
            more = x != -1 if g & 0x10 else x != 0
            if more:
                g |= 0x20
            out.append(chr(g + 48))
    return "".join(out)


def _string_to_counts(s: str) -> list[int]:
    counts: list[int] = []
    p = 0
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            g = ord(s[p]) - 48
            x |= (g & 0x1F) << (5 * k)
            more = bool(g & 0x20)
            p += 1
            k += 1
            if not more and g & 0x10:
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts
