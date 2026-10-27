"""COCO RLE encoding of core/masks.py."""

import importlib.util

import numpy as np
import pytest

from scenepipe.core.masks import mask_bbox, rle_area, rle_decode, rle_encode


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip(seed):
    rng = np.random.default_rng(seed)
    h, w = rng.integers(1, 80, 2)
    m = rng.random((h, w)) < rng.random()
    rle = rle_encode(m)
    assert rle["size"] == [h, w] and isinstance(rle["counts"], str)
    assert np.array_equal(rle_decode(rle), m)
    assert rle_area(rle) == m.sum()


def test_large_runs_and_edges():
    m = np.zeros((1037, 1557), bool)
    m[100:900, 200:1400] = True  # long runs need several 5-bit groups
    m[0, 0] = True  # mask starting with a one
    assert np.array_equal(rle_decode(rle_encode(m)), m)
    assert np.array_equal(rle_decode(rle_encode(np.zeros((3, 4), bool))), np.zeros((3, 4), bool))
    assert np.array_equal(rle_decode(rle_encode(np.ones((3, 4), bool))), np.ones((3, 4), bool))


def test_known_encoding():
    # column-major runs: 1 zero, 2 ones, 3 zeros -> counts [1, 2, 3]
    m = np.array([[0, 1, 0], [1, 0, 0]], bool)
    assert rle_encode(m)["counts"] == "123"
    assert rle_decode({"size": [2, 3], "counts": [1, 2, 3]}).tolist() == m.tolist()


@pytest.mark.skipif(importlib.util.find_spec("pycocotools") is None, reason="no pycocotools")
def test_matches_pycocotools():
    from pycocotools import mask as cocomask

    rng = np.random.default_rng(7)
    m = rng.random((60, 90)) < 0.3
    ref = cocomask.encode(np.asfortranarray(m.astype(np.uint8)))
    assert rle_encode(m)["counts"] == ref["counts"].decode()


def test_bbox():
    m = np.zeros((10, 20), bool)
    assert mask_bbox(m) == [0, 0, 0, 0]
    m[2:5, 3:9] = True
    assert mask_bbox(m) == [3, 2, 9, 5]
