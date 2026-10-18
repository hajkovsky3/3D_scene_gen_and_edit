"""Loading frames as uint8 RGB arrays (shared by phases that compare renders with images)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path, scale: float = 1.0, size: tuple[int, int] | None = None) -> np.ndarray:
    """uint8 [H,W,3]; resized by `scale`, or to exactly `size` = (W, H) when given."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is None and scale != 1.0:
            size = (max(round(im.width * scale), 1), max(round(im.height * scale), 1))
        if size is not None and tuple(size) != im.size:
            down = size[0] < im.width
            im = im.resize(tuple(size), Image.Resampling.BOX if down else Image.Resampling.BICUBIC)
        return np.asarray(im).copy()
