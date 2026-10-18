"""Reading `cameras.json`. Format: docs/data-contracts.md, sections 4 and 6."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Camera:
    """PINHOLE camera, OpenCV convention, COLMAP pixel centers (+0.5)."""

    frame_id: str
    image: str  # path relative to the workspace root
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray  # float64[4,4]

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float64)

    @property
    def center(self) -> np.ndarray:
        r, t = self.world_to_camera[:3, :3], self.world_to_camera[:3, 3]
        return -r.T @ t

    def scaled(self, width: int, height: int) -> Camera:
        """Same camera for the image resized to width x height."""
        sx, sy = width / self.width, height / self.height
        return Camera(
            self.frame_id,
            self.image,
            width,
            height,
            self.fx * sx,
            self.fy * sy,
            self.cx * sx,
            self.cy * sy,
            self.world_to_camera,
        )


def load_cameras(path: Path) -> list[Camera]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("convention") != "opencv_world_to_camera":
        raise ValueError(f"{path}: unsupported convention {data.get('convention')!r}")
    return [
        Camera(
            frame_id=c["frame_id"],
            image=c["image"],
            width=int(c["width"]),
            height=int(c["height"]),
            fx=float(c["fx"]),
            fy=float(c["fy"]),
            cx=float(c["cx"]),
            cy=float(c["cy"]),
            world_to_camera=np.asarray(c["world_to_camera"], np.float64),
        )
        for c in data["cameras"]
    ]


def scene_scale(cams: list[Camera]) -> float:
    """Max camera distance from the mean camera center, x1.1 (gsplat convention)."""
    centers = np.stack([c.center for c in cams])
    return float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1) or 1.0
