"""Files of the manual annotation (format: docs/evaluation.md, Ručná anotácia).

workspace/<scene>/annotations/
  frames.json            {"frames": [frame_id, ...]} the frames chosen for annotation
  frames/<frame_id>.json {"frame_id", "image", "width", "height",
                          "objects": [{"instance": "chair#1", "label": "chair", "rle": {...}}]}
  vocabulary.txt         every object class present in the scene, one per line (also those
                         not drawn in any frame); ground truth for the tag recall of 04
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scenepipe.core.masks import rle_decode, rle_encode
from scenepipe.core.workspace import Workspace

DIR = "annotations"


def label_of(instance: str) -> str:
    """`chair#2` -> `chair` (the class name, lowercase, single spaces)."""
    return " ".join(instance.split("#", 1)[0].lower().split())


@dataclass
class AnnotatedObject:
    instance: str
    label: str
    mask: np.ndarray  # bool [H, W]


class AnnotationStore:
    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.root = ws.root / DIR

    def exists(self) -> bool:
        return (self.root / "frames.json").is_file()

    def frames(self) -> list[str]:
        return json.loads((self.root / "frames.json").read_text("utf-8"))["frames"]

    def set_frames(self, frames: list[str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _write_json(self.root / "frames.json", {"frames": list(frames)})

    def image_rel(self, frame_id: str) -> str:
        return f"01_sfm/images_undistorted/{frame_id}.jpg"

    def frame_path(self, frame_id: str) -> Path:
        return self.root / "frames" / f"{frame_id}.json"

    def load_frame(self, frame_id: str) -> list[AnnotatedObject]:
        path = self.frame_path(frame_id)
        if not path.is_file():
            return []
        data = json.loads(path.read_text("utf-8"))
        return [
            AnnotatedObject(o["instance"], o["label"], rle_decode(o["rle"]).astype(bool))
            for o in data["objects"]
        ]

    def save_frame(
        self, frame_id: str, width: int, height: int, objects: list[AnnotatedObject]
    ) -> None:
        data = {
            "frame_id": frame_id,
            "image": self.image_rel(frame_id),
            "width": width,
            "height": height,
            "objects": [
                {"instance": o.instance, "label": o.label, "rle": rle_encode(o.mask)}
                for o in objects
                if o.mask.any()
            ],
        }
        self.frame_path(frame_id).parent.mkdir(parents=True, exist_ok=True)
        _write_json(self.frame_path(frame_id), data)

    def annotated_frames(self) -> list[str]:
        """Frames with a saved annotation file, in the order of frames.json."""
        return [f for f in self.frames() if self.frame_path(f).is_file()]

    def vocabulary(self) -> list[str]:
        path = self.root / "vocabulary.txt"
        if not path.is_file():
            return []
        lines = (label_of(s) for s in path.read_text("utf-8").splitlines())
        return sorted({s for s in lines if s and not s.startswith("#")})

    def save_vocabulary(self, labels: list[str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        text = "".join(f"{s}\n" for s in sorted({label_of(s) for s in labels if s.strip()}))
        (self.root / "vocabulary.txt").write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
