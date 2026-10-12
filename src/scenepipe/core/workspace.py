"""Workspace layout helpers. Layout is specified in docs/data-contracts.md, section 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PHASES: tuple[str, ...] = (
    "00_ingest",
    "01_sfm",
    "02_reconstruction",
    "03_tagging",
    "04_segmentation",
    "05_mask_embedding",
    "06_lifting",
    "07_scene_graph",
    "08_prompt_parsing",
    "09_grounding",
    "10_asset_generation",
    "11_editing",
    "12_rendering",
)
BUILD_PHASES = PHASES[:8]
EDIT_PHASES = PHASES[8:12]


@dataclass(frozen=True)
class Workspace:
    """Root of one scene: workspace/<scene_id>/."""

    root: Path

    @classmethod
    def open(cls, base: Path, scene_id: str) -> Workspace:
        return cls(Path(base) / scene_id)

    def phase_dir(self, phase: str, edit_id: str | None = None) -> Path:
        """Output directory of a phase. Edit-loop phases live under edits/<edit_id>/."""
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        if phase in EDIT_PHASES:
            if edit_id is None:
                raise ValueError(f"phase {phase} requires edit_id")
            return self.root / "edits" / edit_id / phase
        return self.root / phase

    def state_dir(self, version: str) -> Path:
        return self.root / "state" / version

    def rel(self, path: Path) -> str:
        """Path relative to the workspace root, as stored in JSON outputs and manifests."""
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def abs(self, rel_path: str) -> Path:
        return self.root / rel_path
