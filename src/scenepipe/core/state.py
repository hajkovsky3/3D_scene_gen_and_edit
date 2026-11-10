"""Scene states (`state/vNNN/state.json`). Format: docs/data-contracts.md, section 9.

A state only points to files, nothing is copied. `state/current` is a text file holding the
name of the current state (symlinks are unreliable on Windows).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from scenepipe.core.workspace import Workspace

log = logging.getLogger(__name__)

INITIAL = "v000"
_VERSION = re.compile(r"^v\d{3}$")


def write_initial_state(ws: Workspace) -> Path:
    """Create state/v000 from the build outputs (03, 07, 08). Called by the runner after 08.

    A rebuild overwrites v000. `current` is set to v000 unless later states exist, which were
    derived from the previous build; those are kept and a warning is logged."""
    graph = ws.abs("08_scene_graph/scene_graph.json")
    doc = json.loads(graph.read_text(encoding="utf-8"))
    state = {
        "version": INITIAL,
        "parent": None,
        "created_by": "08_scene_graph/manifest.json",
        "gaussians": "03_refinement/gaussians.ply",
        "gaussian_instances": "07_lifting/gaussian_instances.npz",
        "scene_graph": "08_scene_graph/scene_graph.json",
        "node_features": "08_scene_graph/node_features.npy",
        "embedding_model": doc["embedding_model"],
        "cameras": "01_sfm/cameras.json",
    }
    for key in ("created_by", "gaussians", "gaussian_instances", "node_features", "cameras"):
        if not ws.abs(state[key]).is_file():
            raise FileNotFoundError(f"state {INITIAL}: {state[key]} does not exist")
    d = ws.state_dir(INITIAL)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "state.json"
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    later = [v for v in versions(ws) if v != INITIAL]
    if later:
        log.warning("state %s rebuilt; states %s derive from the previous build", INITIAL, later)
    else:
        set_current(ws, INITIAL)
    return path


def versions(ws: Workspace) -> list[str]:
    root = ws.root / "state"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if _VERSION.match(p.name) and p.is_dir())


def read_state(ws: Workspace, version: str) -> dict[str, Any]:
    return json.loads((ws.state_dir(version) / "state.json").read_text(encoding="utf-8"))


def current(ws: Workspace) -> str | None:
    path = ws.root / "state" / "current"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def set_current(ws: Workspace, version: str) -> None:
    (ws.root / "state" / "current").write_text(version + "\n", encoding="utf-8")
