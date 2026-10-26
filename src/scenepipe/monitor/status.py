"""Scene status for the monitor: reads progress.json / manifest.json from disk.

Read-only. Works for phases running anywhere (another Python env, WSL) as long as they write
into the same workspace directory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scenepipe.core.progress import read_progress
from scenepipe.core.workspace import EDIT_PHASES, PHASES

STALL_AFTER_S = 600  # a running phase without a progress update for this long is "stalled"

NAMES = {
    "00_ingest": "Ingest",
    "01_sfm": "SfM",
    "02_reconstruction": "Rekonštrukcia 3DGS",
    "03_refinement": "Vyčistenie gaussiánov",
    "04_tagging": "Tagovanie",
    "05_segmentation": "Segmentácia",
    "06_mask_embedding": "Embeddingy masiek",
    "07_lifting": "Lifting",
    "08_scene_graph": "Scene graph",
    "09_prompt_parsing": "Parsovanie promptu",
    "10_grounding": "Grounding",
    "11_asset_generation": "Generovanie objektu",
    "12_editing": "Úprava",
    "13_rendering": "Renderovanie",
}


def list_scenes(workspace: Path) -> list[str]:
    """Scene directories, most recently modified first."""
    if not workspace.is_dir():
        return []
    scenes = []
    for d in workspace.iterdir():
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        if (d / "input").is_dir() or any((d / p).is_dir() for p in PHASES):
            scenes.append((_mtime(d), d.name))
    return [name for _, name in sorted(scenes, reverse=True)]


def latest_edit(scene_dir: Path) -> str | None:
    edits = scene_dir / "edits"
    if not edits.is_dir():
        return None
    dirs = [d for d in edits.iterdir() if d.is_dir()]
    return max(dirs, key=_mtime).name if dirs else None


def scene_status(workspace: Path, scene: str, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    root = workspace / scene
    edit_id = latest_edit(root)
    phases = []
    for phase in PHASES:
        if phase in EDIT_PHASES:
            d = root / "edits" / edit_id / phase if edit_id else None
        else:
            d = root / phase
        phases.append(phase_status(phase, d, now))
    return {
        "scene": scene,
        "edit_id": edit_id,
        "now": now.isoformat(timespec="seconds"),
        "phases": phases,
    }


def phase_status(phase: str, d: Path | None, now: datetime) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": phase,
        "num": phase[:2],
        "name": NAMES[phase],
        "status": "pending",
        "impl": None,
        "fraction": 0.0,
        "stage": "",
        "message": "",
        "steps": [],
        "step": None,  # index of `stage` in `steps`, None if not listed
        "elapsed_s": None,
        "eta_s": None,
        "idle_s": None,
        "error": None,
    }
    if d is None or not d.is_dir():
        return out
    prog = read_progress(d)
    manifest = _read_json(d / "manifest.json")
    if prog:
        out.update(
            impl=prog.get("impl"),
            fraction=float(prog.get("fraction") or 0.0),
            stage=prog.get("stage") or "",
            message=prog.get("message") or "",
            error=prog.get("error"),
            steps=list(prog.get("steps") or []),
        )
        if out["stage"] in out["steps"]:
            out["step"] = out["steps"].index(out["stage"])
    if prog and prog.get("status") == "running":
        started, updated = _ts(prog.get("started_at")), _ts(prog.get("updated_at"))
        idle = (now - updated).total_seconds() if updated else None
        out["status"] = "stalled" if idle is not None and idle > STALL_AFTER_S else "running"
        out["idle_s"] = idle
        if started:
            elapsed = (now - started).total_seconds()
            out["elapsed_s"] = elapsed
            f = out["fraction"]
            if out["status"] == "running" and f >= 0.02:
                out["eta_s"] = elapsed * (1 - f) / f
    elif prog and prog.get("status") in ("failed", "stopped"):
        out["status"] = prog["status"]
        out["elapsed_s"] = _span(prog.get("started_at"), prog.get("updated_at"))
    elif manifest:
        out.update(status="done", fraction=1.0, stage="hotovo", message="")
        out["impl"] = out["impl"] or manifest.get("impl")
        out["elapsed_s"] = _span(manifest.get("started_at"), manifest.get("finished_at"))
    return out


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        t = datetime.fromisoformat(value)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _span(a: str | None, b: str | None) -> float | None:
    ta, tb = _ts(a), _ts(b)
    return (tb - ta).total_seconds() if ta and tb else None


def _mtime(d: Path) -> float:
    try:
        return max([d.stat().st_mtime] + [c.stat().st_mtime for c in d.iterdir()])
    except OSError:
        return 0.0


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}


def input_summary(scene_dir: Path) -> dict[str, int]:
    """Counts and total size of the scene's input files (as accepted by phase 00)."""
    out = {"videos": 0, "photos": 0, "bytes": 0}
    inp = scene_dir / "input"
    if not inp.is_dir():
        return out
    for f in inp.rglob("*"):
        ext = f.suffix.lower()
        if not f.is_file() or ext not in VIDEO_EXTS | PHOTO_EXTS:
            continue
        out["videos" if ext in VIDEO_EXTS else "photos"] += 1
        out["bytes"] += f.stat().st_size
    return out
