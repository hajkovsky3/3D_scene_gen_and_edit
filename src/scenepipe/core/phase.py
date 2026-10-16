"""Phase interface, implementation registry and manifest writing.

Every phase implementation subclasses `Phase`. The contract (input/output files and their
formats) is defined in phases/XX_*.md; this module only enforces the common mechanics.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from scenepipe.core.progress import Progress
from scenepipe.core.workspace import Workspace

_REGISTRY: dict[str, dict[str, type[Phase]]] = {}


def register(cls: type[Phase]) -> type[Phase]:
    """Class decorator: makes an implementation selectable via config `impl`."""
    _REGISTRY.setdefault(cls.phase, {})[cls.impl] = cls
    return cls


def get_impl(phase: str, impl: str) -> type[Phase]:
    try:
        return _REGISTRY[phase][impl]
    except KeyError as e:
        known = sorted(_REGISTRY.get(phase, {}))
        raise KeyError(f"no implementation {impl!r} for phase {phase!r}; known: {known}") from e


class ContractError(RuntimeError):
    """Raised when phase outputs do not satisfy the phase contract."""


class Phase(ABC):
    phase: ClassVar[str]  # e.g. "01_sfm"
    impl: ClassVar[str]  # e.g. "colmap"
    impl_version: ClassVar[str] = "0"
    contract_version: ClassVar[int] = 1
    defaults: ClassVar[dict[str, Any]] = {}  # merged under the given config

    def __init__(self, ws: Workspace, config: dict[str, Any], edit_id: str | None = None):
        self.ws = ws
        self.config = {**self.defaults, **config}
        self.edit_id = edit_id
        self.out_dir = ws.phase_dir(self.phase, edit_id)
        self.progress = Progress(self.out_dir, self.phase, self.impl)

    @abstractmethod
    def inputs(self) -> dict[str, Path]:
        """Named input files/dirs this run reads (absolute paths). Must exist before run."""

    @abstractmethod
    def run(self) -> dict[str, Path]:
        """Do the work, write outputs into self.out_dir, return named outputs (absolute paths)."""

    @abstractmethod
    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        """Check outputs against the contract in phases/XX_*.md; raise ContractError."""

    def execute(self) -> dict[str, Path]:
        """Run the phase end to end: check inputs, run, validate, write manifest.

        Progress is reported in <out_dir>/progress.json (see core/progress.py); `run`
        implementations call self.progress.update()/stage() to refine it.
        """
        ins = self.inputs()
        missing = [k for k, p in ins.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{self.phase}: missing inputs {missing}")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stale = self.out_dir / "manifest.json"  # an old manifest must not look like a result
        stale.unlink(missing_ok=True)
        started = _now()
        self.progress.start()
        try:
            outs = self.run()
            self.progress.update(stage="validácia", message="", force=True)
            self.validate_outputs(outs)
            self._write_manifest(ins, outs, started, _now())
        except BaseException as e:
            self.progress.fail(e)
            raise
        self.progress.finish()
        return outs

    def _write_manifest(self, ins, outs, started, finished) -> None:
        manifest = {
            "phase": self.phase,
            "impl": self.impl,
            "impl_version": self.impl_version,
            "contract_version": self.contract_version,
            "config": self.config,
            "inputs": {k: self._file_entry(p) for k, p in ins.items()},
            "outputs": {k: self._file_entry(p) for k, p in outs.items()},
            "started_at": started,
            "finished_at": finished,
            "git_commit": _git_commit(),
            "host": platform.node(),
        }
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _file_entry(self, path: Path) -> dict[str, str | None]:
        return {"path": self.ws.rel(path), "sha256": _sha256(path) if path.is_file() else None}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
