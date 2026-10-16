"""Progress reporting: every running phase keeps <phase_dir>/progress.json up to date.

The file is for monitoring only (scenepipe.monitor); it is not part of any phase contract.
Format: docs/data-contracts.md, section "Priebeh fázy (progress.json)".
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROGRESS_FILE = "progress.json"
MIN_WRITE_INTERVAL_S = 1.0
REPLACE_RETRIES = 5
REPLACE_RETRY_S = 0.05

log = logging.getLogger(__name__)


class Progress:
    """Writes progress.json atomically; frequent updates are throttled to ~1 per second."""

    def __init__(self, out_dir: Path, phase: str, impl: str):
        self.path = Path(out_dir) / PROGRESS_FILE
        self.data: dict[str, Any] = {
            "phase": phase,
            "impl": impl,
            "status": "running",
            "stage": "",
            "fraction": 0.0,
            "message": "",
            "started_at": _now(),
            "updated_at": _now(),
            "error": None,
        }
        self._last_write = 0.0

    def start(self) -> None:
        self._write()

    def update(
        self,
        fraction: float | None = None,
        stage: str | None = None,
        message: str | None = None,
        force: bool = False,
    ) -> None:
        """Set the overall fraction (0..1) of the phase, the current stage and a short message."""
        changed_stage = stage is not None and stage != self.data["stage"]
        if fraction is not None:
            self.data["fraction"] = round(min(max(float(fraction), 0.0), 1.0), 4)
        if stage is not None:
            self.data["stage"] = stage
        if message is not None:
            self.data["message"] = message
        if force or changed_stage or time.monotonic() - self._last_write >= MIN_WRITE_INTERVAL_S:
            self._write()

    def stage(self, lo: float, hi: float, name: str) -> StageProgress:
        """Sub-reporter mapping its own 0..1 range onto [lo, hi] of the phase."""
        self.update(lo, name, "")
        return StageProgress(self, lo, hi, name)

    def finish(self) -> None:
        self.data.update(status="done", fraction=1.0, stage="hotovo", message="")
        self._write()

    def fail(self, error: BaseException) -> None:
        if isinstance(error, KeyboardInterrupt):  # Ctrl+C or SIGTERM (see cli.py)
            self.data.update(status="stopped", error=str(error) or "zastavené")
        else:
            self.data.update(status="failed", error=f"{type(error).__name__}: {error}"[:500])
        self._write()

    def _write(self) -> None:
        """Best effort: progress is only for monitoring and must never fail the phase.

        On Windows the rename fails while another process (the monitor, an antivirus scan)
        has progress.json open, so it is retried briefly and otherwise skipped.
        """
        self.data["updated_at"] = _now()
        tmp = self.path.with_suffix(".json.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            for attempt in range(REPLACE_RETRIES):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if attempt == REPLACE_RETRIES - 1:
                        raise
                    time.sleep(REPLACE_RETRY_S)
        except OSError as e:
            log.debug("progress.json not written: %s", e)
        self._last_write = time.monotonic()


class StageProgress:
    def __init__(self, parent: Progress, lo: float, hi: float, name: str):
        self.parent, self.lo, self.hi, self.name = parent, lo, hi, name

    def update(self, fraction: float, message: str | None = None) -> None:
        f = min(max(fraction, 0.0), 1.0)
        self.parent.update(self.lo + (self.hi - self.lo) * f, self.name, message)


def mark_stopped(phase_dir: Path, reason: str = "zastavené používateľom") -> None:
    """Mark a still 'running' progress.json as stopped (its process was killed from outside)."""
    data = read_progress(phase_dir)
    if not data or data.get("status") != "running":
        return
    p = Progress(phase_dir, data.get("phase", ""), data.get("impl", ""))
    p.data.update(data)
    p.data.update(status="stopped", error=reason)
    p._write()


def read_progress(phase_dir: Path) -> dict[str, Any] | None:
    path = Path(phase_dir) / PROGRESS_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
