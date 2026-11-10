"""Repository-relative paths, and where model weights live.

Configs name weights relative to the repository root ("checkpoints/huggingface", ...). When the
environment variable SCENEPIPE_CHECKPOINTS is set, "checkpoints/..." resolves under that
directory instead: in WSL, reading weights from the Linux disk is much faster than from the
Windows drive (/mnt/c), so `~/sp-gs.sh` can point it at a copy in ~/checkpoints.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS_ENV = "SCENEPIPE_CHECKPOINTS"


def repo_path(path: str | Path) -> Path:
    """Absolute path of a config value given relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def checkpoint_path(path: str | Path) -> Path:
    """Like repo_path, but "checkpoints/..." goes to $SCENEPIPE_CHECKPOINTS when it is set."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = os.environ.get(CHECKPOINTS_ENV)
    if root and p.parts and p.parts[0] == "checkpoints":
        return Path(root).expanduser().joinpath(*p.parts[1:])
    return REPO_ROOT / p
