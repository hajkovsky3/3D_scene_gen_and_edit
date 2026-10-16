"""Pipeline monitor window: python -m scenepipe.monitor [--scene ID] [--workspace DIR]."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline monitor (desktop window)")
    ap.add_argument("--workspace", type=Path, default=Path("workspace"))
    ap.add_argument("--scene", help="scene shown first (default: most recently modified)")
    args = ap.parse_args()

    from scenepipe.monitor.app import run  # imports customtkinter (extra "monitor")

    run(args.workspace, args.scene)


if __name__ == "__main__":
    main()
