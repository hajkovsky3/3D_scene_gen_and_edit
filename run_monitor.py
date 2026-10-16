"""Open the pipeline monitor window: double-click this file, or `python run_monitor.py [--scene ID]`.

Works from any directory and with any Python: if it is not the project's environment (.venv),
the script restarts itself there (without a console window) and exits.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_BIN = VENV / ("Scripts" if os.name == "nt" else "bin")


def in_project_env() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def relaunch_in_venv() -> bool:
    """Start this script with the venv's Python; False if the venv does not exist."""
    for name in ("pythonw.exe", "python.exe", "python"):  # pythonw: no console window
        exe = VENV_BIN / name
        if exe.is_file():
            subprocess.Popen([str(exe), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT)
            return True
    return False


def show_error(text: str) -> None:
    print(text, file=sys.stderr)
    try:  # pythonw has no console, so show a dialog as well
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Scenepipe monitor", text)
        root.destroy()
    except (ImportError, RuntimeError) as e:  # no Tk / no display: the stderr text is enough
        print(f"(dialog not shown: {e})", file=sys.stderr)


def main() -> None:
    if not in_project_env():
        if relaunch_in_venv():
            return
        show_error(
            f"Nenašiel som prostredie projektu {VENV}.\n"
            'Vytvor ho a nainštaluj: pip install -e ".[dev,ingest,sfm,monitor]"'
        )
        sys.exit(1)

    os.chdir(ROOT)
    sys.argv = [sys.argv[0], "--workspace", str(ROOT / "workspace"), *sys.argv[1:]]
    try:
        from scenepipe.monitor.__main__ import main as monitor_main
    except ImportError as e:
        show_error(f'Chýba balík: {e}\nNainštaluj: pip install -e ".[monitor]"')
        sys.exit(1)
    monitor_main()


if __name__ == "__main__":
    main()
