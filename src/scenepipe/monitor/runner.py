"""Runs pipeline phases one after another as subprocesses, each in its configured environment.

Which command runs which phase is machine specific: configs/local.yaml (not in git, template
configs/local.example.yaml). Without it every phase runs with the current Python.
Each phase writes its output to workspace/<scene>/logs/<phase>.log; progress itself is read
by the monitor from progress.json as for any other run.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

from scenepipe.cli import CONFIG_DIR, REPO_ROOT
from scenepipe.core.progress import mark_stopped

LOCAL_CONFIG = CONFIG_DIR / "local.yaml"
DEFAULT_RUNNER = {"command": ["{python}", "-m", "scenepipe", "{args}"]}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window per phase on Windows


def console_python() -> str:
    """The current interpreter, but python.exe instead of pythonw.exe (phases log to stdout)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").is_file():
        return str(exe.parent / "python.exe")
    return str(exe)


@dataclass
class Runners:
    runners: dict[str, dict]
    phases: dict[str, str]

    @classmethod
    def load(cls, path: Path = LOCAL_CONFIG) -> Runners:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        data = data or {}
        runners = {"local": DEFAULT_RUNNER, **(data.get("runners") or {})}
        return cls(runners, data.get("phases") or {})

    def spec(self, phase: str) -> dict:
        name = self.phases.get(phase, self.phases.get("default", "local"))
        if name not in self.runners:
            raise KeyError(f"configs/local.yaml: unknown runner {name!r} for {phase}")
        return self.runners[name]

    def command(self, phase: str, args: list[str], key: str = "command") -> list[str] | None:
        template = self.spec(phase).get(key)
        if not template:
            return None
        values = {
            "python": console_python(),
            "args": shlex.join(args),
            "phase": phase,
            "scene": _arg(args, "--scene"),
        }
        out: list[str] = []
        for part in template:
            if part == "{args}":
                out.extend(args)
            else:
                out.append(part.format(**values))
        return out


class PipelineRun:
    """Runs `phases` of `scene` in order in a background thread; stops at the first failure.

    The GUI reads `state`, `current`, `message` and `log_path` (plain attributes).
    """

    def __init__(
        self, scene: str, phases: list[str], profile: str, workspace: Path, runners: Runners
    ):
        self.scene, self.phases, self.profile = scene, phases, profile
        self.workspace, self.runners = workspace, runners
        self.state = "running"  # running | done | failed | stopped
        self.current: str | None = None
        self.message = ""
        self.log_path: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self.run, daemon=True, name=f"pipeline-{scene}")

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def run(self) -> None:
        logs = self.workspace / self.scene / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for i, phase in enumerate(self.phases, start=1):
            if self._stop_requested.is_set():
                break
            self.current = phase
            self.message = f"{i}/{len(self.phases)}"
            self.log_path = logs / f"{phase}.log"
            cmd = self.runners.command(phase, self._args(phase))
            try:
                with open(self.log_path, "w", encoding="utf-8") as log:
                    log.write("$ " + shlex.join(cmd) + "\n\n")
                    log.flush()
                    self._proc = subprocess.Popen(
                        cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
                    )  # fmt: skip
                    code = self._proc.wait()
            except OSError as e:
                self.state, self.message = "failed", f"{phase}: nedá sa spustiť ({e})"
                return
            if self._stop_requested.is_set():
                mark_stopped(self.workspace / self.scene / phase)  # killed before it could
                self.state, self.message = "stopped", f"zastavené počas {phase}"
                return
            if code != 0:
                self.state, self.message = "failed", f"{phase} skončila s kódom {code}"
                return
        if self._stop_requested.is_set():
            self.state, self.message = "stopped", "zastavené"
        else:
            self.state, self.current, self.message = "done", None, "hotovo"

    def stop(self) -> None:
        self._stop_requested.set()
        proc, phase = self._proc, self.current
        if proc and proc.poll() is None:
            proc.terminate()
            stop_cmd = phase and self.runners.command(phase, self._args(phase), key="stop")
            if stop_cmd:  # e.g. WSL: terminating wsl.exe does not stop the Linux process
                subprocess.run(
                    stop_cmd, check=False, capture_output=True, creationflags=NO_WINDOW, timeout=30
                )

    def _args(self, phase: str) -> list[str]:
        args = ["run", phase, "--scene", self.scene, "--profile", self.profile]
        if self.workspace.resolve() != (REPO_ROOT / "workspace").resolve():
            args += ["--workspace", str(self.workspace.resolve())]
        return args


def _arg(args: list[str], name: str) -> str:
    return args[args.index(name) + 1] if name in args else ""
