"""Runs pipeline phases as subprocesses, each in its configured environment; independent
phases (by PHASE_DEPS) run side by side.

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
import time
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
    max_parallel: int = 1  # phases running at once (local.yaml `max_parallel`)

    @classmethod
    def load(cls, path: Path = LOCAL_CONFIG) -> Runners:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        data = data or {}
        runners = {"local": DEFAULT_RUNNER, **(data.get("runners") or {})}
        return cls(runners, data.get("phases") or {}, int(data.get("max_parallel") or 1))

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


# Build phases a phase reads from (its `inputs()`); same DAG as phases/README.md. A phase
# starts as soon as those of them that are part of the run have finished.
PHASE_DEPS: dict[str, tuple[str, ...]] = {
    "00_ingest": (),
    "01_sfm": ("00_ingest",),
    "02_reconstruction": ("01_sfm",),
    "03_refinement": ("01_sfm", "02_reconstruction"),
    "04_tagging": ("00_ingest",),
    "05_segmentation": ("01_sfm", "04_tagging"),
    "06_mask_embedding": ("01_sfm", "04_tagging", "05_segmentation"),
    "07_lifting": ("01_sfm", "03_refinement", "05_segmentation", "06_mask_embedding"),
    "08_scene_graph": (
        "01_sfm", "03_refinement", "04_tagging", "05_segmentation", "07_lifting",
    ),
}  # fmt: skip
POLL_S = 0.2


class PipelineRun:
    """Runs `phases` of `scene` in a background thread, each phase as a subprocess.

    Independent phases run side by side (at most `max_parallel` at once, configs/local.yaml):
    e.g. 04 -> 05 -> 06 next to 02 -> 03. After a failure no new phase starts; the ones
    already running finish. The GUI reads `state`, `active`, `current`, `message` and
    `log_path` (plain attributes).
    """

    def __init__(
        self,
        scene: str,
        phases: list[str],
        profile: str,
        workspace: Path,
        runners: Runners,
        max_parallel: int | None = None,
    ):
        self.scene, self.phases, self.profile = scene, phases, profile
        self.workspace, self.runners = workspace, runners
        self.max_parallel = max(int(max_parallel or runners.max_parallel), 1)
        self.state = "running"  # running | done | failed | stopped
        self.active: list[str] = []  # phases running now, in start order
        self.current: str | None = None  # the first of them (or the failed phase)
        self.done: list[str] = []
        self.message = ""
        self.log_path: Path | None = None
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self.run, daemon=True, name=f"pipeline-{scene}")

    @property
    def _proc(self) -> subprocess.Popen | None:  # the first running process (tests, stop)
        with self._lock:
            return next(iter(self._procs.values()), None)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def ready(self, pending: list[str], finished: set[str]) -> list[str]:
        """Pending phases whose dependencies within this run have all finished."""
        in_run = set(self.phases)
        return [
            p for p in pending
            if all(d in finished or d not in in_run for d in PHASE_DEPS.get(p, ()))
        ]  # fmt: skip

    def run(self) -> None:
        logs = self.workspace / self.scene / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        pending, finished, failed = list(self.phases), set(), None
        logs_open: dict[str, object] = {}
        while pending or self._procs:
            # start what can start
            while (
                failed is None and not self._stop_requested.is_set()
                and len(self._procs) < self.max_parallel
            ):  # fmt: skip
                ready = self.ready(pending, finished)
                if not ready:
                    break
                phase = ready[0]
                pending.remove(phase)
                self.log_path = logs / f"{phase}.log"
                cmd = self.runners.command(phase, self._args(phase))
                try:
                    log = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
                    log.write("$ " + shlex.join(cmd) + "\n\n")
                    log.flush()
                    proc = subprocess.Popen(
                        cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
                    )  # fmt: skip
                except OSError as e:
                    failed = (phase, f"{phase}: nedá sa spustiť ({e})")
                    break
                with self._lock:
                    self._procs[phase] = proc
                logs_open[phase] = log
                self._update_active()
            if not self._procs:
                break  # nothing running and nothing can start (stop, failure or done)
            time.sleep(POLL_S)
            for phase, proc in list(self._procs.items()):
                code = proc.poll()
                if code is None:
                    continue
                with self._lock:
                    del self._procs[phase]
                logs_open.pop(phase).close()
                if self._stop_requested.is_set():
                    mark_stopped(self.workspace / self.scene / phase)  # killed before it could
                elif code != 0:
                    failed = failed or (phase, f"{phase} skončila s kódom {code}")
                else:
                    finished.add(phase)
                    self.done.append(phase)
                self._update_active()
        if self._stop_requested.is_set():
            self.state, self.message = "stopped", "zastavené"
        elif failed:
            self.current = failed[0]
            self.state, self.message = "failed", failed[1]
        elif pending:  # cannot happen with a valid DAG, but never report "done" falsely
            self.state, self.message = "failed", f"nespustené: {', '.join(pending)}"
        else:
            self.state, self.current, self.message = "done", None, "hotovo"

    def _update_active(self) -> None:
        self.active = list(self._procs)
        self.current = self.active[0] if self.active else self.current
        self.message = f"{len(self.done)}/{len(self.phases)} hotových"

    def stop(self) -> None:
        self._stop_requested.set()
        with self._lock:
            procs = dict(self._procs)
        for phase, proc in procs.items():
            if proc.poll() is not None:
                continue
            proc.terminate()
            stop_cmd = self.runners.command(phase, self._args(phase), key="stop")
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
