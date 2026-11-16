"""Command line: python -m scenepipe run <phase> --scene ID [--profile fast] [--set key=value]
and python -m scenepipe build --scene ID [--profile fast] [--from 02_reconstruction] [--to ...].

Config = configs/default.yaml, overlaid by configs/<profile>.yaml (only the keys it sets),
then by --set overrides (values parsed as YAML). The section of a phase is its id without
the number (01_sfm -> sfm).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from scenepipe.core.phase import get_impl
from scenepipe.core.state import write_initial_state
from scenepipe.core.workspace import BUILD_PHASES, EDIT_PHASES, PHASES, Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

log = logging.getLogger("scenepipe")


def section(phase: str) -> str:
    return phase.split("_", 1)[1]


def load_config(phase: str, profile: str | None = None, overrides: list[str] = ()) -> dict:
    """Effective config section of a phase (including `impl`)."""
    cfg = dict(_read_yaml(CONFIG_DIR / "default.yaml").get(section(phase)) or {})
    if profile and profile != "default":
        path = CONFIG_DIR / f"{profile}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"unknown profile {profile!r}: {path} does not exist")
        cfg.update(_read_yaml(path).get(section(phase)) or {})
    for kv in overrides:
        key, sep, value = kv.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {kv!r}")
        cfg[key] = yaml.safe_load(value)
    return cfg


def profiles() -> list[str]:
    """Available profiles: 'default' plus every configs/*.yaml except local files whose
    top-level keys are all phase sections (skips data files such as tag_synonyms.yaml)."""
    sections = {section(p) for p in PHASES}
    names = []
    for path in CONFIG_DIR.glob("*.yaml"):
        if path.stem == "default" or path.stem.startswith("local"):
            continue
        data = _read_yaml(path)
        if isinstance(data, dict) and data and set(data) <= sections:
            names.append(path.stem)
    return ["default", *sorted(names)]


def impl_module(phase: str, impl: str) -> str:
    return f"scenepipe.phases.p{phase}.{impl}"


def is_implemented(phase: str, profile: str | None = None) -> bool:
    """True if the configured implementation of the phase exists (does not import it)."""
    impl = load_config(phase, profile).get("impl")
    if not impl:
        return False
    try:
        return importlib.util.find_spec(impl_module(phase, impl)) is not None
    except ModuleNotFoundError:
        return False


def run_phase(
    phase: str,
    scene: str,
    workspace: Path,
    profile: str | None = None,
    overrides: list[str] = (),
    edit_id: str | None = None,
) -> dict[str, Path]:
    cfg = load_config(phase, profile, overrides)
    impl = cfg.pop("impl")
    importlib.import_module(impl_module(phase, impl))  # registers the implementation
    ws = Workspace.open(workspace, scene)
    t0 = time.perf_counter()
    outs = get_impl(phase, impl)(ws, cfg, edit_id).execute()
    log.info("%s (%s) done in %.0f s", phase, impl, time.perf_counter() - t0)
    if phase == "08_scene_graph":  # the build ends with the initial scene state
        log.info("state written: %s", ws.rel(write_initial_state(ws)))
    return outs


def run_build(
    scene: str,
    workspace: Path,
    profile: str | None = None,
    first: str = BUILD_PHASES[0],
    last: str = BUILD_PHASES[-1],
) -> dict[str, float]:
    """Run build phases first..last one after another in the current Python; each phase also
    logs to workspace/<scene>/logs/<phase>.log. Returns wall time per phase in seconds."""
    phases = BUILD_PHASES[BUILD_PHASES.index(first) : BUILD_PHASES.index(last) + 1]
    if not phases:
        raise ValueError(f"empty phase range {first}..{last}")
    log_dir = Workspace.open(workspace, scene).root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    times: dict[str, float] = {}
    for phase in phases:
        handler = logging.FileHandler(log_dir / f"{phase}.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        t0 = time.perf_counter()
        try:
            run_phase(phase, scene, workspace, profile)
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
        times[phase] = time.perf_counter() - t0
    for phase, s in times.items():
        log.info("build %-20s %7.0f s", phase, s)
    log.info("build total %.0f s (%.1f min)", sum(times.values()), sum(times.values()) / 60)
    return times


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scenepipe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one phase")
    run.add_argument("phase", choices=PHASES)
    run.add_argument("--scene", required=True)
    run.add_argument("--workspace", type=Path, default=REPO_ROOT / "workspace")
    run.add_argument("--profile", default="default", help="configs/<profile>.yaml overlay")
    run.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    run.add_argument("--edit", dest="edit_id", help="edit id (phases 09-12)")
    build = sub.add_parser("build", help="run the build phases 00-08 one after another")
    build.add_argument("--scene", required=True)
    build.add_argument("--workspace", type=Path, default=REPO_ROOT / "workspace")
    build.add_argument("--profile", default="default", help="configs/<profile>.yaml overlay")
    build.add_argument("--from", dest="first", choices=BUILD_PHASES, default=BUILD_PHASES[0])
    build.add_argument("--to", dest="last", choices=BUILD_PHASES, default=BUILD_PHASES[-1])
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr
    )
    if args.cmd == "run" and args.phase in EDIT_PHASES and not args.edit_id:
        ap.error(f"{args.phase} needs --edit")
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        if args.cmd == "build":
            run_build(args.scene, args.workspace, args.profile, args.first, args.last)
        else:
            run_phase(
                args.phase, args.scene, args.workspace, args.profile, args.overrides, args.edit_id
            )
    except KeyboardInterrupt as e:
        log.warning("%s stopped: %s", args.cmd, e)
        return 130
    return 0


class Stopped(KeyboardInterrupt):
    """Raised on SIGTERM so that Phase.execute records the phase as stopped."""


def _on_sigterm(_signum, _frame) -> None:
    raise Stopped("zastavené (SIGTERM)")


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
