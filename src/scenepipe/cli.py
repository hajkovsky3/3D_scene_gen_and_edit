"""Command line: python -m scenepipe run <phase> --scene ID [--profile fast] [--set key=value].

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
from scenepipe.core.workspace import EDIT_PHASES, PHASES, Workspace

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
    """Available profiles: 'default' plus every configs/*.yaml except local files."""
    names = [p.stem for p in CONFIG_DIR.glob("*.yaml")]
    return ["default"] + sorted(n for n in names if n != "default" and not n.startswith("local"))


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
    return outs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scenepipe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one phase")
    run.add_argument("phase", choices=PHASES)
    run.add_argument("--scene", required=True)
    run.add_argument("--workspace", type=Path, default=REPO_ROOT / "workspace")
    run.add_argument("--profile", default="default", help="configs/<profile>.yaml overlay")
    run.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VAL")
    run.add_argument("--edit", dest="edit_id", help="edit id (phases 08-11)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr
    )
    if args.phase in EDIT_PHASES and not args.edit_id:
        ap.error(f"{args.phase} needs --edit")
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        run_phase(
            args.phase, args.scene, args.workspace, args.profile, args.overrides, args.edit_id
        )
    except KeyboardInterrupt as e:
        log.warning("%s stopped: %s", args.phase, e)
        return 130
    return 0


class Stopped(KeyboardInterrupt):
    """Raised on SIGTERM so that Phase.execute records the phase as stopped."""


def _on_sigterm(_signum, _frame) -> None:
    raise Stopped("zastavené (SIGTERM)")


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
