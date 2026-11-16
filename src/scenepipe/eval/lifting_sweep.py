"""Threshold sweep of phase 07 against the manual annotation (calibration, docs/evaluation.md).

Every variant runs 07 with the given overrides (on top of default.yaml + profile), its output
is copied to 07_lifting/_cmp/sweep/<name>/ and scored with lifting_gt (ground truth) and
lifting_consistency (fidelity to the 05 masks). The 07 output present before the sweep is
restored at the end. Needs CUDA (gsplat). Environment `sp-gs`.

    python -m scenepipe.eval.lifting_sweep --scene m360_counter --profile fast \
        --grid merge_sim=0.65,0.75,0.85 min_coverage=0.1,0.2,0.35
    python -m scenepipe.eval.lifting_sweep --scene m360_counter --one-at-a-time \
        --grid merge_sim=0.65,0.75,0.85 merge_overlap=0.2,0.3,0.5

`--grid` makes the full product of the values; with `--one-at-a-time` every key is varied
alone with the others at their configured values (fewer runs).
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import yaml

from scenepipe.cli import load_config, run_phase
from scenepipe.core.workspace import Workspace
from scenepipe.eval import lifting_consistency, lifting_gt
from scenepipe.eval.gt import REPO_ROOT

PHASE = "07_lifting"


def variants(grid: dict[str, list], one_at_a_time: bool) -> list[dict]:
    if not grid:
        return [{}]
    if one_at_a_time:
        out: list[dict] = [{}]
        for key, values in grid.items():
            out += [{key: v} for v in values]
        return out
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def _name(v: dict) -> str:
    return "_".join(f"{k}={v[k]}" for k in sorted(v)) or "base"


def _copy_out(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("_cmp", "debug"))


def _restore(backup: Path, out: Path) -> None:
    for p in out.iterdir():
        if p.name != "_cmp":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    for p in backup.iterdir():
        dst = out / p.name
        shutil.copytree(p, dst) if p.is_dir() else shutil.copy2(p, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", type=Path, default=REPO_ROOT / "workspace")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--grid", nargs="*", default=[], metavar="KEY=V1,V2")
    ap.add_argument("--one-at-a-time", action="store_true")
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    grid = {}
    for item in args.grid:
        key, _, vals = item.partition("=")
        grid[key] = [yaml.safe_load(v) for v in vals.split(",")]
    ws = Workspace.open(args.workspace, args.scene)
    out = ws.root / PHASE
    sweep = out / "_cmp" / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    backup = sweep / "_before"
    if out.joinpath("instances.json").is_file():
        _copy_out(out, backup)
    base_cfg = load_config(PHASE, args.profile)
    results = []
    try:
        for v in variants(grid, args.one_at_a_time):
            name = _name(v)
            overrides = [f"{k}={json.dumps(val)}" for k, val in v.items()] + ["num_debug=0"]
            run_phase(PHASE, args.scene, args.workspace, args.profile, overrides)
            _copy_out(out, sweep / name)
            rel = f"{PHASE}/_cmp/sweep/{name}"
            gt = lifting_gt.evaluate(ws, rel, args.scale)
            cons = lifting_consistency.evaluate(ws, rel, args.scale)
            row = {
                "variant": name,
                "config": {k: v.get(k, base_cfg.get(k)) for k in grid},
                **{
                    k: gt[k]
                    for k in (
                        "num_instances_without_stuff",
                        "miou_frame",
                        "miou_3d",
                        "recall_3d_at_0.5",
                        "fragments_mean",
                        "label_accuracy_3d",
                    )
                },
                "num_merged": len(gt["merged"]),
                "consistency_mean_iou": cons["mean_iou_all"],
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if backup.is_dir():
            _restore(backup, out)
    (sweep / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"results: {sweep / 'results.json'}")


if __name__ == "__main__":
    main()
