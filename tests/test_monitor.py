"""Progress reporting (core/progress.py, Phase.execute) and the monitor's status reader."""

import itertools
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pytest

from scenepipe.core.phase import Phase
from scenepipe.core.progress import Progress, mark_stopped, read_progress
from scenepipe.core.workspace import Workspace
from scenepipe.monitor.status import STALL_AFTER_S, list_scenes, phase_status, scene_status

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


class DummyPhase(Phase):
    phase = "04_tagging"
    impl = "dummy"
    defaults: ClassVar[dict] = {"fail": False}

    def inputs(self) -> dict[str, Path]:
        return {}

    def steps(self) -> list[str]:
        return ["práca"]

    def run(self) -> dict[str, Path]:
        stage = self.progress.stage(0.2, 0.8, "práca")
        stage.update(0.5, "polovica")
        assert read_progress(self.out_dir)["stage"] == "práca"
        assert read_progress(self.out_dir)["steps"] == ["práca", "validácia"]
        if self.config["fail"]:
            raise RuntimeError("boom")
        out = self.out_dir / "tags.json"
        out.write_text("{}", encoding="utf-8")
        return {"tags": out}

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        pass


def test_progress_stage_mapping_and_atomic_write(tmp_path):
    p = Progress(tmp_path, "01_sfm", "colmap")
    p.start()
    st = p.stage(0.1, 0.5, "párovanie")
    st.update(0.5, "dvojica 50 z 100")
    p.update(force=True)
    data = read_progress(tmp_path)
    assert data["stage"] == "párovanie" and data["fraction"] == pytest.approx(0.3)
    assert data["message"] == "dvojica 50 z 100" and data["status"] == "running"
    assert not list(tmp_path.glob("*.tmp"))
    p.fail(ValueError("x"))
    assert read_progress(tmp_path)["status"] == "failed"
    assert read_progress(tmp_path)["error"] == "ValueError: x"


def test_progress_write_failure_never_raises(tmp_path, monkeypatch):
    """A locked progress.json (Windows: open in another process) must not fail the phase."""
    import scenepipe.core.progress as progress_mod

    def locked(*_args):
        raise PermissionError("locked")

    monkeypatch.setattr(progress_mod.os, "replace", locked)
    monkeypatch.setattr(progress_mod, "REPLACE_RETRY_S", 0.0)
    p = Progress(tmp_path, "00_ingest", "basic")
    p.start()
    p.update(0.5, "x", force=True)
    p.finish()
    assert read_progress(tmp_path) is None


def test_execute_reports_done_and_failure(tmp_path):
    ws = Workspace.open(tmp_path, "demo")
    DummyPhase(ws, {}).execute()
    done = read_progress(ws.phase_dir("04_tagging"))
    assert done["status"] == "done" and done["fraction"] == 1.0
    assert phase_status("04_tagging", ws.phase_dir("04_tagging"), NOW)["status"] == "done"

    with pytest.raises(RuntimeError):
        DummyPhase(ws, {"fail": True}).execute()
    d = ws.phase_dir("04_tagging")
    assert not (d / "manifest.json").exists()  # stale manifest removed at start
    st = phase_status("04_tagging", d, NOW)
    assert st["status"] == "failed" and "boom" in st["error"]
    assert st["steps"] == ["práca", "validácia"] and st["step"] == 0  # failed in "práca"


def write_progress(
    d: Path,
    status: str,
    fraction: float,
    started: datetime,
    updated: datetime,
    steps: list[str] | None = None,
):
    d.mkdir(parents=True, exist_ok=True)
    data = {
        "phase": d.name,
        "impl": "x",
        "status": status,
        "stage": "tréning",
        "fraction": fraction,
        "message": "krok 10/100",
        "started_at": started.isoformat(),
        "updated_at": updated.isoformat(),
        "error": None,
    }
    if steps is not None:  # progress.json written before steps existed has no such key
        data["steps"] = steps
    (d / "progress.json").write_text(json.dumps(data), encoding="utf-8")


def test_running_eta_and_stalled(tmp_path):
    d = tmp_path / "02_reconstruction"
    write_progress(d, "running", 0.25, NOW - timedelta(minutes=10), NOW - timedelta(seconds=5))
    st = phase_status("02_reconstruction", d, NOW)
    assert st["status"] == "running" and st["elapsed_s"] == 600
    assert st["eta_s"] == pytest.approx(1800)

    stale = NOW - timedelta(seconds=STALL_AFTER_S + 1)
    write_progress(d, "running", 0.25, NOW - timedelta(hours=1), stale)
    st = phase_status("02_reconstruction", d, NOW)
    assert st["status"] == "stalled" and st["eta_s"] is None
    assert st["steps"] == [] and st["step"] is None


def test_steps_current_index(tmp_path):
    d = tmp_path / "02_reconstruction"
    steps = ["načítanie snímok", "tréning", "export PLY", "validácia"]
    write_progress(d, "running", 0.4, NOW, NOW, steps)
    st = phase_status("02_reconstruction", d, NOW)
    assert st["steps"] == steps and st["step"] == 1
    write_progress(d, "running", 0.4, NOW, NOW, ["iné"])  # stage not among the steps
    assert phase_status("02_reconstruction", d, NOW)["step"] is None


def test_scene_status_old_runs_and_edits(tmp_path):
    root = tmp_path / "room"
    (root / "input").mkdir(parents=True)
    d00 = root / "00_ingest"
    d00.mkdir()
    manifest = {"impl": "basic", "started_at": "2026-10-16T10:00:00+00:00",
                "finished_at": "2026-10-16T10:01:12+00:00"}  # fmt: skip
    (d00 / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d00 / "frames.json").write_text(json.dumps({"frames": [{}] * 3, "rejected": [{}]}), "utf-8")
    write_progress(root / "edits" / "e001" / "12_editing", "running", 0.5, NOW, NOW)

    assert list_scenes(tmp_path) == ["room"]
    st = scene_status(tmp_path, "room", NOW)
    by_id = {p["id"]: p for p in st["phases"]}
    assert st["edit_id"] == "e001" and len(st["phases"]) == 14
    assert by_id["00_ingest"]["status"] == "done" and by_id["00_ingest"]["elapsed_s"] == 72
    assert by_id["01_sfm"]["status"] == "pending"
    assert by_id["12_editing"]["status"] == "running"


def test_window_smoke(tmp_path):
    """The window builds and renders one status update (skipped without a display)."""
    ctk = pytest.importorskip("customtkinter")
    import tkinter as tk

    from scenepipe.monitor.app import MonitorApp

    steps = ["načítanie snímok", "tréning", "export PLY", "validácia"]
    write_progress(tmp_path / "room" / "02_reconstruction", "running", 0.4, NOW, NOW, steps)
    d00 = tmp_path / "room" / "00_ingest"
    d00.mkdir()
    manifest = {"impl": "basic", "started_at": "2026-10-16T10:00:00+00:00",
                "finished_at": "2026-10-16T10:01:12+00:00"}  # fmt: skip
    (d00 / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        ctk.set_appearance_mode("light")
        app = MonitorApp(tmp_path, "room")
    except tk.TclError:
        pytest.skip("no display")
    try:
        app._poll(reschedule=False)
        app.update()
        assert app.scene == "room"
        row = app.rows["02_reconstruction"]
        assert row.badge.cget("text").startswith("bez aktualizácie")
        marks = [label.cget("text")[0] for label in row.step_labels]
        assert marks == ["✓", "▸", "○", "○"]
        done = app.rows["00_ingest"]
        assert done.badge.cget("text") == "1 min 12 s" and done.times.cget("text") == ""
        app._focus("02_reconstruction")
        # a finished phase without its own steps and a phase that never ran: planned steps
        app._focus("00_ingest")
        app.update()
        assert done.expanded and not row.expanded
        marks = [label.cget("text")[0] for label in done.step_labels]
        assert marks and set(marks) == {"✓"}
        app._focus("01_sfm")
        app.update()
        marks = [label.cget("text")[0] for label in app.rows["01_sfm"].step_labels]
        assert marks and set(marks) == {"○"}
        app._focus("01_sfm")  # second click collapses
        assert app.selected is None and not app.rows["01_sfm"].expanded
    finally:
        app.destroy()


def test_graph_edges_are_orthogonal_with_45_degree_corners():
    pytest.importorskip("tkinter")  # the monitor module imports tkinter (absent in WSL envs)
    from scenepipe.monitor.app import CORNER, EDGES, NH, NODES, NW, chamfer

    def border(nid, x, y):
        n = NODES[nid]
        w = n.get("w", NW)
        x0, x1, y0, y1 = n["x"] - w / 2, n["x"] + w / 2, n["y"] - NH / 2, n["y"] + NH / 2
        on_x = x in (x0, x1) and y0 <= y <= y1
        on_y = y in (y0, y1) and x0 <= x <= x1
        return on_x or on_y

    for a, b, route, _dashed in EDGES:
        assert border(a, *route[0]) and border(b, *route[-1]), (a, b)
        for (x0, y0), (x1, y1) in itertools.pairwise(route):
            assert x0 == x1 or y0 == y1, (a, b)
        line = chamfer(route, CORNER)
        for (x0, y0), (x1, y1) in itertools.pairwise(line):
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            assert dx < 1e-9 or dy < 1e-9 or abs(dx - dy) < 1e-9, (a, b)


def test_planned_steps(tmp_path):
    from scenepipe.monitor import status

    status._PLANNED.clear()
    steps, note = status.planned_steps(tmp_path, "00_ingest", "default")
    assert steps[-1] == "validácia" and "default" in note
    steps, note = status.planned_steps(tmp_path, "09_prompt_parsing", "default")
    assert steps == [] and "nie je implementovaná" in note
    # implementation not importable here: steps of the latest run of the same impl
    write_progress(tmp_path / "a" / "04_tagging", "done", 1.0, NOW, NOW, ["x", "validácia"])
    assert status._last_run_steps(tmp_path, "04_tagging", "x") == (["x", "validácia"], "a")
    assert status._last_run_steps(tmp_path, "04_tagging", "ram_plus") is None
    status._PLANNED.clear()


def test_stop_is_recorded(tmp_path):
    ws = Workspace.open(tmp_path, "demo")

    class Interrupted(DummyPhase):
        def run(self):
            raise KeyboardInterrupt("zastavené (SIGTERM)")

    with pytest.raises(KeyboardInterrupt):
        Interrupted(ws, {}).execute()
    st = phase_status("04_tagging", ws.phase_dir("04_tagging"), NOW)
    assert st["status"] == "stopped" and st["error"] == "zastavené (SIGTERM)"

    d = tmp_path / "x" / "02_reconstruction"  # killed from outside: still "running" on disk
    write_progress(d, "running", 0.3, NOW, NOW)
    mark_stopped(d)
    assert read_progress(d)["status"] == "stopped"


def test_cli_config_profiles_and_implemented():
    from scenepipe.cli import is_implemented, load_config, profiles

    assert profiles()[0] == "default" and "fast" in profiles()
    assert "tag_synonyms" not in profiles()  # data file, not a profile
    assert load_config("01_sfm")["matcher"] == "auto"
    assert load_config("01_sfm", "fast")["matcher"] == "retrieval"
    assert load_config("01_sfm", "fast", ["matcher=exhaustive"])["matcher"] == "exhaustive"
    assert is_implemented("00_ingest") and not is_implemented("10_grounding")


def test_runner_commands_and_pipeline(tmp_path):
    import sys

    from scenepipe.monitor.runner import PipelineRun, Runners

    runners = Runners(
        {
            "local": {"command": ["{python}", "-m", "scenepipe", "{args}"]},
            "wsl": {
                "command": ["wsl", "bash", "-lc", "python -m scenepipe {args}"],
                "stop": ["wsl", "pkill", "-f", "scenepipe run {phase} --scene {scene}"],
            },
        },
        {"02_reconstruction": "wsl"},
    )
    args = ["run", "02_reconstruction", "--scene", "room"]
    assert runners.command("00_ingest", args) == [sys.executable, "-m", "scenepipe", *args]
    assert runners.command("02_reconstruction", args)[-1] == (
        "python -m scenepipe run 02_reconstruction --scene room"
    )
    assert runners.command("02_reconstruction", args, key="stop")[-1] == (
        "scenepipe run 02_reconstruction --scene room"
    )
    assert runners.command("00_ingest", args, key="stop") is None

    def run_with(code: str, phases: list[str]) -> PipelineRun:
        fake = Runners({"local": {"command": ["{python}", "-c", code]}}, {})
        run = PipelineRun("room", phases, "fast", tmp_path, fake)
        run.start()
        return run

    ok = run_with("print('hi')", ["00_ingest", "01_sfm"])
    ok.join(30)
    assert ok.state == "done" and (tmp_path / "room" / "logs" / "01_sfm.log").is_file()
    bad = run_with("import sys; sys.exit(3)", ["00_ingest", "01_sfm"])
    bad.join(30)
    assert bad.state == "failed" and bad.current == "00_ingest" and "3" in bad.message
    slow = run_with("import time; time.sleep(30)", ["00_ingest"])
    for _ in range(100):
        if slow._proc is not None:
            break
        time.sleep(0.05)
    slow.stop()
    slow.join(30)
    assert slow.state == "stopped"


def test_pipeline_runs_independent_phases_in_parallel(tmp_path):
    from scenepipe.monitor.runner import PHASE_DEPS, PipelineRun, Runners

    # every dependency is an earlier build phase (the scheduler never waits forever)
    order = list(PHASE_DEPS)
    assert all(order.index(d) < order.index(p) for p, deps in PHASE_DEPS.items() for d in deps)

    fake = Runners({"local": {"command": ["{python}", "-c", "import time; time.sleep(1.5)"]}}, {})
    phases = ["00_ingest", "01_sfm", "04_tagging"]
    run = PipelineRun("room", phases, "fast", tmp_path, fake, max_parallel=2)
    assert run.ready(phases, set()) == ["00_ingest"]
    assert run.ready(["01_sfm", "04_tagging"], {"00_ingest"}) == ["01_sfm", "04_tagging"]
    # a dependency outside the run counts as done (its outputs are on disk)
    assert run.ready(["05_segmentation"], set()) == []
    assert PipelineRun("r", ["05_segmentation"], "fast", tmp_path, fake).ready(
        ["05_segmentation"], set()
    ) == ["05_segmentation"]
    seen = set()
    run.start()
    for _ in range(200):
        seen.add(tuple(sorted(run.active)))
        if run.state != "running":
            break
        time.sleep(0.05)
    run.join(30)
    assert run.state == "done" and sorted(run.done) == sorted(phases)
    assert ("01_sfm", "04_tagging") in seen  # both ran at once

    one = PipelineRun("room", phases, "fast", tmp_path, fake, max_parallel=1)
    one.start()
    for _ in range(200):
        assert len(one.active) <= 1
        if one.state != "running":
            break
        time.sleep(0.05)
    one.join(30)
    assert one.state == "done"
