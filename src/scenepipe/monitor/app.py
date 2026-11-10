"""Desktop window of the pipeline monitor (CustomTkinter): phase graph + progress of each phase.

Reads workspace files only (via status.py), polls every POLL_MS.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from typing import Any

import customtkinter as ctk

from scenepipe.core.workspace import BUILD_PHASES, PHASES
from scenepipe.monitor.run_panel import RunPanel
from scenepipe.monitor.runner import PipelineRun, Runners
from scenepipe.monitor.status import list_scenes, planned_steps, scene_status
from scenepipe.monitor.theme import C

POLL_MS = 1500
SCENES_EVERY_S = 10
PULSE_MS = 90

STATUS_TEXT = {
    "pending": "čaká",
    "running": "beží",
    "done": "hotovo",
    "stalled": "bez aktualizácie",
    "stopped": "zastavené",
    "failed": "chyba",
}
STATUS_COLOR = {
    "pending": "pending_ink",
    "running": "accent",
    "done": "done",
    "stalled": "warn",
    "stopped": "pending_ink",
    "failed": "fail",
}
NODE_FILL = {
    "pending": ("pending", "line"),
    "running": ("accent_soft", "accent"),
    "done": ("done_soft", "done"),
    "stalled": ("warn_soft", "warn"),
    "stopped": ("pending", "pending_ink"),
    "failed": ("fail_soft", "fail"),
}

# Checklist of a phase's steps: (mark, color) of the current step by phase status.
CURRENT_STEP = {
    "running": ("▸", "accent"),
    "stalled": ("▸", "warn"),
    "failed": ("✗", "fail"),
    "stopped": ("■", "pending_ink"),
}

# Graph layout in logical units (mirrors the DAG in phases/README.md); x, y = node centre.
# Columns are 180 apart and nodes 150 wide, so edges bend in the 30-unit gaps between them.
VIEW_W, VIEW_H = 1200, 612
NW, NH = 150, 60
CORNER = 6  # edge corners are cut at 45 degrees by this much (logical units)
NODES: dict[str, dict[str, Any]] = {
    "00_ingest": {"x": 100, "y": 175},
    "01_sfm": {"x": 280, "y": 100},
    "04_tagging": {"x": 280, "y": 250},
    "02_reconstruction": {"x": 460, "y": 100},
    "03_refinement": {"x": 640, "y": 100},
    "05_segmentation": {"x": 460, "y": 250},
    "06_mask_embedding": {"x": 640, "y": 250},
    "07_lifting": {"x": 820, "y": 175},
    "08_scene_graph": {"x": 1000, "y": 175},
    "S0": {"x": 1140, "y": 175, "label": "state/v000", "kind": "state", "w": 100},
    "SN": {"x": 70, "y": 477, "label": "stav vN", "kind": "state", "w": 100},
    "R01": {"x": 70, "y": 567, "label": "z 01 SfM", "kind": "ref", "w": 100},
    "09_prompt_parsing": {"x": 250, "y": 412},
    "10_grounding": {"x": 430, "y": 482},
    "11_asset_generation": {"x": 610, "y": 412},
    "12_editing": {"x": 790, "y": 482},
    "S1": {"x": 950, "y": 482, "label": "stav vN+1", "kind": "state", "w": 100},
    "13_rendering": {"x": 1110, "y": 482},
}
# Edges = the files a phase reads (phases 00-08: `inputs()` in the code, 09-13: their
# specification); same DAG as phases/README.md. Routes are orthogonal (source, target,
# waypoints, dashed); corners are drawn cut at 45 degrees (CORNER). They run in the 30-unit
# gaps between columns, in the channel between the build rows (y 130-220) and in buses above
# and below the rows; edges from one source share the bus. "R01" stands for the cameras and
# undistorted images of 01 read by the edit loop.
EDGES: list[tuple[str, str, list[tuple[float, float]], bool]] = [
    ("00_ingest", "01_sfm", [(175, 160), (190, 160), (190, 100), (205, 100)], False),
    ("00_ingest", "04_tagging", [(175, 190), (190, 190), (190, 250), (205, 250)], False),
    ("01_sfm", "02_reconstruction", [(355, 100), (385, 100)], False),
    ("01_sfm", "03_refinement", [(300, 70), (300, 52), (620, 52), (620, 70)], False),
    ("01_sfm", "07_lifting", [(300, 70), (300, 52), (800, 52), (800, 145)], False),
    ("01_sfm", "08_scene_graph", [(300, 70), (300, 52), (980, 52), (980, 145)], False),
    ("01_sfm", "05_segmentation", [(300, 130), (300, 150), (440, 150), (440, 220)], False),
    ("01_sfm", "06_mask_embedding", [(300, 130), (300, 150), (620, 150), (620, 220)], False),
    ("02_reconstruction", "03_refinement", [(535, 100), (565, 100)], False),
    ("03_refinement", "07_lifting", [(715, 100), (730, 100), (730, 160), (745, 160)], False),
    ("03_refinement", "08_scene_graph", [(680, 70), (680, 36), (1020, 36), (1020, 145)], False),
    ("04_tagging", "05_segmentation", [(355, 250), (385, 250)], False),
    ("04_tagging", "06_mask_embedding", [(260, 280), (260, 314), (600, 314), (600, 280)], True),
    ("04_tagging", "08_scene_graph", [(260, 280), (260, 314), (1020, 314), (1020, 205)], False),
    ("05_segmentation", "06_mask_embedding", [(535, 250), (565, 250)], False),
    ("05_segmentation", "07_lifting", [(460, 280), (460, 298), (840, 298), (840, 205)], False),
    ("05_segmentation", "08_scene_graph", [(460, 280), (460, 298), (1000, 298), (1000, 205)], False),
    ("06_mask_embedding", "07_lifting", [(715, 250), (730, 250), (730, 190), (745, 190)], False),
    ("07_lifting", "08_scene_graph", [(895, 175), (925, 175)], False),
    ("08_scene_graph", "S0", [(1075, 175), (1090, 175)], False),
    ("SN", "09_prompt_parsing", [(120, 462), (145, 462), (145, 412), (175, 412)], False),
    ("SN", "10_grounding", [(120, 487), (355, 487)], False),
    ("SN", "12_editing", [(120, 500), (140, 500), (140, 527), (770, 527), (770, 512)], False),
    ("R01", "10_grounding", [(120, 567), (450, 567), (450, 512)], False),
    ("R01", "12_editing", [(120, 567), (810, 567), (810, 512)], False),
    ("R01", "13_rendering", [(120, 567), (1110, 567), (1110, 512)], False),
    ("09_prompt_parsing", "10_grounding", [(325, 412), (340, 412), (340, 470), (355, 470)], False),
    ("10_grounding", "11_asset_generation",
     [(505, 470), (520, 470), (520, 412), (535, 412)], False),
    ("10_grounding", "12_editing", [(505, 494), (715, 494)], False),
    ("11_asset_generation", "12_editing", [(685, 412), (700, 412), (700, 470), (715, 470)], False),
    ("12_editing", "S1", [(865, 482), (900, 482)], False),
    ("S1", "13_rendering", [(1000, 482), (1035, 482)], False),
]  # fmt: skip
# Short names that fit a graph node; the side panel shows the full name.
SHORT = {
    "02_reconstruction": "Rekonštrukcia",
    "03_refinement": "Refinement",
    "08_scene_graph": "Scene graph",
    "06_mask_embedding": "Embeddingy",
    "09_prompt_parsing": "Parsovanie",
    "11_asset_generation": "Nový objekt",
}


def fmt_dur(s: float | None) -> str:
    if s is None:
        return "–"
    s = max(0, round(s))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {sec:02d} s"
    return f"{sec} s"


def chamfer(pts: list[tuple[float, float]], r: float) -> list[tuple[float, float]]:
    """Orthogonal polyline with every corner cut at 45 degrees (equal cut on both legs)."""
    out = [pts[0]]
    for (x0, y0), (x1, y1), (x2, y2) in zip(pts, pts[1:], pts[2:], strict=False):
        d_in, d_out = abs(x1 - x0) + abs(y1 - y0), abs(x2 - x1) + abs(y2 - y1)
        c = min(r, d_in / 2, d_out / 2)
        if c <= 0:
            out.append((x1, y1))
            continue
        out.append((x1 - (x1 - x0) / d_in * c, y1 - (y1 - y0) / d_in * c))
        out.append((x1 + (x2 - x1) / d_out * c, y1 + (y2 - y1) / d_out * c))
    out.append(pts[-1])
    return out


def fit(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


class PhaseRow(ctk.CTkFrame):
    """One phase in the side panel; widgets are created once and updated in place."""

    def __init__(self, master, phase_id: str, on_click: Callable[[str], None]):
        super().__init__(
            master, fg_color=C["panel"], border_width=1, border_color=C["line"], corner_radius=9
        )
        self.phase_id = phase_id
        self.expanded = False  # show the step checklist whatever the status (row selected)
        self.columnconfigure(0, weight=1)
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 6))
        head.columnconfigure(3, weight=1)
        bold = ctk.CTkFont(size=14, weight="bold")
        self.num = ctk.CTkLabel(head, text=phase_id[:2], font=bold, text_color=C["muted"])
        self.num.grid(row=0, column=0, padx=(0, 7))
        self.name = ctk.CTkLabel(head, text="", font=bold, text_color=C["ink"])
        self.name.grid(row=0, column=1)
        self.impl = ctk.CTkLabel(head, text="", font=ctk.CTkFont(size=12), text_color=C["muted"])
        self.impl.grid(row=0, column=2, padx=(7, 0))
        self.badge = ctk.CTkLabel(head, text="", font=ctk.CTkFont(size=12, weight="bold"))
        self.badge.grid(row=0, column=3, sticky="e")
        self.chevron = ctk.CTkLabel(
            head, text="▸", width=14, font=ctk.CTkFont(size=15), text_color=C["muted"]
        )
        self.chevron.grid(row=0, column=4, padx=(8, 0))
        for w in (self, head, self.num, self.name, self.impl, self.badge, self.chevron):
            w.bind("<Button-1>", lambda _e: on_click(phase_id))
            w.configure(cursor="hand2")
        self.bar = ctk.CTkProgressBar(self, height=6, corner_radius=3, fg_color=C["pending"])
        self.bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.steps_box = ctk.CTkFrame(self, fg_color="transparent")
        self.steps_box.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))
        self.step_labels: list[ctk.CTkLabel] = []
        self.step_names: list[str] = []
        self.step_fonts = (ctk.CTkFont(size=12), ctk.CTkFont(size=12, weight="bold"))
        detail = ctk.CTkFrame(self, fg_color="transparent")
        detail.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 6))
        detail.columnconfigure(0, weight=1)
        small = ctk.CTkFont(size=12)
        self.stage = ctk.CTkLabel(
            detail, text="", font=small, text_color=C["ink"], justify="left", anchor="w",
            wraplength=250,
        )  # fmt: skip
        self.stage.grid(row=0, column=0, sticky="w")
        self.times = ctk.CTkLabel(detail, text="", font=small, text_color=C["muted"])
        self.times.grid(row=0, column=1, sticky="ne", padx=(8, 0))
        self.error = ctk.CTkLabel(
            self, text="", font=small, text_color=C["fail"], justify="left", anchor="w",
            wraplength=340,
        )  # fmt: skip
        self.error.grid(row=4, column=0, sticky="w", padx=12, pady=(0, 6))
        self.note = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=11), text_color=C["muted"], anchor="w",
            justify="left", wraplength=340,
        )  # fmt: skip
        self.note.grid(row=5, column=0, sticky="w", padx=12, pady=(0, 6))

    def update_from(self, p: dict[str, Any], planned: tuple[list[str], str] | None = None) -> None:
        """`planned`: steps and their source for a selected row without steps of its own."""
        status = p["status"]
        active = status in ("running", "stalled")
        self.name.configure(text=p["name"])
        self.impl.configure(text=p["impl"] or "")
        if status == "done" and p["elapsed_s"] is not None:
            badge = fmt_dur(p["elapsed_s"])  # how long it took, instead of "hotovo"
        else:
            badge = STATUS_TEXT[status] + (f" · {round(p['fraction'] * 100)} %" if active else "")
        self.badge.configure(text=badge, text_color=C[STATUS_COLOR[status]])
        border = {"running": "accent", "stalled": "warn", "failed": "fail"}.get(status, "line")
        if self.expanded and not active:
            border = "state"
        self.configure(border_color=C[border], border_width=2 if active or self.expanded else 1)
        self.chevron.configure(text="▾" if self.expanded else "▸")

        if status == "pending":
            self.bar.grid_remove()
        else:
            bar_color = {"done": "done", "stalled": "warn", "failed": "fail",
                         "stopped": "pending_ink"}.get(status, "accent")  # fmt: skip
            self.bar.configure(progress_color=C[bar_color])
            self.bar.set(p["fraction"])
            self.bar.grid()

        steps, note = p["steps"], ""
        if self.expanded and not steps and planned is not None:
            steps, note = planned
            if not steps:
                note = note or "kroky nie sú známe"
        live = status in CURRENT_STEP and bool(p["steps"])  # always shown while it matters
        show_steps = bool(steps) and (live or self.expanded)
        self._update_steps(steps if show_steps else None, p)
        self.note.configure(text=note)
        self.note.grid() if note and self.expanded else self.note.grid_remove()
        if status in ("pending", "done"):
            stage = ""
        elif show_steps and p["step"] is not None:
            stage = ""  # the step and its message are in the checklist
        else:
            stage = " — ".join(x for x in (p["stage"], p["message"]) if x)
        if status == "running":
            eta = f"\nzostáva ~{fmt_dur(p['eta_s'])}" if p["eta_s"] is not None else ""
            times = fmt_dur(p["elapsed_s"]) + eta
        elif status == "stalled":
            times = f"bez zmeny {fmt_dur(p['idle_s'])}"
        elif status in ("failed", "stopped") and p["elapsed_s"] is not None:
            times = fmt_dur(p["elapsed_s"])
        else:
            times = ""  # done: the duration is in the badge
        self.stage.configure(text=stage)
        self.times.configure(text=times)
        self.stage.master.grid() if (stage or times) else self.stage.master.grid_remove()
        self.error.configure(
            text=p["error"] or "", text_color=C["fail" if status == "failed" else "muted"]
        )
        self.error.grid() if p["error"] else self.error.grid_remove()

    def _update_steps(self, steps: list[str] | None, p: dict[str, Any]) -> None:
        """Checklist of the phase's steps: done ✓, current (by status), pending ○."""
        if steps is None:
            self.steps_box.grid_remove()
            return
        if steps != self.step_names:
            for label in self.step_labels:
                label.destroy()
            self.step_names = list(steps)
            self.step_labels = [
                ctk.CTkLabel(self.steps_box, text="", font=self.step_fonts[0], anchor="w",
                             justify="left", height=18, wraplength=340)
                for _ in self.step_names
            ]  # fmt: skip
            for i, label in enumerate(self.step_labels):
                label.grid(row=i, column=0, sticky="w")
        status = p["status"]
        current = p["step"] if status in CURRENT_STEP else None
        message = p["message"]
        for i, (name, label) in enumerate(zip(self.step_names, self.step_labels, strict=True)):
            if status == "done" or (current is not None and i < current):
                mark, color, bold = "✓", "done", False
            elif i == current:
                (mark, color), bold = CURRENT_STEP[status], True
            else:
                mark, color, bold = "○", "pending_ink", False
            text = f"{mark}  {name}" + (f"  ·  {message}" if i == current and message else "")
            label.configure(text=text, text_color=C[color], font=self.step_fonts[bold])
        self.steps_box.grid()


class MonitorApp(ctk.CTk):
    def __init__(self, workspace: Path, scene: str | None):
        super().__init__(fg_color=C["bg"])
        self.workspace = workspace
        self.title("Scenepipe monitor")
        # fit the screen (minus taskbar), then maximize where the platform supports it
        w = min(1680, self.winfo_screenwidth() - 60)
        h = min(940, self.winfo_screenheight() - 120)
        self.geometry(f"{w}x{h}+20+20")
        self.minsize(1100, 640)
        if sys.platform == "win32":
            self.after(0, lambda: self.state("zoomed"))
        self.scenes: list[str] = []
        self.scene = scene
        self.data: dict[str, Any] | None = None
        self.last_ok = 0.0
        self.last_scenes = 0.0
        self.pulse = 0
        self.halos: dict[str, int] = {}
        self.selected: str | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_header()
        self._build_body()
        self._refresh_scenes(force=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll)
        self.after(PULSE_MS, self._animate)

    # --- layout ---------------------------------------------------------------------------

    def _build_header(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=56)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(4, weight=1)
        ctk.CTkLabel(
            bar, text="Scenepipe monitor", font=ctk.CTkFont(size=17, weight="bold"),
            text_color=C["ink"],
        ).grid(row=0, column=0, padx=(22, 18), pady=12)  # fmt: skip
        self.overall = ctk.CTkLabel(bar, text="", text_color=C["muted"])
        self.overall.grid(row=0, column=3, sticky="w")
        self.live = ctk.CTkLabel(bar, text="načítavam…", text_color=C["muted"])
        self.live.grid(row=0, column=5, padx=22)

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=18, pady=18)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.run_panel = RunPanel(body, self.workspace, self._select_scene, self._make_run)
        self.run_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        left = ctk.CTkFrame(body, fg_color=C["panel"], border_width=1, border_color=C["line"])
        left.grid(row=0, column=1, sticky="nsew", padx=(0, 16))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        self._section_title(left, "GRAF FÁZ").grid(row=0, column=0, sticky="w", padx=16, pady=10)
        self.canvas = tk.Canvas(left, highlightthickness=0, bd=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=10)
        self.canvas.bind("<Configure>", lambda _e: self._draw_graph())
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        legend = ctk.CTkFrame(left, fg_color="transparent")
        legend.grid(row=2, column=0, sticky="w", padx=16, pady=(4, 12))
        for i, (status, text) in enumerate(STATUS_TEXT.items()):
            color = C[NODE_FILL[status][1]] if status != "pending" else C["pending_ink"]
            ctk.CTkLabel(legend, text="■", text_color=color).grid(row=0, column=2 * i)
            ctk.CTkLabel(legend, text=text, text_color=C["muted"]).grid(
                row=0, column=2 * i + 1, padx=(4, 16)
            )

        right = ctk.CTkFrame(body, fg_color=C["panel"], border_width=1, border_color=C["line"])
        right.grid(row=0, column=2, sticky="nsew")
        right.rowconfigure(1, weight=1)
        self._section_title(right, "PRIEBEH FÁZ").grid(
            row=0, column=0, sticky="w", padx=16, pady=10
        )
        self.list = ctk.CTkScrollableFrame(right, width=390, fg_color="transparent")
        self.list.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 10))
        self.list.columnconfigure(0, weight=1)
        self.rows: dict[str, PhaseRow] = {}
        r = 0
        self._section_title(self.list, "BUILD").grid(row=r, column=0, sticky="w", pady=(0, 4))
        for phase in BUILD_PHASES:
            r += 1
            self.rows[phase] = PhaseRow(self.list, phase, self._focus)
            self.rows[phase].grid(row=r, column=0, sticky="ew", pady=4, padx=(0, 6))
        r += 1
        self.edit_title = self._section_title(self.list, "EDIT LOOP")
        self.edit_title.grid(row=r, column=0, sticky="w", pady=(12, 4))
        for phase in (p for p in PHASES if p not in BUILD_PHASES):
            r += 1
            self.rows[phase] = PhaseRow(self.list, phase, self._focus)
            self.rows[phase].grid(row=r, column=0, sticky="ew", pady=4, padx=(0, 6))

    def _section_title(self, master, text: str) -> ctk.CTkLabel:
        font = ctk.CTkFont(size=12, weight="bold")
        return ctk.CTkLabel(master, text=text, font=font, text_color=C["muted"])

    # --- data -----------------------------------------------------------------------------

    def _refresh_scenes(self, force: bool = False) -> None:
        if not force and time.monotonic() - self.last_scenes < SCENES_EVERY_S:
            return
        self.last_scenes = time.monotonic()
        self.scenes = list_scenes(self.workspace)
        if self.scene not in self.scenes:
            self.scene = self.scenes[0] if self.scenes else None
        builds = {}
        for s in self.scenes:
            phases = scene_status(self.workspace, s)["phases"]
            done = sum(p["status"] == "done" for p in phases if p["id"] in BUILD_PHASES)
            builds[s] = f"build {done}/{len(BUILD_PHASES)}" if done else ""
        self.run_panel.set_scenes(self.scenes, self.scene, builds)

    def _select_scene(self, scene: str) -> None:
        self.scene = scene
        self._refresh_scenes(force=True)
        self._poll(reschedule=False)

    def _make_run(self, scene: str, phases: list[str], profile: str) -> PipelineRun:
        self.scene = scene
        return PipelineRun(scene, phases, profile, self.workspace, Runners.load())

    def _on_close(self) -> None:
        if self.run_panel.stop_for_close():
            self.destroy()

    def _poll(self, reschedule: bool = True) -> None:
        try:
            self._refresh_scenes()
            self.run_panel.refresh()
            if self.scene:
                self.data = scene_status(self.workspace, self.scene)
                self.last_ok = time.time()
                self._update_rows()
                self._draw_graph()
        except OSError as e:  # workspace on a network drive / WSL path may hiccup
            self.live.configure(text=f"chyba čítania: {e}", text_color=C["fail"])
        else:
            stamp = time.strftime("%H:%M:%S", time.localtime(self.last_ok))
            self.live.configure(text=f"●  aktualizované {stamp}", text_color=C["done"])
        if reschedule:
            self.after(POLL_MS, self._poll)

    def _update_rows(self) -> None:
        assert self.data is not None
        phases = self.data["phases"]
        profile = self.run_panel.selected_profile()
        for p in phases:
            row = self.rows[p["id"]]
            row.expanded = p["id"] == self.selected
            planned = None
            if row.expanded and not p["steps"]:
                planned = planned_steps(self.workspace, p["id"], profile)
            row.update_from(p, planned)
        edit = self.data["edit_id"]
        self.edit_title.configure(
            text=f"EDIT LOOP · edits/{edit}" if edit else "EDIT LOOP · zatiaľ žiadna úprava"
        )
        build = [p for p in phases if p["id"] in BUILD_PHASES]
        done = sum(p["status"] == "done" for p in build)
        running = next((p for p in phases if p["status"] in ("running", "stalled")), None)
        text = f"build {done}/{len(build)}"
        if running:
            text += f"   ·   beží {running['num']} {running['name']}"
        self.overall.configure(text=text)

    # --- graph ----------------------------------------------------------------------------

    def _col(self, key: str) -> str:
        light, dark = C[key]
        return dark if ctk.get_appearance_mode() == "Dark" else light

    def _transform(self) -> tuple[float, float, float]:
        w, h = max(self.canvas.winfo_width(), 10), max(self.canvas.winfo_height(), 10)
        s = min(w / VIEW_W, h / VIEW_H)
        return s, (w - VIEW_W * s) / 2, (h - VIEW_H * s) / 2

    def _draw_graph(self) -> None:
        cv = self.canvas
        cv.delete("all")
        cv.configure(bg=self._col("panel"))
        s, ox, oy = self._transform()

        def pt(x: float, y: float) -> tuple[float, float]:
            return ox + x * s, oy + y * s

        by_id = {p["id"]: p for p in self.data["phases"]} if self.data else {}
        # bands
        for y0, y1, label in ((4, 328, "BUILD · zo snímok, raz pre scénu"),
                              (342, 608, "EDIT LOOP · pre každý prompt")):  # fmt: skip
            self._round_rect(*pt(4, y0), *pt(1196, y1), 14 * s, fill="", outline=self._col("line"),
                             dash=(4, 5))  # fmt: skip
            cv.create_text(*pt(18, y0 + 22), text=label, anchor="w", fill=self._col("muted"),
                           font=self._font(15, s, bold=True))  # fmt: skip
        # edges
        hot_edges = []
        for a, b, route, dashed in EDGES:
            hot = b in by_id and by_id[b]["status"] == "running"
            item = cv.create_line(
                *(c for x, y in chamfer(route, CORNER) for c in pt(x, y)),
                width=2.4 if hot else 1.5, fill=self._col("accent" if hot else "edge"),
                dash=(5, 4) if dashed else (), arrow=tk.LAST,
                arrowshape=(7 * s + 2, 8.5 * s + 2, 3.5 * s + 1), capstyle=tk.BUTT,
                joinstyle=tk.MITER,
            )  # fmt: skip
            if hot:
                hot_edges.append(item)
        for item in hot_edges:  # running inputs on top of crossing edges
            cv.tag_raise(item)
        # nodes
        self.halos = {}
        for nid, n in NODES.items():
            w = n.get("w", NW)
            x0, y0 = pt(n["x"] - w / 2, n["y"] - NH / 2)
            x1, y1 = pt(n["x"] + w / 2, n["y"] + NH / 2)
            r = 10 * s
            if "kind" in n:
                state = n["kind"] == "state"
                self._round_rect(
                    x0, y0, x1, y1, r,
                    fill=self._col("state_soft" if state else "panel"),
                    outline=self._col("state" if state else "edge"),
                    dash=() if state else (3, 3), width=1.5,
                )  # fmt: skip
                cv.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=n["label"],
                               fill=self._col("state" if state else "muted"),
                               font=self._font(16, s))  # fmt: skip
                continue
            p = by_id.get(nid, {"status": "pending", "name": nid, "fraction": 0, "stage": ""})
            status = p["status"]
            fill, outline = NODE_FILL[status]
            if status in ("running", "stalled"):
                self.halos[nid] = self._round_rect(
                    x0, y0, x1, y1, r, fill="", outline=self._col(outline), width=2
                )
            width = 3.5 if nid == self.selected else 2.5 if status != "done" else 1.5
            if status == "pending" and nid != self.selected:
                width = 1.2
            self._round_rect(x0, y0, x1, y1, r, fill=self._col(fill), outline=self._col(outline),
                             width=width, tags=("node", f"node:{nid}"))  # fmt: skip
            if status in ("running", "stalled"):
                sub = f"{round(p['fraction'] * 100)} % · {fit(p['stage'], 12)}"
            elif status == "done" and p.get("elapsed_s") is not None:
                sub = fmt_dur(p["elapsed_s"])
            else:
                sub = STATUS_TEXT[status]
            sub_col = self._col("pending_ink" if status == "pending" else STATUS_COLOR[status])
            cv.create_text(x0 + 12 * s, y0 + 17 * s, anchor="w", fill=sub_col,
                           text=f"{nid[:2]}  ·  {sub}", font=self._font(14, s),
                           tags=(f"node:{nid}",))  # fmt: skip
            ink = self._col("pending_ink" if status == "pending" else "ink")
            cv.create_text(x0 + 12 * s, y0 + 38 * s, anchor="w", fill=ink,
                           text=fit(SHORT.get(nid, p["name"]), 14),
                           font=self._font(18, s, bold=True), tags=(f"node:{nid}",))  # fmt: skip
            if status in ("running", "stalled"):
                bx0, bx1, by = x0 + 12 * s, x1 - 12 * s, y1 - 7 * s
                cv.create_line(bx0, by, bx1, by, fill=self._col("line"), width=3 * s)
                cv.create_line(bx0, by, bx0 + (bx1 - bx0) * p["fraction"], by,
                               fill=self._col(outline), width=3 * s)  # fmt: skip

    @staticmethod
    def _font(size: float, scale: float, bold: bool = False) -> tuple:
        """Font with a pixel size (negative in Tk), so it scales with the graph, not the DPI."""
        px = -max(8, round(size * scale))
        return ("Segoe UI", px, "bold") if bold else ("Segoe UI", px)

    def _round_rect(self, x0, y0, x1, y1, r, **kw) -> int:
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
        pts = [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
            x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
        ]  # fmt: skip
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    def _animate(self) -> None:
        """Pulsing halo around running phases."""
        self.pulse = (self.pulse + 1) % 20
        s, _, _ = self._transform()
        grow = self.pulse / 20
        for nid, item in self.halos.items():
            n = NODES[nid]
            w, pad = n.get("w", NW), 2 + 10 * grow
            _, ox, oy = self._transform()
            x0, y0 = ox + (n["x"] - w / 2 - pad) * s, oy + (n["y"] - NH / 2 - pad) * s
            x1, y1 = ox + (n["x"] + w / 2 + pad) * s, oy + (n["y"] + NH / 2 + pad) * s
            r = min((10 + pad) * s, (x1 - x0) / 2, (y1 - y0) / 2)
            pts = [
                x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
                x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
            ]  # fmt: skip
            self.canvas.coords(item, *pts)
            status = self.data and next(
                (p["status"] for p in self.data["phases"] if p["id"] == nid), "running"
            )
            base = "warn" if status == "stalled" else "accent"
            self.canvas.itemconfigure(item, outline=self._blend(base, "panel", grow))
        self.after(PULSE_MS, self._animate)

    def _blend(self, a: str, b: str, t: float) -> str:
        ca, cb = self._col(a), self._col(b)
        rgb = [
            round(int(ca[i : i + 2], 16) * (1 - t) + int(cb[i : i + 2], 16) * t) for i in (1, 3, 5)
        ]
        return "#" + "".join(f"{v:02x}" for v in rgb)

    def _on_canvas_click(self, event) -> None:
        for item in self.canvas.find_overlapping(event.x, event.y, event.x, event.y):
            for tag in self.canvas.gettags(item):
                if tag.startswith("node:"):
                    self._focus(tag[5:])
                    return

    def _focus(self, nid: str) -> None:
        """Select a phase (graph node or side-panel row): expands its steps; again = collapse."""
        if nid not in self.rows:  # state nodes
            return
        self.selected = None if self.selected == nid else nid
        if self.data:
            self._update_rows()
        self._draw_graph()
        if self.selected:
            row = self.rows[nid]
            self.update_idletasks()
            total = max(row.master.winfo_height(), 1)
            self.list._parent_canvas.yview_moveto(max(0.0, (row.winfo_y() - 40) / total))


def run(workspace: Path, scene: str | None) -> None:
    ctk.set_appearance_mode("system")
    app = MonitorApp(workspace, scene)
    app.mainloop()
