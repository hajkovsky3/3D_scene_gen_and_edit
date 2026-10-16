"""Left panel of the monitor window: choose an input (scene), add a new one, start/stop a run."""

from __future__ import annotations

import os
import re
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from scenepipe.cli import is_implemented, profiles
from scenepipe.core.workspace import BUILD_PHASES
from scenepipe.monitor.runner import PipelineRun
from scenepipe.monitor.status import NAMES, PHOTO_EXTS, VIDEO_EXTS, input_summary
from scenepipe.monitor.theme import C

SCENE_ID = re.compile(r"^[a-z0-9_-]+$")
PROFILE_LABELS = {"default": "plný", "fast": "rýchly"}
RUN_TEXT = {"running": "beží", "done": "hotovo", "failed": "chyba", "stopped": "zastavené"}
RUN_COLOR = {"running": "accent", "done": "done", "failed": "fail", "stopped": "warn"}


def plural(n: int, one: str, few: str, many: str) -> str:
    return one if n == 1 else few if 2 <= n <= 4 else many


def describe_input(summary: dict[str, int]) -> str:
    parts = []
    if v := summary["videos"]:
        parts.append(f"{v} {plural(v, 'video', 'videá', 'videí')}")
    if p := summary["photos"]:
        parts.append(f"{p} {plural(p, 'fotka', 'fotky', 'fotiek')}")
    if not parts:
        return "bez vstupu"
    return " · ".join(parts) + f" · {summary['bytes'] / 2**20:.0f} MB"


def suggest_scene_id(path: str) -> str:
    stem = re.sub(r"[^a-z0-9_-]+", "_", Path(path).stem.lower()).strip("_")
    return stem or "scena"


class SceneItem(ctk.CTkFrame):
    """Clickable scene entry: name (bold) and a one-line description, left aligned."""

    def __init__(self, master, on_click: Callable[[], None]):
        super().__init__(master, corner_radius=8, border_width=1, fg_color="transparent")
        self.selected = False
        self.columnconfigure(0, weight=1)
        self.name = ctk.CTkLabel(
            self, text="", anchor="w", font=ctk.CTkFont(size=14, weight="bold"),
            text_color=C["ink"], height=20,
        )  # fmt: skip
        self.name.grid(row=0, column=0, sticky="ew", padx=12, pady=(7, 0))
        self.detail = ctk.CTkLabel(
            self, text="", anchor="w", font=ctk.CTkFont(size=12), text_color=C["muted"],
            height=18,
        )  # fmt: skip
        self.detail.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 7))
        for w in (self, self.name, self.detail):
            w.bind("<Button-1>", lambda _e: on_click())
            w.bind("<Enter>", lambda _e: self._hover(True))
            w.bind("<Leave>", lambda _e: self._hover(False))
            w.configure(cursor="hand2")

    def show(self, name: str, detail: str, selected: bool) -> None:
        self.selected = selected
        self.name.configure(text=name)
        self.detail.configure(text=detail)
        self._hover(False)

    def _hover(self, on: bool) -> None:
        if self.selected:
            self.configure(fg_color=C["accent_soft"], border_color=C["accent"])
        else:
            self.configure(fg_color=C["pending"] if on else "transparent", border_color=C["line"])


class RunPanel(ctk.CTkFrame):
    """Owns the PipelineRun; the window calls refresh() on every poll."""

    def __init__(
        self,
        master,
        workspace: Path,
        on_select: Callable[[str], None],
        make_run: Callable[[str, list[str], str], PipelineRun],
    ):
        super().__init__(master, fg_color=C["panel"], border_width=1, border_color=C["line"])
        self.workspace = workspace
        self.on_select = on_select
        self.make_run = make_run
        self.run: PipelineRun | None = None
        self.scene: str | None = None
        self.buttons: dict[str, SceneItem] = {}
        self.summaries: dict[str, dict[str, int]] = {}
        self.busy = ""  # e.g. "kopírujem…" while a new input is copied
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        bold12 = ctk.CTkFont(size=12, weight="bold")
        ctk.CTkLabel(self, text="VSTUPY", font=bold12, text_color=C["muted"]).grid(
            row=0, column=0, sticky="w", padx=16, pady=(10, 4)
        )
        self.scene_list = ctk.CTkScrollableFrame(self, fg_color="transparent", width=230)
        self.scene_list.grid(row=1, column=0, sticky="nsew", padx=(8, 4))
        self.scene_list.columnconfigure(0, weight=1)
        ctk.CTkButton(
            self, text="+  Nový vstup…", command=self._add_input, height=34,
            fg_color="transparent", border_width=1, border_color=C["line"],
            text_color=C["ink"], hover_color=C["pending"],
        ).grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 14))  # fmt: skip

        ctk.CTkFrame(self, height=1, fg_color=C["line"]).grid(row=3, column=0, sticky="ew")
        ctk.CTkLabel(self, text="SPUSTENIE", font=bold12, text_color=C["muted"]).grid(
            row=4, column=0, sticky="w", padx=16, pady=(12, 4)
        )
        form = ctk.CTkFrame(self, fg_color="transparent")
        form.grid(row=5, column=0, sticky="ew", padx=14)
        form.columnconfigure(1, weight=1)
        self.profile_names = profiles()
        labels = [PROFILE_LABELS.get(p, p) for p in self.profile_names]
        ctk.CTkLabel(form, text="profil", text_color=C["muted"]).grid(row=0, column=0, sticky="w")
        self.profile = ctk.CTkSegmentedButton(form, values=labels, command=self._on_profile)
        self.profile.set(PROFILE_LABELS.get("fast", labels[0]) if "fast" in self.profile_names
                         else labels[0])  # fmt: skip
        self.profile.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=4)
        ctk.CTkLabel(form, text="od fázy", text_color=C["muted"]).grid(row=1, column=0, sticky="w")
        self.start_phase = ctk.CTkOptionMenu(form, values=["–"], dynamic_resizing=False)
        self.start_phase.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=4)
        self._on_profile(self.profile.get())

        self.run_btn = ctk.CTkButton(
            self, text="▶   Spustiť", command=self._start, height=40,
            font=ctk.CTkFont(size=14, weight="bold"), fg_color=C["accent"],
        )  # fmt: skip
        self.run_btn.grid(row=6, column=0, sticky="ew", padx=14, pady=(12, 6))
        self.stop_btn = ctk.CTkButton(
            self, text="■   Zastaviť", command=self._stop, height=34, state="disabled",
            fg_color="transparent", border_width=1, border_color=C["line"], text_color=C["fail"],
            hover_color=C["fail_soft"],
        )  # fmt: skip
        self.stop_btn.grid(row=7, column=0, sticky="ew", padx=14)
        self.status = ctk.CTkLabel(
            self, text="", justify="left", anchor="w", wraplength=230, text_color=C["muted"]
        )
        self.status.grid(row=8, column=0, sticky="ew", padx=16, pady=(10, 2))
        self.log_btn = ctk.CTkButton(
            self, text="Otvoriť log", command=self._open_log, height=28, state="disabled",
            fg_color="transparent", text_color=C["accent"], hover_color=C["pending"], anchor="w",
        )  # fmt: skip
        self.log_btn.grid(row=9, column=0, sticky="w", padx=10, pady=(0, 12))

    # --- scenes ---------------------------------------------------------------------------

    def set_scenes(self, scenes: list[str], selected: str | None, builds: dict[str, str]) -> None:
        """Rebuild the list when scenes change; refresh texts and the highlight otherwise."""
        self.summaries = {s: input_summary(self.workspace / s) for s in scenes}
        if list(self.buttons) != scenes:
            for b in self.buttons.values():
                b.destroy()
            self.buttons = {}
            for i, s in enumerate(scenes):
                item = SceneItem(self.scene_list, lambda s=s: self.on_select(s))
                item.grid(row=i, column=0, sticky="ew", pady=3, padx=(0, 4))
                self.buttons[s] = item
        for s, item in self.buttons.items():
            detail = describe_input(self.summaries[s])
            if builds.get(s):
                detail += f" · {builds[s]}"
            active = self.run is not None and self.run.state == "running" and self.run.scene == s
            item.show(("▶  " if active else "") + s, detail, s == selected)
        self.scene = selected

    # --- run ------------------------------------------------------------------------------

    def _profile_name(self) -> str:
        label = self.profile.get()
        return next((p for p in self.profile_names if PROFILE_LABELS.get(p, p) == label), label)

    def _runnable(self, profile: str) -> list[str]:
        """Implemented build phases from 00 on (the chain stops at the first missing one)."""
        out = []
        for phase in BUILD_PHASES:
            if not is_implemented(phase, profile):
                break
            out.append(phase)
        return out

    def _on_profile(self, _label: str) -> None:
        phases = self._runnable(self._profile_name())
        self.phase_labels = {f"{p[:2]}  {NAMES[p]}": p for p in phases}
        values = list(self.phase_labels) or ["–"]
        self.start_phase.configure(values=values)
        if self.start_phase.get() not in values:
            self.start_phase.set(values[0])

    def _start(self) -> None:
        scene = self.scene
        if not scene:
            return
        start = self.phase_labels.get(self.start_phase.get())
        if start is None:
            messagebox.showwarning("Spustenie", "Žiadna fáza nie je implementovaná.")
            return
        phases = list(self.phase_labels.values())
        phases = phases[phases.index(start) :]
        if start == "00_ingest" and not any(
            self.summaries.get(scene, {}).get(k) for k in ("videos", "photos")
        ):
            messagebox.showwarning(
                "Spustenie", f"Scéna {scene} nemá vstupné súbory (workspace/{scene}/input)."
            )
            return
        done = [p for p in phases if (self.workspace / scene / p / "manifest.json").exists()]
        if done and not messagebox.askyesno(
            "Prepísať výsledky?",
            f"Scéna {scene} už má výsledky fáz {', '.join(p[:2] for p in done)}.\n"
            "Spustenie ich prepíše. Pokračovať?",
        ):
            return
        self.run = self.make_run(scene, phases, self._profile_name())
        self.run.start()
        self.refresh()

    def _stop(self) -> None:
        run = self.run
        if run is None or run.state != "running":
            return
        if messagebox.askyesno("Zastaviť beh?", f"Zastaviť beh scény {run.scene}?"):
            self.stop_btn.configure(state="disabled")
            threading.Thread(target=run.stop, daemon=True).start()

    def stop_for_close(self) -> bool:
        """Called when the window closes; returns False to keep the window open."""
        run = self.run
        if run is None or run.state != "running":
            return True
        if not messagebox.askyesno(
            "Beh ešte prebieha", f"Beh scény {run.scene} sa zastaví. Zavrieť okno?"
        ):
            return False
        run.stop()
        return True

    def _open_log(self) -> None:
        if self.run and self.run.log_path and self.run.log_path.exists():
            os.startfile(self.run.log_path)

    def refresh(self) -> None:
        run = self.run
        running = run is not None and run.state == "running"
        self.run_btn.configure(state="disabled" if running or self.busy else "normal")
        if not running:
            self.stop_btn.configure(state="disabled")
        elif self.stop_btn.cget("state") == "disabled" and not run._stop_requested.is_set():
            self.stop_btn.configure(state="normal")
        if self.busy:
            self.status.configure(text=self.busy, text_color=C["muted"])
        elif run is None:
            self.status.configure(text="Vyber vstup a spusti build.", text_color=C["muted"])
        else:
            phase = f"\n{run.current[:2]} {NAMES[run.current]} ({run.message})" if (
                running and run.current
            ) else f"\n{run.message}"  # fmt: skip
            self.status.configure(
                text=f"{run.scene}: {RUN_TEXT[run.state]}{phase}",
                text_color=C[RUN_COLOR[run.state]],
            )
        has_log = run is not None and run.log_path is not None
        self.log_btn.configure(state="normal" if has_log else "disabled")

    # --- new input ------------------------------------------------------------------------

    def _add_input(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS | PHOTO_EXTS))
        files = filedialog.askopenfilenames(
            title="Vyber video alebo fotky scény",
            filetypes=[("Video a fotky", exts), ("Všetky súbory", "*.*")],
        )
        if not files:
            return
        default = suggest_scene_id(files[0])
        dialog = ctk.CTkInputDialog(
            title="Nový vstup",
            text=f"Názov scény (malé písmená, číslice, _ a -).\nPrázdne = {default}",
        )
        name = (dialog.get_input() or "").strip() or default
        if not SCENE_ID.match(name):
            messagebox.showerror("Nový vstup", f"Neplatný názov scény: {name!r}")
            return
        target = self.workspace / name / "input"
        if target.exists() and any(target.iterdir()):
            messagebox.showerror("Nový vstup", f"Scéna {name} už má vstupné súbory.")
            return
        self.busy = f"kopírujem {len(files)} {plural(len(files), 'súbor', 'súbory', 'súborov')}…"
        self.refresh()

        def copy() -> None:
            try:
                target.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy2(f, target / Path(f).name)
                error = None
            except OSError as e:
                error = str(e)
            self.after(0, lambda: self._copied(name, error))

        threading.Thread(target=copy, daemon=True).start()

    def _copied(self, name: str, error: str | None) -> None:
        self.busy = ""
        if error:
            messagebox.showerror("Nový vstup", f"Kopírovanie zlyhalo: {error}")
        self.on_select(name)
        self.refresh()
