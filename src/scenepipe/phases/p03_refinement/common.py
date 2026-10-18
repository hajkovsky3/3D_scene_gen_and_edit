"""Phase 03_refinement: shared mechanics of all implementations (inputs, before/after
evaluation, report, validation). Spec: phases/03_refinement.md

An implementation only provides `refine()`. Pruning runs on the CPU (NumPy); torch and gsplat
are imported only for the evaluation (`eval: true`), so `eval: false` runs without a GPU.
"""

from __future__ import annotations

import json
import logging
import shutil
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from scenepipe.core.cameras import Camera, load_cameras, scene_scale
from scenepipe.core.gaussians import Gaussians, read_ply, write_ply
from scenepipe.core.images import load_image
from scenepipe.core.phase import ContractError, Phase

log = logging.getLogger(__name__)

CRITERIA = ("crop", "opacity", "views", "scale", "aniso")  # order in which removals are counted
JPEG_QUALITY = 95
PSNR_CONSISTENCY_DB = 0.05
PSNR_DROP_WARN_DB = 1.0
REFINE_END = 0.5  # share of the phase progress taken by refine() when evaluating


class RefinementBase(Phase):
    phase = "03_refinement"
    contract_version = 1
    defaults: ClassVar[dict[str, Any]] = {
        "crop": False,
        "crop_quantile": 0.01,
        "crop_margin": 0.1,
        "min_opacity": 0.02,
        "min_views": 3,
        "max_scale_frac": 0,
        "max_aniso": 0,
        "aniso_action": "clamp",
        "eval": True,
        "lpips": True,
        "num_test_renders": 8,
    }

    def inputs(self) -> dict[str, Path]:
        return {
            "gaussians": self.ws.abs("02_reconstruction/gaussians.ply"),
            "reconstruction_report": self.ws.abs("02_reconstruction/report.json"),
            "reconstruction_manifest": self.ws.abs("02_reconstruction/manifest.json"),
            "cameras": self.ws.abs("01_sfm/cameras.json"),
            "points": self.ws.abs("01_sfm/points.npz"),
            "images": self.ws.abs("01_sfm/images_undistorted"),
        }

    @abstractmethod
    def refine(
        self, g: Gaussians, train: list[Camera], radius: float
    ) -> tuple[Gaussians | None, dict[str, int], int, dict[str, Any]]:
        """Return (refined gaussians or None if unchanged, removed per criterion, clamped,
        extra report fields)."""

    # --- run -------------------------------------------------------------------------------

    def run(self) -> dict[str, Path]:
        cfg = self.config
        src = self.ws.abs("02_reconstruction/gaussians.ply")
        rec = json.loads(self.ws.abs("02_reconstruction/report.json").read_text("utf-8"))
        rec_cfg = json.loads(self.ws.abs("02_reconstruction/manifest.json").read_text("utf-8"))
        rec_cfg = rec_cfg.get("config", {})
        if (self.out_dir / "test_renders").exists():
            shutil.rmtree(self.out_dir / "test_renders")

        cams = load_cameras(self.ws.abs("01_sfm/cameras.json"))
        test_ids = list(rec.get("test_frame_ids", []))
        train = [c for c in cams if c.frame_id not in set(test_ids)]
        radius = scene_scale(cams)
        self.progress.update(0.0, "načítanie", "gaussiány z 02")
        g = read_ply(src)

        refined, removed, clamped, extra = self.refine(g, train, radius)
        ply = self.out_dir / "gaussians.ply"
        if refined is None:
            shutil.copyfile(src, ply)
            refined = g
        else:
            write_ply(ply, refined)
        outs: dict[str, Path] = {"gaussians": ply}

        sh_degree = int(rec.get("sh_degree", 3))
        antialiased = bool(rec_cfg.get("antialiased", True))
        warnings: list[str] = []
        per_view: list[dict[str, Any]] = []
        if cfg["eval"] and test_ids:
            outs["test_renders"] = self.out_dir / "test_renders"
            by_id = {c.frame_id: c for c in cams}
            size = tuple(rec["eval_resolution"])
            test = [by_id[f] for f in test_ids]
            per_view = self._evaluate(g, refined, test, size, sh_degree, antialiased)
        elif cfg["eval"]:
            warnings.append("02 has no test views, before/after metrics skipped")

        report = {
            "num_input": len(g),
            "num_output": len(refined),
            "removed": removed,
            "clamped": clamped,
            "crop_box": extra.get("crop_box"),
            "scene_radius": round(radius, 6),
            "num_train_views": len(train),
            "before": _mean_metrics(per_view, "before"),
            "after": _mean_metrics(per_view, "after"),
            "per_view": per_view,
            "test_frame_ids": test_ids if per_view else [],
            "eval_resolution": rec.get("eval_resolution") if per_view else None,
            "sh_degree": sh_degree,
            "antialiased": antialiased,
            "lpips_net": "alex" if per_view and cfg["lpips"] else None,
            "warnings": warnings,
        }
        b, a = report["before"]["psnr"], report["after"]["psnr"]
        if b is not None and a is not None and b - a > PSNR_DROP_WARN_DB:
            warnings.append(f"PSNR dropped by {b - a:.2f} dB after refinement")
        outs["report"] = self.out_dir / "report.json"
        outs["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
        return outs

    def _evaluate(
        self,
        before: Gaussians,
        after: Gaussians,
        test: list[Camera],
        size: tuple[int, int],
        sh_degree: int,
        antialiased: bool,
    ) -> list[dict[str, Any]]:
        import torch

        from scenepipe.core.metrics import Lpips, psnr, ssim
        from scenepipe.core.render import render, to_params

        if not torch.cuda.is_available():
            raise RuntimeError("03_refinement: eval needs a CUDA GPU (or set eval: false)")
        cfg = self.config
        renders_dir = self.out_dir / "test_renders"
        renders_dir.mkdir()
        p_before = to_params(before.with_sh_degree(sh_degree))
        p_after = p_before if after is before else to_params(after.with_sh_degree(sh_degree))
        lpips = Lpips() if cfg["lpips"] else None
        n_save = int(cfg["num_test_renders"])
        save = set(np.linspace(0, len(test) - 1, min(n_save, len(test))).round().astype(int))
        stage = self.progress.stage(REFINE_END, 1.0, "vyhodnotenie")
        per_view = []
        with torch.no_grad():
            for i, cam in enumerate(test):
                stage.update(i / len(test), f"testovací pohľad {i + 1}/{len(test)}")
                gt = self._load_gt(cam, size)
                cam_s = cam.scaled(*size)
                view: dict[str, Any] = {"frame_id": cam.frame_id}
                rgbs = []
                for key, params in (("before", p_before), ("after", p_after)):
                    rgb, _, _, _ = render(params, [cam_s], sh_degree, antialiased=antialiased)
                    rgbs.append(rgb[0].clamp(0, 1))
                    view[key] = {
                        "psnr": round(psnr(rgb, gt).item(), 4),
                        "ssim": round(ssim(rgb.clamp(0, 1), gt).item(), 4),
                        "lpips": None if lpips is None else round(lpips(rgb, gt).item(), 4),
                    }
                per_view.append(view)
                if i in save:
                    row = torch.cat([*rgbs, gt[0]], dim=1)  # before | after | reference
                    Image.fromarray((row * 255).round().byte().cpu().numpy()).save(
                        renders_dir / f"{cam.frame_id}.jpg", quality=JPEG_QUALITY
                    )
        return per_view

    def _load_gt(self, cam: Camera, size: tuple[int, int]):
        import torch

        img = load_image(self.ws.abs(cam.image), size=size)
        return torch.from_numpy(img).cuda().float()[None] / 255

    # --- validation ------------------------------------------------------------------------

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        try:
            g = read_ply(outputs["gaussians"])
        except ValueError as e:
            raise ContractError(f"gaussians.ply: {e}") from e
        if len(g) == 0:
            raise ContractError("gaussians.ply has no gaussians")
        for name in ("means", "sh_dc", "sh_rest", "opacity", "scales", "quats"):
            if not np.isfinite(getattr(g, name)).all():
                raise ContractError(f"gaussians.ply: {name} contains NaN/inf")
        if (np.linalg.norm(g.quats, axis=1) < 1e-8).any():
            raise ContractError("gaussians.ply: zero quaternion")

        report = json.loads(outputs["report"].read_text(encoding="utf-8"))
        required = {"num_input", "num_output", "removed", "clamped", "before", "after"}
        required |= {"per_view", "test_frame_ids", "warnings"}
        if missing := required - set(report):
            raise ContractError(f"report.json: missing keys {sorted(missing)}")
        if report["num_output"] != len(g) or not 0 < len(g) <= report["num_input"]:
            raise ContractError("report.json: num_output does not match gaussians.ply")
        if set(report["removed"]) != set(CRITERIA):
            raise ContractError(f"report.json: removed must have keys {list(CRITERIA)}")
        if report["num_input"] - report["num_output"] != sum(report["removed"].values()):
            raise ContractError("report.json: removed counts do not add up")

        if report["per_view"]:
            if not any(outputs["test_renders"].glob("*.jpg")):
                raise ContractError("test_renders/ is empty")
            self._check_consistency(outputs["gaussians"], report)
        for w in report["warnings"]:
            log.warning("03_refinement: %s", w)

    def _check_consistency(self, ply: Path, report: dict[str, Any]) -> None:
        """Render a test view from the saved PLY: PSNR must match the reported `after`."""
        import torch

        from scenepipe.core.metrics import psnr
        from scenepipe.core.render import render, to_params

        view = report["per_view"][0]
        cam = next(
            c
            for c in load_cameras(self.ws.abs("01_sfm/cameras.json"))
            if c.frame_id == view["frame_id"]
        )
        size = tuple(report["eval_resolution"])
        params = to_params(read_ply(ply, sh_degree=report["sh_degree"]))
        with torch.no_grad():
            rgb, _, _, _ = render(
                params, [cam.scaled(*size)], report["sh_degree"], antialiased=report["antialiased"]
            )
            value = psnr(rgb, self._load_gt(cam, size)).item()
        if abs(value - view["after"]["psnr"]) > PSNR_CONSISTENCY_DB:
            raise ContractError(
                f"PLY re-render PSNR {value:.3f} dB != reported {view['after']['psnr']:.3f} dB "
                f"for {view['frame_id']}"
            )


def _mean_metrics(per_view: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for m in ("psnr", "ssim", "lpips"):
        vals = [v[key][m] for v in per_view if v[key][m] is not None]
        out[m] = round(float(np.mean(vals)), 4) if vals else None
    return out
