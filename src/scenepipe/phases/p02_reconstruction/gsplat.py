"""Phase 02_reconstruction, implementation `gsplat` (own trainer on gsplat). Spec:
phases/02_reconstruction.md

Hyperparameters follow gsplat's examples/simple_trainer.py defaults (Apache 2.0); the code is
written from scratch on top of the public gsplat API.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from gsplat import DefaultStrategy, MCMCStrategy
from PIL import Image

from scenepipe.core.cameras import Camera, load_cameras
from scenepipe.core.gaussians import Gaussians, read_ply, rgb_to_sh_dc, write_ply
from scenepipe.core.metrics import Lpips, psnr, ssim
from scenepipe.core.phase import ContractError, Phase, register
from scenepipe.core.render import from_params, render, to_params

log = logging.getLogger(__name__)

REFERENCE_ITERATIONS = 30_000  # strategy schedules below are tuned for this many steps
SSIM_LAMBDA = 0.2
SH_DEGREE_INTERVAL = 1000
JPEG_QUALITY = 95
PSNR_CONSISTENCY_DB = 0.05
MIN_REFINE_EVERY = 10  # keeps densification sane for short (test) runs
TRAIN_END = 0.95  # share of the phase progress taken by training
PROGRESS_EVERY = 50  # steps between progress updates (writes are throttled anyway)


@register
class ReconstructionGsplat(Phase):
    phase = "02_reconstruction"
    impl = "gsplat"
    impl_version = "1"
    contract_version = 1
    defaults: ClassVar[dict[str, Any]] = {
        "iterations": 30000,
        "sh_degree": 3,
        "strategy": "mcmc",
        "cap_max": 1_000_000,
        "resolution_scale": 1.0,
        "test_every": 8,
        "antialiased": True,
        "export_depth": False,
        "lpips": True,
        "num_test_renders": 8,
        "seed": 0,
    }

    def inputs(self) -> dict[str, Path]:
        return {
            "cameras": self.ws.abs("01_sfm/cameras.json"),
            "images": self.ws.abs("01_sfm/images_undistorted"),
            "points": self.ws.abs("01_sfm/points.npz"),
        }

    # --- run -------------------------------------------------------------------------------

    def run(self) -> dict[str, Path]:
        cfg = self.config
        if not torch.cuda.is_available():
            raise RuntimeError("02_reconstruction/gsplat needs a CUDA GPU")
        if cfg["strategy"] not in ("default", "mcmc"):
            raise ValueError(f"unknown strategy {cfg['strategy']!r}")
        torch.manual_seed(int(cfg["seed"]))
        np.random.seed(int(cfg["seed"]))
        torch.cuda.reset_peak_memory_stats()
        for name in ("test_renders", "depth"):
            if (self.out_dir / name).exists():
                shutil.rmtree(self.out_dir / name)

        cams = load_cameras(self.ws.abs("01_sfm/cameras.json"))
        train, test = split_views(cams, int(cfg["test_every"]))
        if not train:
            raise ValueError("02_reconstruction: no training views")
        self.progress.update(0.0, "načítanie snímok", f"{len(cams)} kamier")
        images = {c.frame_id: self._load_image(c) for c in cams}
        scaled = {}  # cameras at the training resolution
        for c in cams:
            h, w = images[c.frame_id].shape[:2]
            scaled[c.frame_id] = c.scaled(w, h)

        scene_scale = camera_scene_scale(cams)
        params = self._init_params(scene_scale)
        train_s = [scaled[c.frame_id] for c in train]
        t0 = time.perf_counter()
        self._train(params, train_s, images, scene_scale)
        train_time = time.perf_counter() - t0

        self.progress.update(TRAIN_END, "export PLY", f"{len(params['means'])} gaussiánov")
        g = from_params(params)
        ply = self.out_dir / "gaussians.ply"
        write_ply(ply, g)

        outs: dict[str, Path] = {"gaussians": ply}
        per_view = self._evaluate(params, [scaled[c.frame_id] for c in test], images)
        outs["test_renders"] = self.out_dir / "test_renders"
        if cfg["export_depth"]:
            outs["depth"] = self._export_depth(params, train)

        report = self._report(g, train_s, test, per_view, train_time, scene_scale)
        outs["report"] = self.out_dir / "report.json"
        outs["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
        return outs

    def _load_image(self, cam: Camera) -> torch.Tensor:
        """uint8 [H,W,3] on the CPU at the training resolution."""
        s = float(self.config["resolution_scale"])
        return load_image(self.ws.abs(cam.image), s)

    def _init_params(self, scene_scale: float) -> torch.nn.ParameterDict:
        cfg = self.config
        mcmc = cfg["strategy"] == "mcmc"
        with np.load(self.ws.abs("01_sfm/points.npz")) as pts:
            xyz, rgb = pts["xyz"].astype(np.float32), pts["rgb"].astype(np.float32) / 255
        if mcmc and len(xyz) > int(cfg["cap_max"]):
            keep = np.random.default_rng(int(cfg["seed"])).choice(
                len(xyz), int(cfg["cap_max"]), replace=False
            )
            xyz, rgb = xyz[keep], rgb[keep]
        n = len(xyz)
        # isotropic scale from the mean distance to the 3 nearest neighbours
        dist = knn_mean_dist(torch.from_numpy(xyz).cuda(), k=3).cpu().numpy()
        init_scale, init_opacity = (0.1, 0.5) if mcmc else (1.0, 0.1)
        log_scale = np.log(np.maximum(dist * init_scale, 1e-7))
        k = (int(cfg["sh_degree"]) + 1) ** 2 - 1
        g = Gaussians(
            means=xyz,
            sh_dc=rgb_to_sh_dc(rgb),
            sh_rest=np.zeros((n, k, 3), np.float32),
            opacity=np.full(n, math.log(init_opacity / (1 - init_opacity)), np.float32),
            scales=np.repeat(log_scale[:, None], 3, axis=1).astype(np.float32),
            quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        )
        return torch.nn.ParameterDict(
            {name: torch.nn.Parameter(t) for name, t in to_params(g).items()}
        )

    def _train(
        self,
        params: torch.nn.ParameterDict,
        train: list[Camera],
        images: dict[str, torch.Tensor],
        scene_scale: float,
    ) -> None:
        cfg = self.config
        iters = int(cfg["iterations"])
        lrs = {
            "means": 1.6e-4 * scene_scale,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20,
        }
        optimizers = {
            name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}], eps=1e-15)
            for name, lr in lrs.items()
        }
        means_sched = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max(iters, 1))
        )
        strategy, state = self._strategy(iters, scene_scale)
        strategy.check_sanity(params, optimizers)

        gen = torch.Generator().manual_seed(int(cfg["seed"]))
        order: list[int] = []
        log_every = max(iters // 20, 1)
        for step in range(iters):
            if not order:
                order = torch.randperm(len(train), generator=gen).tolist()
            cam = train[order.pop()]
            gt = images[cam.frame_id].cuda(non_blocking=True).float()[None] / 255
            sh_degree = min(step // SH_DEGREE_INTERVAL, int(cfg["sh_degree"]))
            rgb, _, _, info = render(
                params,
                [cam],
                sh_degree,
                antialiased=bool(cfg["antialiased"]),
                absgrad=getattr(strategy, "absgrad", False),
            )
            if isinstance(strategy, DefaultStrategy):
                strategy.step_pre_backward(params, optimizers, state, step, info)
            l1 = torch.abs(rgb - gt).mean()
            loss = (1 - SSIM_LAMBDA) * l1 + SSIM_LAMBDA * (1 - ssim(rgb, gt))
            if isinstance(strategy, MCMCStrategy):  # regularizers from the MCMC paper
                loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean()
                loss = loss + 0.01 * torch.exp(params["scales"]).mean()
            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            means_sched.step()
            if isinstance(strategy, DefaultStrategy):
                strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
            else:
                lr = means_sched.get_last_lr()[0]
                strategy.step_post_backward(params, optimizers, state, step, info, lr=lr)
            if step % PROGRESS_EVERY == 0:
                self.progress.update(
                    TRAIN_END * (step + 1) / iters,
                    "tréning",
                    f"krok {step + 1}/{iters}, N={len(params['means'])}, loss {loss.item():.4f}",
                )
            if step % log_every == 0 or step == iters - 1:
                log.info(
                    "step %d/%d loss %.4f N=%d", step + 1, iters, loss.item(), len(params["means"])
                )

    def _strategy(self, iters: int, scene_scale: float):
        """gsplat strategy with its schedule scaled to the iteration count."""
        f = iters / REFERENCE_ITERATIONS
        if self.config["strategy"] == "mcmc":
            s = MCMCStrategy(
                cap_max=int(self.config["cap_max"]),
                refine_start_iter=round(500 * f),
                refine_stop_iter=round(25_000 * f),
                refine_every=max(round(100 * f), MIN_REFINE_EVERY),
            )
            return s, s.initialize_state()
        s = DefaultStrategy(
            refine_start_iter=round(500 * f),
            refine_stop_iter=round(15_000 * f),
            reset_every=max(round(3000 * f), 10 * MIN_REFINE_EVERY),
            refine_every=max(round(100 * f), MIN_REFINE_EVERY),
        )
        return s, s.initialize_state(scene_scale=scene_scale)

    @torch.no_grad()
    def _evaluate(
        self, params: torch.nn.ParameterDict, test: list[Camera], images: dict[str, torch.Tensor]
    ) -> list[dict[str, Any]]:
        cfg = self.config
        renders_dir = self.out_dir / "test_renders"
        renders_dir.mkdir()
        if not test:
            return []
        lpips = Lpips() if cfg["lpips"] else None
        n_save = int(cfg["num_test_renders"])
        save = set(np.linspace(0, len(test) - 1, min(n_save, len(test))).round().astype(int))
        per_view = []
        stage = self.progress.stage(TRAIN_END + 0.01, 1.0, "vyhodnotenie")
        for i, cam in enumerate(test):
            stage.update(i / len(test), f"testovací pohľad {i + 1}/{len(test)}")
            gt = images[cam.frame_id].cuda().float()[None] / 255
            rgb = self._render_eval(params, cam)
            per_view.append(
                {
                    "frame_id": cam.frame_id,
                    "psnr": round(psnr(rgb, gt).item(), 4),
                    "ssim": round(ssim(rgb.clamp(0, 1), gt).item(), 4),
                    "lpips": None if lpips is None else round(lpips(rgb, gt).item(), 4),
                }
            )
            if i in save:
                pair = torch.cat([rgb[0].clamp(0, 1), gt[0]], dim=1)
                Image.fromarray((pair * 255).round().byte().cpu().numpy()).save(
                    renders_dir / f"{cam.frame_id}.jpg", quality=JPEG_QUALITY
                )
        return per_view

    def _render_eval(self, params, cam: Camera) -> torch.Tensor:
        rgb, _, _, _ = render(
            params,
            [cam],
            int(self.config["sh_degree"]),
            antialiased=bool(self.config["antialiased"]),
        )
        return rgb

    @torch.no_grad()
    def _export_depth(self, params: torch.nn.ParameterDict, train: list[Camera]) -> Path:
        """Expected z-depth at the full cameras.json resolution; 0 where nothing is rendered."""
        depth_dir = self.out_dir / "depth"
        depth_dir.mkdir()
        for cam in train:
            _, z, alpha, _ = render(
                params,
                [cam],
                int(self.config["sh_degree"]),
                antialiased=bool(self.config["antialiased"]),
                depth=True,
            )
            z = torch.where(alpha > 1e-3, z, torch.zeros_like(z))[0]
            np.save(depth_dir / f"{cam.frame_id}.npy", z.cpu().numpy().astype(np.float16))
        return depth_dir

    def _report(self, g, train, test, per_view, train_time, scene_scale) -> dict[str, Any]:
        def mean(key: str) -> float | None:
            vals = [v[key] for v in per_view if v[key] is not None]
            return round(float(np.mean(vals)), 4) if vals else None

        warnings = []
        if not test:
            warnings.append("no test views (test_every <= 0 or too few cameras)")
        return {
            "num_gaussians": len(g),
            "iterations": int(self.config["iterations"]),
            "train_time_s": round(train_time, 1),
            "psnr": mean("psnr"),
            "ssim": mean("ssim"),
            "lpips": mean("lpips"),
            "lpips_net": "alex" if self.config["lpips"] else None,
            "sh_degree": int(self.config["sh_degree"]),
            "strategy": self.config["strategy"],
            "test_frame_ids": [c.frame_id for c in test],
            "num_train_views": len(train),
            "eval_resolution": [train[0].width, train[0].height],
            "scene_scale": round(scene_scale, 6),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "gpu": torch.cuda.get_device_name(0),
            "per_view": per_view,
            "warnings": warnings,
        }

    # --- validation ------------------------------------------------------------------------

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        cfg = self.config
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

        cams = load_cameras(self.ws.abs("01_sfm/cameras.json"))
        with np.load(self.ws.abs("01_sfm/points.npz")) as pts:
            extent_pts = np.vstack([pts["xyz"], np.stack([c.center for c in cams])])
        extent = float(np.linalg.norm(extent_pts.max(0) - extent_pts.min(0)))
        if float(np.exp(g.scales.max())) > extent:
            raise ContractError(
                f"gaussians.ply: max scale {np.exp(g.scales.max()):.3g} > scene size {extent:.3g}"
            )

        report = json.loads(outputs["report"].read_text(encoding="utf-8"))
        required = {"num_gaussians", "iterations", "train_time_s", "psnr", "ssim", "lpips"}
        required |= {"sh_degree", "test_frame_ids", "per_view"}
        if missing := required - set(report):
            raise ContractError(f"report.json: missing keys {sorted(missing)}")
        if report["num_gaussians"] != len(g) or report["sh_degree"] != int(cfg["sh_degree"]):
            raise ContractError("report.json does not match gaussians.ply")
        by_id = {c.frame_id: c for c in cams}
        if not set(report["test_frame_ids"]) <= set(by_id):
            raise ContractError("report.json: unknown test frame_id")
        if report["test_frame_ids"] and not any(outputs["test_renders"].glob("*.jpg")):
            raise ContractError("test_renders/ is empty")

        # export/import consistency: render a test view from the saved PLY
        if report["per_view"]:
            view = report["per_view"][0]
            cam = by_id[view["frame_id"]]
            gt_img = self._load_image(cam)
            params = to_params(read_ply(outputs["gaussians"], sh_degree=int(cfg["sh_degree"])))
            with torch.no_grad():
                rgb = self._render_eval(params, cam.scaled(gt_img.shape[1], gt_img.shape[0]))
                value = psnr(rgb, gt_img.cuda().float()[None] / 255).item()
            if abs(value - view["psnr"]) > PSNR_CONSISTENCY_DB:
                raise ContractError(
                    f"PLY re-render PSNR {value:.3f} dB != reported {view['psnr']:.3f} dB "
                    f"for {view['frame_id']} (export/import mismatch)"
                )

        if cfg["export_depth"]:
            train_ids = sorted(set(by_id) - set(report["test_frame_ids"]))
            for fid in train_ids:
                path = outputs["depth"] / f"{fid}.npy"
                if not path.is_file():
                    raise ContractError(f"depth/{fid}.npy missing")
                d = np.load(path)
                if d.dtype != np.float16 or d.shape != (by_id[fid].height, by_id[fid].width):
                    raise ContractError(f"depth/{fid}.npy: dtype/shape {d.dtype} {d.shape}")
        for w in report["warnings"]:
            log.warning("02_reconstruction: %s", w)


# --- helpers ---------------------------------------------------------------------------------


def split_views(cams: list[Camera], test_every: int) -> tuple[list[Camera], list[Camera]]:
    """Every test_every-th camera (0, k, 2k, ...) is held out; test_every <= 0: no holdout."""
    if test_every <= 0 or len(cams) < 2:
        return list(cams), []
    test = [c for i, c in enumerate(cams) if i % test_every == 0]
    train = [c for i, c in enumerate(cams) if i % test_every != 0]
    return train, test


def camera_scene_scale(cams: list[Camera]) -> float:
    """Max camera distance from the mean camera center, x1.1 (gsplat convention)."""
    centers = np.stack([c.center for c in cams])
    return float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1) or 1.0


def load_image(path: Path, scale: float) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if scale != 1.0:
            size = (max(round(im.width * scale), 1), max(round(im.height * scale), 1))
            im = im.resize(size, Image.Resampling.BOX if scale < 1 else Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy())


def knn_mean_dist(xyz: torch.Tensor, k: int = 3, max_elems: int = 1 << 25) -> torch.Tensor:
    """Mean distance of every point to its k nearest neighbours (brute force, chunked)."""
    n = len(xyz)
    if n <= 1:
        return torch.full((n,), 0.01, device=xyz.device)
    k = min(k, n - 1)
    chunk = max(max_elems // n, 1)
    out = torch.empty(n, device=xyz.device)
    for i in range(0, n, chunk):
        d = torch.cdist(xyz[i : i + chunk], xyz)
        d[torch.arange(len(d)), torch.arange(i, i + len(d))] = float("inf")
        out[i : i + chunk] = d.topk(k, largest=False).values.mean(1)
    return out.clamp_min(1e-7)
