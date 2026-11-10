"""Quality of gaussians from novel views, without reference images (docs/evaluation.md).

Holdout metrics of 02 (PSNR on test views) only see views on the camera trajectory. Here every
test view of 02 is moved off the trajectory (up along +Z and back against the viewing
direction, in world units; after 01 the RMS camera spread is 1) and rendered, and two
reference-free scores are computed per view:
- hole fraction: share of pixels with accumulated alpha < 0.5 (missing geometry, black areas),
- depth inconsistency: 1 - Pearson correlation between the rendered inverse depth and a
  monocular relative inverse depth of the rendered image itself (Depth Anything V2 Small,
  pixels with alpha > 0.5). Floaters and haze draw something near the camera where the image
  looks like a far wall, so the two disagree.
The same scores on the unshifted test views are the baseline. Note: `depth_reg` of 02 trains
the same correlation on the training views, so it is not independent of that prior.

Needs CUDA (gsplat). Environment `sp-gs`.

    python -m scenepipe.eval.novel_views --scene m360_room
    python -m scenepipe.eval.novel_views --scene m360_room --gaussians 03_refinement --save renders/nv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenepipe.core.cameras import Camera, load_cameras
from scenepipe.core.paths import checkpoint_path
from scenepipe.core.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"  # Apache 2.0, as in phase 02
DEPTH_MODEL_DIR = "checkpoints/huggingface"
DEFAULT_OFFSETS = ((0.0, 0.0), (0.3, 0.3), (0.6, 0.6))  # (up, back); (0, 0) = baseline
MIN_ALPHA = 0.5
MIN_PIXELS = 16


def shifted(cam: Camera, up: float, back: float) -> Camera:
    """Same orientation, center moved by `up` along world +Z and `back` against the view."""
    r, t = cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3]
    center = -r.T @ t + up * np.array([0.0, 0.0, 1.0]) - back * r[2]
    w2c = cam.world_to_camera.copy()
    w2c[:3, 3] = -r @ center
    return Camera(
        cam.frame_id, cam.image, cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, w2c
    )


def depth_inconsistency(inv_depth: np.ndarray, mono: np.ndarray, mask: np.ndarray) -> float | None:
    """1 - Pearson correlation of two [H, W] maps over `mask`; None with too few pixels."""
    if int(mask.sum()) < MIN_PIXELS:
        return None
    x = inv_depth[mask].astype(np.float64)
    y = mono[mask].astype(np.float64)
    x, y = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(1 - (x * y).sum() / denom) if denom > 0 else None


def evaluate(
    ws: Workspace,
    gaussians: str,
    offsets: list[tuple[float, float]],
    scale: float,
    save: Path | None,
) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    from scenepipe.core.gaussians import read_ply
    from scenepipe.core.render import render, to_params

    if not torch.cuda.is_available():
        raise RuntimeError("novel_views needs a CUDA GPU (gsplat)")
    rec = json.loads(ws.abs("02_reconstruction/report.json").read_text(encoding="utf-8"))
    manifest = json.loads(ws.abs("02_reconstruction/manifest.json").read_text(encoding="utf-8"))
    antialiased = bool(manifest.get("config", {}).get("antialiased", True))
    sh_degree = int(rec["sh_degree"])
    by_id = {c.frame_id: c for c in load_cameras(ws.abs("01_sfm/cameras.json"))}
    test = [by_id[f] for f in rec["test_frame_ids"]]
    if not test:
        raise ValueError("02 has no test views (test_every <= 0)")
    w, h = (max(round(v * scale), 1) for v in rec["eval_resolution"])
    params = to_params(read_ply(ws.abs(f"{gaussians}/gaussians.ply"), sh_degree=sh_degree))

    cache = checkpoint_path(DEPTH_MODEL_DIR)
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, cache_dir=cache)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, cache_dir=cache)
    model = model.cuda().eval()
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)

    results = []
    with torch.no_grad():
        for up, back in offsets:
            views = []
            for cam in test:
                c = shifted(cam, up, back).scaled(w, h)
                rgb, z, alpha, _ = render(
                    params, [c], sh_degree, antialiased=antialiased, depth=True
                )
                rgb8 = (rgb[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
                a = alpha[0].cpu().numpy()
                inv = 1 / np.maximum(z[0].cpu().numpy(), 1e-6)
                inp = processor(images=rgb8, return_tensors="pt").to("cuda")
                mono = model(**inp).predicted_depth  # [1, h', w'] relative inverse depth
                mono = (
                    torch.nn.functional.interpolate(
                        mono[None], size=(h, w), mode="bilinear", align_corners=False
                    )[0, 0]
                    .cpu()
                    .numpy()
                )
                views.append(
                    {
                        "frame_id": cam.frame_id,
                        "hole_frac": round(float((a < MIN_ALPHA).mean()), 4),
                        "depth_inconsistency": _round(
                            depth_inconsistency(inv, mono, a > MIN_ALPHA)
                        ),
                    }
                )
                if save is not None:
                    name = f"{cam.frame_id}_up{up:g}_back{back:g}.jpg"
                    Image.fromarray(rgb8).save(save / name, quality=95)
            results.append({"up": up, "back": back, **_means(views), "per_view": views})
    return {
        "scene": ws.root.name,
        "gaussians": f"{gaussians}/gaussians.ply",
        "num_views": len(test),
        "resolution": [w, h],
        "offsets": results,
    }


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def _means(views: list[dict]) -> dict:
    out = {}
    for key in ("hole_frac", "depth_inconsistency"):
        vals = [v[key] for v in views if v[key] is not None]
        out[key] = round(float(np.mean(vals)), 4) if vals else None
    return out


def _offsets(text: str) -> list[tuple[float, float]]:
    """'0,0;0.3,0.3' -> [(0, 0), (0.3, 0.3)]"""
    return [tuple(float(v) for v in item.split(",")) for item in text.split(";")]  # type: ignore[misc]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--gaussians", default="02_reconstruction", help="phase dir with gaussians.ply")
    ap.add_argument(
        "--offsets",
        type=_offsets,
        default=list(DEFAULT_OFFSETS),
        help="'up,back;up,back;...' in world units (default '0,0;0.3,0.3;0.6,0.6')",
    )
    ap.add_argument("--scale", type=float, default=0.5, help="of the 02 eval resolution")
    ap.add_argument("--save", type=Path, help="directory for the rendered views")
    ap.add_argument("--per-view", action="store_true", help="print per-view scores too")
    args = ap.parse_args(argv)
    out = evaluate(
        Workspace.open(Path(args.workspace), args.scene),
        args.gaussians,
        args.offsets,
        args.scale,
        args.save,
    )
    if not args.per_view:
        for o in out["offsets"]:
            o.pop("per_view")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
