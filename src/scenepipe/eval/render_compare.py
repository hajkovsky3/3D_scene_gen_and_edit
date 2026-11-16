"""Side-by-side renders of several gaussian sets (variants of 02/03) for a visual comparison.

Rows are views (camera frames of 01, optionally moved off the trajectory as in novel_views),
columns are the variants; the grid is saved as one JPEG with a caption per column.

Needs CUDA (gsplat). Environment `sp-gs`.

    python -m scenepipe.eval.render_compare --scene m360_room \
        --gaussians 03_refinement/_cmp/keep0/gaussians.ply 03_refinement/_cmp/keep5/gaussians.ply \
        --frames f000289 f000033 --offsets "0,0;0.3,0.3" --out 03_refinement/_cmp/keep.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scenepipe.core.cameras import load_cameras
from scenepipe.core.gaussians import read_ply
from scenepipe.core.workspace import Workspace
from scenepipe.eval.novel_views import shifted

REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTION_H = 28


def render_grid(
    ws: Workspace,
    gaussians: list[str],
    frames: list[str],
    offsets: list[tuple[float, float]],
    width: int,
    labels: list[str] | None = None,
) -> Image.Image:
    import torch

    from scenepipe.core.render import render, to_params

    cams = {c.frame_id: c for c in load_cameras(ws.abs("01_sfm/cameras.json"))}
    views = [(cams[f], up, back) for f in frames for up, back in offsets]
    first = cams[frames[0]]
    height = round(width * first.height / first.width)
    labels = labels or [Path(g).parent.name or g for g in gaussians]
    columns = []
    for path in gaussians:
        g = read_ply(ws.abs(path))
        params = to_params(g)
        tiles = []
        with torch.no_grad():
            for cam, up, back in views:
                c = shifted(cam, up, back).scaled(width, height)
                rgb, _, _, _ = render(params, [c], g.sh_degree)
                tiles.append((rgb[0].clamp(0, 1) * 255).round().byte().cpu().numpy())
        columns.append(np.concatenate(tiles, axis=0))
        del params
        torch.cuda.empty_cache()
    grid = np.concatenate(columns, axis=1)
    out = Image.new("RGB", (grid.shape[1], grid.shape[0] + CAPTION_H), (255, 255, 255))
    out.paste(Image.fromarray(grid), (0, CAPTION_H))
    draw = ImageDraw.Draw(out)
    for i, lab in enumerate(labels):
        draw.text((i * width + 6, 6), lab, fill=(0, 0, 0))
    for r, (cam, up, back) in enumerate(views):
        draw.text((6, CAPTION_H + r * height + 6), f"{cam.frame_id} up {up:g} back {back:g}",
                  fill=(255, 255, 0))  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", default=str(REPO_ROOT / "workspace"))
    ap.add_argument("--gaussians", nargs="+", required=True, help="PLY paths in the workspace")
    ap.add_argument("--labels", nargs="*", help="column captions (default: parent directory)")
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--offsets", default="0,0", help="'up,back;up,back' in world units")
    ap.add_argument("--width", type=int, default=640, help="tile width, px")
    ap.add_argument("--out", required=True, help="output JPEG, relative to the workspace")
    args = ap.parse_args()
    ws = Workspace.open(Path(args.workspace), args.scene)
    offsets = [tuple(float(v) for v in s.split(",")) for s in args.offsets.split(";")]
    img = render_grid(ws, args.gaussians, args.frames, offsets, args.width, args.labels)
    out = ws.abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=92)
    print(out)


if __name__ == "__main__":
    main()
