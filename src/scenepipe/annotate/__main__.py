"""Annotation tool: python -m scenepipe.annotate --scene m360_counter [--frames 12]

Opens a local web page (http://localhost:8765). Runs in the environment of phase 05 (`sp-gs`,
SAM 2 from transformers); without SAM (`--no-sam`) masks are drawn with the brush only.
The server runs only while this command runs (Ctrl+C ends it).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image

from scenepipe.annotate.store import AnnotatedObject, AnnotationStore, label_of
from scenepipe.core.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[3]
PAGE = Path(__file__).with_name("page.html")
SAM_MODEL = "facebook/sam2.1-hiera-small"

log = logging.getLogger("scenepipe.annotate")


def choose_frames(ws: Workspace, n: int) -> list[str]:
    """`n` frames spread evenly over the frames of 05 (so that every 05 run has masks there,
    `fast` uses every 3rd frame), else over the cameras of 01."""
    det = ws.abs("05_segmentation/detections.json")
    if det.is_file():
        frames = json.loads(det.read_text("utf-8"))["frames"]
    else:
        cams = json.loads(ws.abs("01_sfm/cameras.json").read_text("utf-8"))["cameras"]
        frames = [c["frame_id"] for c in cams]
    if n >= len(frames):
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in sorted(set(idx))]


def pack_mask(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(bool).ravel()).tobytes()).decode("ascii")


def unpack_mask(data: str, width: int, height: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(data), dtype=np.uint8))
    return bits[: width * height].reshape(height, width).astype(bool)


class Sam:
    """SAM 2 image predictor with the image embedding of the last frame cached."""

    def __init__(self, model_id: str, cache_dir: str | None) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = Sam2Processor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = Sam2Model.from_pretrained(model_id, cache_dir=cache_dir).to(self.device)
        self.model.eval()
        self._key: str | None = None
        self._emb = None
        self._lock = threading.Lock()

    def predict(
        self, key: str, image: Image.Image, points: list[list[float]], box: list[float] | None
    ) -> np.ndarray:
        torch = self.torch
        with self._lock, torch.inference_mode():
            if self._key != key:
                pix = self.proc(images=image, return_tensors="pt")["pixel_values"]
                self._emb = self.model.get_image_embeddings(pix.to(self.device))
                self._key = key
            kwargs = {}
            if points:
                kwargs["input_points"] = [[[[p[0], p[1]] for p in points]]]
                kwargs["input_labels"] = [[[int(p[2]) for p in points]]]
            if box:
                kwargs["input_boxes"] = [[box]]
            inputs = self.proc(images=image, return_tensors="pt", **kwargs)
            out = self.model(
                image_embeddings=self._emb,
                input_points=_to(inputs.get("input_points"), self.device),
                input_labels=_to(inputs.get("input_labels"), self.device),
                input_boxes=_to(inputs.get("input_boxes"), self.device),
                multimask_output=len(points) == 1 and not box,
            )
            masks = self.proc.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"], binarize=True
            )[0]
            scores = out.iou_scores[0, 0].cpu().numpy()
            return masks[0, int(np.argmax(scores))].numpy().astype(bool)


def _to(x, device):
    return None if x is None else x.to(device)


class App:
    def __init__(self, ws: Workspace, store: AnnotationStore, sam: Sam | None) -> None:
        self.ws = ws
        self.store = store
        self.sam = sam

    @lru_cache(maxsize=4)  # noqa: B019 - one App per process
    def image(self, frame_id: str) -> Image.Image:
        with Image.open(self.ws.abs(self.store.image_rel(frame_id))) as im:
            return im.convert("RGB")

    def state(self) -> dict:
        frames = []
        instances: dict[str, int] = {}
        for f in self.store.frames():
            objs = self.store.load_frame(f)
            frames.append({"frame_id": f, "objects": len(objs)})
            for o in objs:
                instances[o.instance] = instances.get(o.instance, 0) + 1
        # no suggestions from 04: the vocabulary is the ground truth for the recall of 04
        return {
            "scene": self.ws.root.name,
            "frames": frames,
            "instances": sorted(instances.items()),
            "vocabulary": self.store.vocabulary(),
            "sam": self.sam is not None,
        }

    def frame(self, frame_id: str) -> dict:
        im = self.image(frame_id)
        return {
            "frame_id": frame_id,
            "width": im.width,
            "height": im.height,
            "objects": [
                {"instance": o.instance, "mask": pack_mask(o.mask)}
                for o in self.store.load_frame(frame_id)
            ],
        }

    def save_frame(self, frame_id: str, body: dict) -> dict:
        im = self.image(frame_id)
        objs = [
            AnnotatedObject(
                o["instance"].strip(),
                label_of(o["instance"]),
                unpack_mask(o["mask"], im.width, im.height),
            )
            for o in body["objects"]
            if o["instance"].strip()
        ]
        self.store.save_frame(frame_id, im.width, im.height, objs)
        voc = set(self.store.vocabulary()) | {o.label for o in objs}
        self.store.save_vocabulary(sorted(voc))
        return {"ok": True, "objects": len(objs)}

    def sam_mask(self, frame_id: str, body: dict) -> dict:
        if self.sam is None:
            return {"error": "SAM is not loaded (--no-sam)"}
        mask = self.sam.predict(frame_id, self.image(frame_id), body["points"], body.get("box"))
        return {"mask": pack_mask(mask)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter than the default
            log.debug(fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: dict, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode("utf-8"), "application/json")

        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0])
            try:
                if path == "/":
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._json(app.state())
                elif path.startswith("/api/image/"):
                    fid = path.rsplit("/", 1)[1]
                    self._send(200, app.ws.abs(app.store.image_rel(fid)).read_bytes(), "image/jpeg")
                elif path.startswith("/api/frame/"):
                    self._json(app.frame(path.rsplit("/", 1)[1]))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # report to the page instead of dropping the connection
                log.exception("GET %s", path)
                self._json({"error": str(e)}, 500)

        def do_POST(self):
            path = unquote(self.path.split("?", 1)[0])
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if path.startswith("/api/frame/"):
                    self._json(app.save_frame(path.rsplit("/", 1)[1], body))
                elif path.startswith("/api/sam/"):
                    self._json(app.sam_mask(path.rsplit("/", 1)[1], body))
                elif path == "/api/vocabulary":
                    app.store.save_vocabulary(body["labels"])
                    self._json({"ok": True, "vocabulary": app.store.vocabulary()})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                log.exception("POST %s", path)
                self._json({"error": str(e)}, 500)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--workspace", type=Path, default=REPO_ROOT / "workspace")
    ap.add_argument("--frames", type=int, default=12, help="frames to annotate (first run only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-sam", action="store_true", help="brush only, no model")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ws = Workspace.open(args.workspace, args.scene)
    store = AnnotationStore(ws)
    if not store.exists():
        store.set_frames(choose_frames(ws, args.frames))
    sam = None
    if not args.no_sam:
        from scenepipe.core.paths import checkpoint_path

        cache = str(checkpoint_path("checkpoints/huggingface"))
        log.info("loading %s ...", SAM_MODEL)
        sam = Sam(SAM_MODEL, cache if os.path.isdir(cache) else None)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(App(ws, store, sam)))
    log.info("%d frames, annotations in %s", len(store.frames()), store.root)
    log.info("open http://localhost:%d  (Ctrl+C to stop)", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
