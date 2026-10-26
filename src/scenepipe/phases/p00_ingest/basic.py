"""Phase 00_ingest, implementation `basic` (OpenCV + Pillow). Spec: phases/00_ingest.md"""

from __future__ import annotations

import io
import json
import logging
import shutil
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from scenepipe.core.phase import ContractError, Phase, register
from scenepipe.core.progress import StageProgress

log = logging.getLogger(__name__)

PHOTO_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}
REJECT_REASONS = {"unsupported_format", "unreadable", "blur", "duplicate", "over_max_frames"}
JPEG_QUALITY = 95
SHARPNESS_SIDE = 512
THUMB_SIZE = 64
JPEG_WRITERS = 2  # threads encoding staged JPEGs while the video is decoded
MOTION_MAX_CORNERS = 300
MOTION_MIN_TRACKED = 20  # fewer tracked corners: motion from phase correlation instead

# EXIF tags (Exif IFD 0x8769)
_EXIF_IFD = 0x8769
_TAG_DATETIME_ORIGINAL = 36867
_TAG_FOCAL = 37386
_TAG_FOCAL_35MM = 41989


@dataclass
class _Frame:
    """A frame that passed loading; its image is already written to the staging dir."""

    source: str
    source_index: int | None
    timestamp_s: float | None
    staged: Path
    width: int
    height: int
    sharpness: float
    thumb: np.ndarray = field(repr=False)
    group: str  # blur percentile is computed per group (one video, or all photos)
    exif_focal_mm: float | None = None
    exif_focal_35mm: float | None = None
    exif_datetime: str | None = None


@register
class IngestBasic(Phase):
    phase = "00_ingest"
    impl = "basic"
    impl_version = "3"
    contract_version = 1
    defaults: ClassVar[dict[str, Any]] = {
        "video_fps": 3,
        "sharpness_window": 5,
        "sharpness_percentile_reject": 30,
        "dedup_threshold": 0.97,
        "max_side": 1600,
        "max_frames": 300,
        "min_frames": 20,
        "blur_filter_photos": False,
        "blur_max_gap_s": 1.0,
        "blur_gap_min_ratio": 0.3,
        "keyframe_motion": 0.12,
        "keyframe_max_gap_s": 2.0,
        "video_hw_decode": False,
    }

    @property
    def input_dir(self) -> Path:
        return self.ws.root / "input"

    def inputs(self) -> dict[str, Path]:
        files = _list_inputs(self.input_dir)
        if not files:
            # Reported as a missing input by Phase.execute.
            return {"input": self.input_dir / "*"}
        return {self.ws.rel(p): p for p in files}

    def steps(self) -> list[str]:
        return ["načítanie vstupov", "filtrovanie snímok", "zápis snímok"]

    def run(self) -> dict[str, Path]:
        images_dir = self.out_dir / "images"
        staging = self.out_dir / "_staging"
        for d in (images_dir, staging):
            if d.exists():
                shutil.rmtree(d)
        staging.mkdir(parents=True)

        stager = _Stager(staging, int(self.config["max_side"]))
        try:
            frames, rejected = self._load_inputs(stager)
        finally:
            stager.close()  # all staged JPEGs are on disk from here on

        self.progress.update(0.9, "filtrovanie snímok", f"{len(frames)} kandidátov")
        frames = self._filter_blur(frames, rejected)
        frames = self._filter_duplicates(frames, rejected)
        frames = self._limit_count(frames, rejected)

        self.progress.update(0.95, "zápis snímok", f"{len(frames)} snímok")
        images_dir.mkdir()
        entries = []
        for i, f in enumerate(frames, start=1):
            frame_id = f"f{i:06d}"
            dst = images_dir / f"{frame_id}.jpg"
            f.staged.replace(dst)
            entries.append(
                {
                    "frame_id": frame_id,
                    "image": self.ws.rel(dst),
                    "width": f.width,
                    "height": f.height,
                    "source": f.source,
                    "source_index": f.source_index,
                    "timestamp_s": None if f.timestamp_s is None else round(f.timestamp_s, 3),
                    "sharpness": round(f.sharpness, 2),
                    "exif_focal_mm": f.exif_focal_mm,
                    "exif_focal_35mm": f.exif_focal_35mm,
                }
            )
        shutil.rmtree(staging)

        rejected.sort(key=lambda r: (r["source"], r["source_index"] or 0))
        frames_json = self.out_dir / "frames.json"
        frames_json.write_text(
            json.dumps(
                {
                    "contract_version": self.contract_version,
                    "frames": entries,
                    "rejected": rejected,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info("00_ingest: %d frames kept, %d rejected", len(entries), len(rejected))
        return {"frames": frames_json, "images": images_dir}

    def _load_inputs(self, stager: _Stager) -> tuple[list[_Frame], list[dict[str, Any]]]:
        rejected: list[dict[str, Any]] = []
        videos: list[_Frame] = []
        photos: list[_Frame] = []
        files = _list_inputs(self.input_dir)
        for n_file, path in enumerate(files):
            rel = self.ws.rel(path)
            ext = path.suffix.lower()
            stage = self.progress.stage(
                0.9 * n_file / len(files), 0.9 * (n_file + 1) / len(files), "načítanie vstupov"
            )
            stage.update(0.0, rel)
            if ext in VIDEO_EXTS:
                frames = self._load_video(path, rel, stager, stage)
                if frames is None:
                    rejected.append(_rej(rel, None, "unreadable"))
                else:
                    videos.extend(frames)
            elif ext in PHOTO_EXTS:
                frame = _load_photo(path, rel, stager)
                if frame is None:
                    rejected.append(_rej(rel, None, "unreadable"))
                else:
                    photos.append(frame)
            else:
                rejected.append(_rej(rel, None, "unsupported_format"))

        if photos and all(f.exif_datetime for f in photos):
            photos.sort(key=lambda f: (f.exif_datetime, f.source))
        return videos + photos, rejected  # videos are already in path order, then time order

    def _load_video(
        self, path: Path, rel: str, stager: _Stager, stage: StageProgress
    ) -> list[_Frame] | None:
        """Sample candidates at video_fps * window, keep the sharpest of each window.

        A window is `sharpness_window` consecutive candidates or, with keyframe_motion > 0,
        a stretch of camera motion of that size (see _MotionMeter), at most keyframe_max_gap_s
        long. Candidates only get a small grayscale copy for the sharpness; the winner of a
        window is converted, resized and staged.
        """
        cap = _open_video(path, bool(self.config["video_hw_decode"]))
        if cap is None:
            return None
        cfg = self.config
        window = max(1, int(cfg["sharpness_window"]))
        rate = float(cfg["video_fps"]) * window  # candidate slots per second
        motion_step = float(cfg["keyframe_motion"])
        max_gap = float(cfg["keyframe_max_gap_s"])
        meter = _MotionMeter() if motion_step > 0 else None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        out: list[_Frame] = []
        best: tuple[float, np.ndarray, int, float] | None = None  # sharpness, bgr, idx, t
        best_window = -1
        motion = 0.0  # accumulated camera motion since the start of the video
        win_start = (0.0, 0.0)  # motion and time at the start of the current motion window
        last_slot = -1
        idx = -1

        def emit() -> None:
            sharp, bgr, i, ts = best
            rgb = stager.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            out.append(stager.stage(rgb, rel, i, ts, sharp, group=rel))

        try:
            while cap.grab():
                idx += 1
                if total > 0 and idx % 30 == 0:
                    stage.update(idx / total, f"{rel}: snímka {idx}/{total}")
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = pos_ms / 1000.0 if (pos_ms > 0 or idx == 0 or fps <= 0) else idx / fps
                slot = int(t * rate)
                if slot == last_slot:
                    continue
                last_slot = slot
                ok, bgr = cap.retrieve()
                if not ok:
                    continue
                gray = cv2.cvtColor(_resize_max(bgr, SHARPNESS_SIDE), cv2.COLOR_BGR2GRAY)
                sharp = _laplacian_var(gray)
                if meter is None:
                    win = slot // window
                    new_window = win != best_window
                    best_window = win
                else:
                    motion += meter.step(gray)
                    new_window = (
                        best is None
                        or motion - win_start[0] >= motion_step
                        or t - win_start[1] >= max_gap
                    )
                    if new_window:
                        win_start = (motion, t)
                if new_window and best is not None:
                    emit()
                    best = None
                if best is None or sharp > best[0]:
                    best = (sharp, bgr, idx, t)
        finally:
            cap.release()
        if best is not None:
            emit()
        if idx < 0:
            return None
        return out

    def _filter_blur(self, frames: list[_Frame], rejected: list[dict]) -> list[_Frame]:
        pct = float(self.config["sharpness_percentile_reject"])
        if pct <= 0:
            return frames
        groups: dict[str, list[float]] = {}
        for f in frames:
            if f.group == "photos" and not self.config["blur_filter_photos"]:
                continue
            groups.setdefault(f.group, []).append(f.sharpness)
        thresholds = {g: float(np.percentile(v, pct)) for g, v in groups.items()}
        blurry = [f.group in thresholds and f.sharpness < thresholds[f.group] for f in frames]
        self._fill_gaps(frames, blurry)
        kept = []
        for f, bad in zip(frames, blurry, strict=True):
            if bad:
                rejected.append(_rej(f.source, f.source_index, "blur"))
                f.staged.unlink()
            else:
                kept.append(f)
        return kept

    def _fill_gaps(self, frames: list[_Frame], blurry: list[bool]) -> None:
        """Greedily keep blurry video frames so that kept frames are <= blur_max_gap_s apart.

        A blurry frame is kept when the next candidate of the same video is more than
        blur_max_gap_s after the last kept frame. Blurry frames before the first and after
        the last sharp frame of a video are not kept, nor frames below blur_gap_min_ratio
        times the median sharpness of their video (those tend to be mislocalized by SfM).
        """
        max_gap = float(self.config["blur_max_gap_s"])
        if max_gap <= 0:
            return
        min_ratio = float(self.config["blur_gap_min_ratio"])
        groups: dict[str, list[float]] = {}
        for f in frames:
            groups.setdefault(f.group, []).append(f.sharpness)
        floor = {g: min_ratio * float(np.median(v)) for g, v in groups.items()}
        last_t: dict[str, float] = {}  # time of the last kept frame per video
        for i, f in enumerate(frames):
            if f.timestamp_s is None:
                continue
            if blurry[i]:
                prev = last_t.get(f.group)
                later = [
                    (g, b) for g, b in zip(frames[i + 1 :], blurry[i + 1 :]) if g.group == f.group
                ]
                if prev is None or not any(not b for _, b in later):
                    continue  # before the first / after the last sharp frame: stays rejected
                if later[0][0].timestamp_s - prev <= max_gap:
                    continue  # the next candidate still closes the gap
                if f.sharpness < floor[f.group]:
                    continue  # too blurry even to fill a gap (would be mislocalized in 01)
                blurry[i] = False
            last_t[f.group] = f.timestamp_s

    def _filter_duplicates(self, frames: list[_Frame], rejected: list[dict]) -> list[_Frame]:
        thr = float(self.config["dedup_threshold"])
        kept: list[_Frame] = []
        for f in frames:
            if kept and _ncc(kept[-1].thumb, f.thumb) > thr:
                rejected.append(_rej(f.source, f.source_index, "duplicate"))
                f.staged.unlink()
            else:
                kept.append(f)
        return kept

    def _limit_count(self, frames: list[_Frame], rejected: list[dict]) -> list[_Frame]:
        n, cap = len(frames), int(self.config["max_frames"])
        if n <= cap:
            return frames
        keep = set(np.round(np.linspace(0, n - 1, cap)).astype(int).tolist())
        kept = []
        for i, f in enumerate(frames):
            if i in keep:
                kept.append(f)
            else:
                rejected.append(_rej(f.source, f.source_index, "over_max_frames"))
                f.staged.unlink()
        return kept

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        data = json.loads(outputs["frames"].read_text(encoding="utf-8"))
        if data.get("contract_version") != self.contract_version:
            raise ContractError(f"frames.json contract_version {data.get('contract_version')}")
        frames, rejected = data["frames"], data["rejected"]

        lo, hi = int(self.config["min_frames"]), int(self.config["max_frames"])
        if not lo <= len(frames) <= hi:
            reasons = dict(Counter(r["reason"] for r in rejected))
            raise ContractError(
                f"00_ingest kept {len(frames)} frames, expected {lo}..{hi}. "
                f"Rejected: {reasons}. Add more input, lower video_fps/dedup or "
                f"sharpness thresholds, or lower min_frames."
            )

        max_side = int(self.config["max_side"])
        expected_files = set()
        for i, fr in enumerate(frames, start=1):
            fid = f"f{i:06d}"
            if fr["frame_id"] != fid:
                raise ContractError(f"frame_id {fr['frame_id']!r} at position {i}, expected {fid}")
            if fr["image"] != f"00_ingest/images/{fid}.jpg":
                raise ContractError(f"{fid}: unexpected image path {fr['image']!r}")
            img_path = self.ws.abs(fr["image"])
            if not img_path.is_file():
                raise ContractError(f"{fid}: image file missing")
            with Image.open(img_path) as im:
                size = im.size
            if size != (fr["width"], fr["height"]):
                raise ContractError(f"{fid}: size {size} != json {fr['width']}x{fr['height']}")
            if max(size) > max_side:
                raise ContractError(f"{fid}: long side {max(size)} > max_side {max_side}")
            expected_files.add(img_path.name)

        actual = {p.name for p in outputs["images"].iterdir()}
        if actual != expected_files:
            raise ContractError(f"images/ has extra files: {sorted(actual - expected_files)}")
        bad = {r["reason"] for r in rejected} - REJECT_REASONS
        if bad:
            raise ContractError(f"unknown reject reasons {sorted(bad)}")


class _Stager:
    """Resizes frames and writes them to the staging dir, keeping only metadata in memory."""

    def __init__(self, staging: Path, max_side: int):
        self.staging = staging
        self.max_side = max_side
        self.count = 0
        self._pool = ThreadPoolExecutor(JPEG_WRITERS)  # Pillow releases the GIL while encoding
        self._pending: list[Future] = []

    def close(self) -> None:
        """Wait until all staged JPEGs are written (re-raises write errors)."""
        try:
            for f in self._pending:
                f.result()
        finally:
            self._pool.shutdown()
            self._pending = []

    def resize(self, rgb: np.ndarray) -> np.ndarray:
        return _resize_max(rgb, self.max_side)

    def stage(
        self,
        rgb: np.ndarray,
        source: str,
        source_index: int | None,
        timestamp_s: float | None,
        sharpness: float,
        group: str,
        **exif: Any,
    ) -> _Frame:
        path = self.staging / f"s{self.count:06d}.jpg"
        self.count += 1
        image = Image.fromarray(rgb)
        self._pending.append(self._pool.submit(image.save, path, quality=JPEG_QUALITY))
        h, w = rgb.shape[:2]
        return _Frame(
            source=source,
            source_index=source_index,
            timestamp_s=timestamp_s,
            staged=path,
            width=w,
            height=h,
            sharpness=sharpness,
            thumb=_thumb(rgb),
            group=group,
            **exif,
        )


def _load_photo(path: Path, rel: str, stager: _Stager) -> _Frame | None:
    try:
        with Image.open(path) as im:
            im.load()
            exif = _read_exif(im)
            im = ImageOps.exif_transpose(im)
            im = _to_srgb(im)
    except (OSError, ValueError, SyntaxError) as e:
        log.warning("cannot read %s: %s", rel, e)
        return None
    rgb = stager.resize(np.asarray(im))
    return stager.stage(rgb, rel, None, None, _sharpness(rgb), group="photos", **exif)


def _read_exif(im: Image.Image) -> dict[str, Any]:
    exif = im.getexif()
    sub = exif.get_ifd(_EXIF_IFD)

    def num(v: Any) -> float | None:
        try:
            x = float(v)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return round(x, 3) if x > 0 else None

    dt = sub.get(_TAG_DATETIME_ORIGINAL)
    return {
        "exif_focal_mm": num(sub.get(_TAG_FOCAL)),
        "exif_focal_35mm": num(sub.get(_TAG_FOCAL_35MM)),
        "exif_datetime": str(dt).strip() if dt else None,
    }


def _to_srgb(im: Image.Image) -> Image.Image:
    icc = im.info.get("icc_profile")
    if icc and im.mode in ("RGB", "RGBA", "CMYK"):
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            return ImageCms.profileToProfile(im.convert("RGB"), src, dst, outputMode="RGB")
        except (ImageCms.PyCMSError, OSError) as e:
            log.warning("ICC conversion failed, assuming sRGB: %s", e)
    return im.convert("RGB")


def _resize_max(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return np.ascontiguousarray(rgb)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _sharpness(rgb: np.ndarray) -> float:
    return _laplacian_var(cv2.cvtColor(_resize_max(rgb, SHARPNESS_SIDE), cv2.COLOR_RGB2GRAY))


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _open_video(path: Path, hw_decode: bool) -> cv2.VideoCapture | None:
    """OpenCV capture; with hw_decode the GPU decoder is used if available (else software)."""
    params = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY] if hw_decode else []
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, params)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(path))  # any other backend
    if not cap.isOpened():
        return None
    return cap


class _MotionMeter:
    """Camera motion between consecutive video candidates, as a fraction of the image size.

    Corners of the previous frame are tracked into the current one (pyramidal Lucas-Kanade);
    the motion is their median displacement divided by the long image side. Unlike a global
    shift this also grows when the camera moves forward. With too few trackable corners
    (blank wall, heavy blur) the global shift from phase correlation is used instead.
    """

    def __init__(self) -> None:
        self.prev: np.ndarray | None = None

    def step(self, gray: np.ndarray) -> float:
        prev, self.prev = self.prev, gray
        if prev is None or prev.shape != gray.shape:
            return 0.0
        side = float(max(gray.shape))
        corners = cv2.goodFeaturesToTrack(prev, MOTION_MAX_CORNERS, 0.01, 8)
        if corners is not None and len(corners) >= MOTION_MIN_TRACKED:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, corners, None)
            ok = status.ravel() == 1
            if ok.sum() >= MOTION_MIN_TRACKED:
                d = np.linalg.norm((moved - corners).reshape(-1, 2)[ok], axis=1)
                return float(np.median(d)) / side
        (dx, dy), _ = cv2.phaseCorrelate(prev.astype(np.float32), gray.astype(np.float32))
        return float(np.hypot(dx, dy)) / side


def _thumb(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    t = cv2.resize(gray, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
    t = t.astype(np.float32)
    return (t - t.mean()) / (t.std() + 1e-6)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).mean())


def _rej(source: str, source_index: int | None, reason: str) -> dict[str, Any]:
    return {"source": source, "source_index": source_index, "reason": reason}


def _list_inputs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.name != "Thumbs.db"
    )
