"""Phase 01_sfm, implementation `colmap` (COLMAP 4.x via pycolmap). Spec: phases/01_sfm.md"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pycolmap
from PIL import Image

from scenepipe.core.phase import ContractError, Phase, register
from scenepipe.phases.p01_sfm.align import compute_alignment
from scenepipe.phases.p01_sfm.retrieval import bow_similarity, select_pairs
from scenepipe.phases.p01_sfm.torch_match import cuda_available, match_pairs

log = logging.getLogger(__name__)

JPEG_QUALITY = 95
MAX_MEAN_REPROJ_PX = 2.0
DIAG_35MM = 43.27  # diagonal of a 36x24 mm frame
MATCH_CHUNKS = 20  # matching is split into ~this many calls for progress reporting
MIN_MATCH_CHUNK = 100


@register
class SfmColmap(Phase):
    phase = "01_sfm"
    impl = "colmap"
    impl_version = "3"
    contract_version = 2
    defaults: ClassVar[dict[str, Any]] = {
        "features": "sift",
        "matcher": "auto",
        "matcher_backend": "default",
        "mapper": "global",
        "fallback_incremental_below": 0.8,
        "single_camera": True,
        "align_up": "floor_ransac",
        "use_gpu": True,
        "max_num_features": 8192,
        "sift_max_image_size": -1,
        "sequential_overlap": 5,
        "retrieval_num_neighbors": 5,
        "retrieval_num_words": 1024,
        "retrieval_sample": 100_000,
        "auto_exhaustive_max": 100,
        "max_camera_dist_ratio": 5.0,
        "min_registered": 10,
        "keep_work": False,
        "num_threads": -1,
        "seed": 0,
    }

    def inputs(self) -> dict[str, Path]:
        return {
            "frames": self.ws.abs("00_ingest/frames.json"),
            "images": self.ws.abs("00_ingest/images"),
        }

    def run(self) -> dict[str, Path]:
        self._check_supported()
        for name in ("_work", "sparse", "images_undistorted"):
            if (self.out_dir / name).exists():
                shutil.rmtree(self.out_dir / name)
        # The COLMAP database is written and read many times per pair: keep the work dir on
        # a local disk (system temp), the workspace may be a slow network/WSL mount.
        work = Path(tempfile.mkdtemp(prefix="scenepipe_01_sfm_"))
        try:
            return self._run(work)
        finally:
            if self.config["keep_work"]:
                shutil.copytree(work, self.out_dir / "_work", dirs_exist_ok=True)
            shutil.rmtree(work, ignore_errors=True)

    def _run(self, work: Path) -> dict[str, Path]:
        cfg = self.config
        frames = json.loads(self.ws.abs("00_ingest/frames.json").read_text(encoding="utf-8"))
        frames = frames["frames"]
        images_dir = self.ws.abs("00_ingest/images")
        names = [Path(f["image"]).name for f in frames]
        db = work / "database.db"

        # 1. features
        timer = _Timer()
        self.progress.update(0.0, "extrakcia príznakov", f"SIFT, {len(names)} snímok")
        single = self._single_camera(frames)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        prior = _prior_focal_px(frames[0]) if single else None
        if prior is not None:
            w, h = frames[0]["width"], frames[0]["height"]
            reader.camera_params = f"{prior},{w / 2},{h / 2},0"
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.num_threads = int(cfg["num_threads"])
        extraction.sift.max_num_features = int(cfg["max_num_features"])
        if int(cfg["sift_max_image_size"]) > 0:
            extraction.max_image_size = int(cfg["sift_max_image_size"])
        pycolmap.extract_features(
            db,
            images_dir,
            image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE if single else pycolmap.CameraMode.AUTO,
            reader_options=reader,
            extraction_options=extraction,
            device=pycolmap.Device.cpu,
        )
        timer.lap("features")

        # 2.-3. matching + geometric verification
        matcher = self._matcher(frames)
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = int(cfg["num_threads"])
        self.progress.update(0.08, "výber dvojíc", matcher)
        if matcher == "retrieval":
            pairs = self._retrieval_pairs(db, names)
        elif matcher == "sequential":
            pairs = select_pairs(None, len(names), int(cfg["sequential_overlap"]), 0)
        else:
            pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
        timer.lap("pairs")
        device = self._match_device()
        if device == "cuda":
            self._match_pairs_torch(db, names, pairs, matching, work)
        else:
            self._match_pairs(db, names, pairs, matching, work)
        timer.lap("matching")

        # 4.-5. mapping (+ bundle adjustment)
        self.progress.update(0.85, "mapping", f"{cfg['mapper']} mapper")
        models: dict[str, list[pycolmap.Reconstruction]] = {}
        if cfg["mapper"] == "global":
            models["global"] = self._map_global(db, images_dir, work / "global")
        best_mapper, best = _largest(models)
        ratio = 0.0 if best is None else best.num_reg_images() / len(frames)
        if cfg["mapper"] == "incremental" or ratio < float(cfg["fallback_incremental_below"]):
            if cfg["mapper"] == "global":
                log.info("global mapper registered %.0f%%, trying incremental", 100 * ratio)
                self.progress.update(0.9, "mapping", "incremental mapper (fallback)")
            models["incremental"] = self._map_incremental(db, images_dir, work / "incremental")
            best_mapper, best = _largest(models)
        if best is None:
            raise RuntimeError("01_sfm: COLMAP did not produce any reconstruction")
        removed = self._remove_outlier_cameras(best)
        raw = work / "raw"
        raw.mkdir()
        best.write_binary(raw)
        timer.lap("mapping")

        # 6. undistortion -> PINHOLE cameras + new images
        self.progress.update(0.95, "undistorcia", f"{best.num_reg_images()} kamier")
        undist = work / "undistorted"
        pycolmap.undistort_images(
            undist, raw, images_dir, output_type="COLMAP", jpeg_quality=JPEG_QUALITY
        )
        recon = pycolmap.Reconstruction(undist / "sparse")
        timer.lap("undistortion")

        # 7. alignment (+Z up, centered, unit RMS camera spread)
        self.progress.update(0.98, "zarovnanie a export")
        image_ids = sorted(recon.reg_image_ids(), key=lambda i: recon.image(i).name)
        rots = np.stack([_cam_from_world(recon.image(i))[:3, :3] for i in image_ids])
        centers = np.stack([recon.image(i).projection_center() for i in image_ids])
        xyz_raw = np.array([p.xyz for p in recon.points3D.values()], dtype=np.float64)
        alignment = compute_alignment(cfg["align_up"], rots, centers, xyz_raw.reshape(-1, 3))
        m = alignment.matrix
        sim3 = pycolmap.Sim3d(
            alignment.scale, pycolmap.Rotation3d(m[:3, :3] / alignment.scale), m[:3, 3]
        )
        recon.transform(sim3)
        recon.update_point_3d_errors()

        sparse = self.out_dir / "sparse"
        sparse.mkdir()
        recon.write_binary(sparse)
        shutil.move(undist / "images", self.out_dir / "images_undistorted")

        # 8. exports
        outs = {
            "sparse": sparse,
            "images_undistorted": self.out_dir / "images_undistorted",
            "cameras": self._write_cameras(recon, image_ids),
            "points": self._write_points(recon),
            "alignment": self._write_json("alignment.json", alignment.to_json()),
        }
        timer.lap("export")
        report = self._report(recon, frames, models, best_mapper, matcher, single, removed)
        report["num_pairs"] = len(pairs)
        report["match_device"] = device
        report["timings_s"] = timer.laps
        outs["report"] = self._write_json("report.json", report)
        return outs

    def _check_supported(self) -> None:
        cfg = self.config
        if cfg["features"] != "sift" or cfg["matcher_backend"] != "default":
            raise NotImplementedError(
                "01_sfm/colmap: only features=sift with matcher_backend=default; ALIKED and "
                "LightGlue need model downloads that are not approved yet (phases/01_sfm.md)"
            )
        if cfg["matcher"] not in ("auto", "sequential", "exhaustive", "retrieval"):
            raise NotImplementedError(f"01_sfm/colmap: matcher {cfg['matcher']!r} not supported")
        if cfg["mapper"] not in ("global", "incremental"):
            raise ValueError(f"unknown mapper {cfg['mapper']!r}")

    def _single_camera(self, frames: list[dict]) -> bool:
        if not self.config["single_camera"]:
            return False
        keys = {(f["width"], f["height"], f.get("exif_focal_35mm")) for f in frames}
        return len(keys) == 1

    def _matcher(self, frames: list[dict]) -> str:
        m = self.config["matcher"]
        if m != "auto":
            return m
        if len(frames) <= int(self.config["auto_exhaustive_max"]):
            return "exhaustive"
        return "retrieval"

    def _retrieval_pairs(self, db_path: Path, names: list[str]) -> list[tuple[int, int]]:
        """Pairs (indices into names) for the `retrieval` matcher; frames in frame_id order."""
        cfg = self.config
        db = pycolmap.Database.open(db_path)
        try:
            ids = {img.name: img.image_id for img in db.read_all_images()}
            descs = [np.asarray(db.read_descriptors(ids[n]).data) for n in names]
        finally:
            db.close()
        sim = None
        if int(cfg["retrieval_num_neighbors"]) > 0:
            sim = bow_similarity(
                descs,
                int(cfg["retrieval_num_words"]),
                int(cfg["retrieval_sample"]),
                seed=int(cfg["seed"]),
            )
        pairs = select_pairs(
            sim, len(names), int(cfg["sequential_overlap"]), int(cfg["retrieval_num_neighbors"])
        )
        total = len(names) * (len(names) - 1) // 2
        log.info("retrieval matcher: %d pairs (exhaustive would be %d)", len(pairs), total)
        return pairs

    def _match_pairs(
        self,
        db: Path,
        names: list[str],
        pairs: list[tuple[int, int]],
        matching: pycolmap.FeatureMatchingOptions,
        work: Path,
    ) -> None:
        """Match + verify the given pairs in chunks, so that progress can be reported."""
        stage = self.progress.stage(0.1, 0.85, "párovanie")
        chunk = max(len(pairs) // MATCH_CHUNKS, MIN_MATCH_CHUNK)
        pairs_file = work / "pairs.txt"
        pairing = pycolmap.ImportedPairingOptions()
        pairing.match_list_path = pairs_file
        for start in range(0, len(pairs), chunk):
            stage.update(start / len(pairs), f"dvojica {start} z {len(pairs)}")
            part = pairs[start : start + chunk]
            pairs_file.write_text(
                "".join(f"{names[i]} {names[j]}\n" for i, j in part), encoding="utf-8"
            )
            pycolmap.match_image_pairs(
                db, matching_options=matching, pairing_options=pairing, device=pycolmap.Device.cpu
            )
        stage.update(1.0, f"{len(pairs)} dvojíc")

    def _match_device(self) -> str:
        """`cuda` when use_gpu is set and PyTorch sees a CUDA GPU, else `cpu` (COLMAP)."""
        if self.config["use_gpu"] and cuda_available():
            return "cuda"
        if self.config["use_gpu"]:
            log.info("01_sfm: no PyTorch with CUDA, matching on the CPU with COLMAP")
        return "cpu"

    def _match_pairs_torch(
        self,
        db_path: Path,
        names: list[str],
        pairs: list[tuple[int, int]],
        matching: pycolmap.FeatureMatchingOptions,
        work: Path,
    ) -> None:
        """Nearest-neighbour matching on the GPU, geometric verification in COLMAP."""
        sift = matching.sift
        db = pycolmap.Database.open(db_path)
        try:
            ids = {img.name: img.image_id for img in db.read_all_images()}
            descs = [np.asarray(db.read_descriptors(ids[n]).data) for n in names]
            stage = self.progress.stage(0.1, 0.4, "párovanie (GPU)")
            for i, j, m in match_pairs(
                descs,
                pairs,
                max_ratio=sift.max_ratio,
                max_distance=sift.max_distance,
                cross_check=sift.cross_check,
                on_progress=lambda done: stage.update(done / len(pairs), f"{done}/{len(pairs)}"),
            ):
                db.write_matches(ids[names[i]], ids[names[j]], m[: matching.max_num_matches])
        finally:
            db.close()
        self._verify_pairs(db_path, names, pairs, work)

    def _verify_pairs(
        self, db: Path, names: list[str], pairs: list[tuple[int, int]], work: Path
    ) -> None:
        """Geometric verification of already matched pairs, in chunks for progress reporting."""
        stage = self.progress.stage(0.4, 0.85, "geometrická verifikácia")
        chunk = max(len(pairs) // MATCH_CHUNKS, MIN_MATCH_CHUNK)
        pairs_file = work / "pairs.txt"
        for start in range(0, len(pairs), chunk):
            stage.update(start / len(pairs), f"dvojica {start} z {len(pairs)}")
            part = pairs[start : start + chunk]
            pairs_file.write_text(
                "".join(f"{names[i]} {names[j]}\n" for i, j in part), encoding="utf-8"
            )
            pycolmap.verify_matches(db, pairs_file, pycolmap.TwoViewGeometryOptions())
        stage.update(1.0, f"{len(pairs)} dvojíc")

    def _map_global(self, db: Path, images: Path, out: Path) -> list[pycolmap.Reconstruction]:
        out.mkdir()
        opts = pycolmap.GlobalPipelineOptions()
        opts.num_threads = int(self.config["num_threads"])
        opts.random_seed = int(self.config["seed"])
        try:
            return list(pycolmap.global_mapping(db, images, out, opts).values())
        except (RuntimeError, ValueError) as e:  # raised by pycolmap when mapping fails
            log.warning("global mapper failed: %s", e)
            return []

    def _map_incremental(self, db: Path, images: Path, out: Path) -> list[pycolmap.Reconstruction]:
        out.mkdir()
        opts = pycolmap.IncrementalPipelineOptions()
        opts.num_threads = int(self.config["num_threads"])
        opts.random_seed = int(self.config["seed"])
        try:
            return list(pycolmap.incremental_mapping(db, images, out, opts).values())
        except (RuntimeError, ValueError) as e:
            log.warning("incremental mapper failed: %s", e)
            return []

    def _write_cameras(self, recon: pycolmap.Reconstruction, image_ids: list[int]) -> Path:
        cams = []
        for i in image_ids:
            img = recon.image(i)
            cam = recon.camera(img.camera_id)
            if cam.model_name != "PINHOLE":
                raise ContractError(f"camera model {cam.model_name} after undistortion")
            fx, fy, cx, cy = (float(v) for v in cam.params)
            w2c = np.eye(4)
            w2c[:3, :] = _cam_from_world(img)
            cams.append(
                {
                    "frame_id": Path(img.name).stem,
                    "image": f"01_sfm/images_undistorted/{img.name}",
                    "width": int(cam.width),
                    "height": int(cam.height),
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "world_to_camera": w2c.tolist(),
                }
            )
        data = {
            "contract_version": self.contract_version,
            "convention": "opencv_world_to_camera",
            "up_axis": "+z",
            "cameras": cams,
        }
        return self._write_json("cameras.json", data)

    def _write_points(self, recon: pycolmap.Reconstruction) -> Path:
        pts = list(recon.points3D.values())
        path = self.out_dir / "points.npz"
        np.savez_compressed(
            path,
            xyz=np.array([p.xyz for p in pts], dtype=np.float32).reshape(-1, 3),
            rgb=np.array([p.color for p in pts], dtype=np.uint8).reshape(-1, 3),
            error=np.array([p.error for p in pts], dtype=np.float32),
            track_len=np.array([p.track.length() for p in pts], dtype=np.int32),
        )
        return path

    def _remove_outlier_cameras(self, recon: pycolmap.Reconstruction) -> list[str]:
        """De-register cameras far away from all others (mislocalized, e.g. blurry frames).

        A camera is an outlier if its distance from the median camera center is more than
        `max_camera_dist_ratio` times the median of these distances. Such cameras would
        dominate the normalization (step 7) and the scene scale in phase 02.
        """
        ratio = float(self.config["max_camera_dist_ratio"])
        ids = list(recon.reg_image_ids())
        if ratio <= 0 or len(ids) < 5:
            return []
        centers = np.stack([recon.image(i).projection_center() for i in ids])
        d = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
        limit = ratio * float(np.median(d))
        removed = []
        for i, di in zip(ids, d, strict=True):
            if di > limit:
                img = recon.image(i)
                removed.append(Path(img.name).stem)
                recon.deregister_frame(img.frame_id)
        if removed:
            log.warning("01_sfm: removed %d outlier cameras: %s", len(removed), removed)
        return sorted(removed)

    def _report(self, recon, frames, models, mapper, matcher, single, removed) -> dict[str, Any]:
        registered = {Path(recon.image(i).name).stem for i in recon.reg_image_ids()}
        n_reg = len(registered)
        err = float(recon.compute_mean_reprojection_error())
        warnings = []
        if removed:
            warnings.append(
                f"removed {len(removed)} outlier cameras far from all others: {removed}"
            )
        if err > MAX_MEAN_REPROJ_PX:
            warnings.append(f"mean reprojection error {err:.2f} px > {MAX_MEAN_REPROJ_PX} px")
        return {
            "num_frames": len(frames),
            "num_registered": n_reg,
            "registered_ratio": round(n_reg / len(frames), 4),
            "num_points": recon.num_points3D(),
            "mean_reprojection_error_px": round(err, 4),
            "mean_track_length": round(float(recon.compute_mean_track_length()), 3),
            "unregistered": [f["frame_id"] for f in frames if f["frame_id"] not in registered],
            "mapper": mapper,
            "matcher": matcher,
            "camera_mode": "single" if single else "auto",
            "models": {
                k: sorted((r.num_reg_images() for r in v), reverse=True) for k, v in models.items()
            },
            "warnings": warnings,
        }

    def _write_json(self, name: str, data: dict) -> Path:
        path = self.out_dir / name
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        cams = json.loads(outputs["cameras"].read_text(encoding="utf-8"))
        if cams.get("contract_version") != self.contract_version:
            raise ContractError("cameras.json: wrong contract_version")
        if cams.get("convention") != "opencv_world_to_camera" or cams.get("up_axis") != "+z":
            raise ContractError("cameras.json: wrong convention/up_axis")
        cam_list = cams["cameras"]
        if len(cam_list) < int(self.config["min_registered"]):
            raise ContractError(
                f"01_sfm registered only {len(cam_list)} cameras "
                f"(min_registered={self.config['min_registered']}); see report.json"
            )
        ids = [c["frame_id"] for c in cam_list]
        if len(set(ids)) != len(ids) or ids != sorted(ids):
            raise ContractError("cameras.json: frame_ids must be unique and sorted")

        ups = []
        for c in cam_list:
            fid = c["frame_id"]
            m = np.asarray(c["world_to_camera"], dtype=np.float64)
            if m.shape != (4, 4) or not np.allclose(m[3], [0, 0, 0, 1]):
                raise ContractError(f"{fid}: world_to_camera must be 4x4 with last row 0001")
            r = m[:3, :3]
            if not np.allclose(r @ r.T, np.eye(3), atol=1e-5) or abs(np.linalg.det(r) - 1) > 1e-5:
                raise ContractError(f"{fid}: rotation is not orthonormal with det 1")
            w, h = c["width"], c["height"]
            if min(c["fx"], c["fy"], w, h) <= 0 or not (0 < c["cx"] < w and 0 < c["cy"] < h):
                raise ContractError(f"{fid}: invalid intrinsics")
            if c["image"] != f"01_sfm/images_undistorted/{fid}.jpg":
                raise ContractError(f"{fid}: unexpected image path {c['image']!r}")
            with Image.open(self.ws.abs(c["image"])) as im:
                if im.size != (w, h):
                    raise ContractError(f"{fid}: image size {im.size} != {w}x{h}")
            ups.append(-r[1, 2])  # z component of the camera's -y axis in world

        alignment = json.loads(outputs["alignment"].read_text(encoding="utf-8"))
        if alignment["up_method"] != "none" and np.mean(np.array(ups) > 0) <= 0.5:
            raise ContractError("most cameras are upside down after alignment (+Z up check)")

        with np.load(outputs["points"]) as pts:
            xyz = pts["xyz"]
            p = len(xyz)
            if p == 0:
                raise ContractError("points.npz is empty")
            expected = {"xyz": (p, 3), "rgb": (p, 3), "error": (p,), "track_len": (p,)}
            for key, shape in expected.items():
                if pts[key].shape != shape:
                    raise ContractError(f"points.npz {key}: shape {pts[key].shape} != {shape}")
            if not (np.isfinite(xyz).all() and np.isfinite(pts["error"]).all()):
                raise ContractError("points.npz contains NaN/inf")

        recon = pycolmap.Reconstruction(outputs["sparse"])
        if recon.num_reg_images() != len(cam_list) or recon.num_points3D() != p:
            raise ContractError("sparse/ does not match cameras.json / points.npz counts")
        report = json.loads(outputs["report"].read_text(encoding="utf-8"))
        for w in report["warnings"]:
            log.warning("01_sfm: %s", w)


def _cam_from_world(image: pycolmap.Image) -> np.ndarray:
    return np.asarray(image.cam_from_world().matrix(), dtype=np.float64)


def _largest(
    models: dict[str, list[pycolmap.Reconstruction]],
) -> tuple[str | None, pycolmap.Reconstruction | None]:
    best: tuple[str | None, pycolmap.Reconstruction | None] = (None, None)
    for name, recs in models.items():
        for r in recs:
            if best[1] is None or r.num_reg_images() > best[1].num_reg_images():
                best = (name, r)
    return best


def _prior_focal_px(frame: dict) -> float | None:
    f35 = frame.get("exif_focal_35mm")
    if not f35:
        return None
    diag = float(np.hypot(frame["width"], frame["height"]))
    return round(f35 * diag / DIAG_35MM, 2)


class _Timer:
    """Wall-clock seconds of consecutive steps, for report.json."""

    def __init__(self) -> None:
        self.laps: dict[str, float] = {}
        self._last = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = round(now - self._last, 2)
        self._last = now
