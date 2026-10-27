"""Phase 04_tagging: shared mechanics of all implementations (frame sampling, normalization,
synonyms, filtering, aggregation, user vocabulary, validation). Spec: phases/04_tagging.md

An implementation only provides `tag_frames()`; it returns raw tags with scores per frame.
Everything after that is plain Python, so the phase logic is testable without a model.
"""

from __future__ import annotations

import json
import logging
import re
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml

from scenepipe.core.phase import ContractError, Phase
from scenepipe.core.progress import StageProgress

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
USER_VOCABULARY = "input/vocabulary.txt"
SOURCES = ("model", "user")
KINDS = ("object", "stuff")
TAGGING_SHARE = 0.95  # share of the phase progress taken by tag_frames()


class TaggingBase(Phase):
    phase = "04_tagging"
    contract_version = 1
    defaults: ClassVar[dict[str, Any]] = {
        "sample_frames": 30,
        "min_frames": 2,
        "score_threshold": 0.6,
        "high_score": 0.85,
        "keep_stuff_classes": True,
        "blacklist_file": "configs/tag_blacklist.txt",
        "stuff_file": "configs/tag_stuff.txt",
        "synonyms_file": "configs/tag_synonyms.yaml",
    }
    uses_model: ClassVar[bool] = True  # False: vocabulary only from input/vocabulary.txt

    def inputs(self) -> dict[str, Path]:
        ins = {"frames": self.ws.abs("00_ingest/frames.json")}
        if self.uses_model:
            ins["images"] = self.ws.abs("00_ingest/images")
        user = self.ws.abs(USER_VOCABULARY)
        if user.is_file() or not self.uses_model:  # required by `fixed`
            ins["user_vocabulary"] = user
        return ins

    @abstractmethod
    def tag_frames(
        self, frames: list[tuple[str, Path]], stage: StageProgress
    ) -> dict[str, list[tuple[str, float]]]:
        """Raw tags with scores in [0, 1] for each (frame_id, image path); may include tags
        below `score_threshold` (they are dropped here)."""

    def model_steps(self) -> list[str]:
        """Progress stages reported by tag_frames(), in order."""
        return ["tagovanie"]

    def steps(self) -> list[str]:
        return [*(self.model_steps() if self.uses_model else []), "agregácia"]

    # --- run -------------------------------------------------------------------------------

    def run(self) -> dict[str, Path]:
        cfg = self.config
        frames = json.loads(self.ws.abs("00_ingest/frames.json").read_text("utf-8"))["frames"]
        synonyms = read_synonyms(_repo_path(cfg["synonyms_file"]))
        blacklist = {canonical(t, synonyms) for t in read_terms(_repo_path(cfg["blacklist_file"]))}
        stuff = {canonical(t, synonyms) for t in read_terms(_repo_path(cfg["stuff_file"]))}
        user_file = self.ws.abs(USER_VOCABULARY)
        user = (
            [canonical(t, synonyms) for t in read_terms(user_file)] if user_file.is_file() else []
        )

        raw: dict[str, list[tuple[str, float]]] = {}
        if self.uses_model:
            picked = sample_frames(frames, int(cfg["sample_frames"]))
            log.info("tagging %d of %d frames", len(picked), len(frames))
            stage = self.progress.stage(0.0, TAGGING_SHARE, self.model_steps()[0])
            raw = self.tag_frames([(f["frame_id"], self.ws.abs(f["image"])) for f in picked], stage)

        self.progress.update(TAGGING_SHARE, "agregácia", "")
        per_frame = {
            fid: normalize_frame_tags(tags, synonyms, float(cfg["score_threshold"]))
            for fid, tags in raw.items()
        }
        vocabulary, excluded = aggregate(
            per_frame,
            user,
            blacklist,
            stuff,
            min_frames=int(cfg["min_frames"]),
            high_score=float(cfg["high_score"]),
            keep_stuff=bool(cfg["keep_stuff_classes"]),
        )
        log.info(
            "vocabulary: %d tags (%s); excluded %d",
            len(vocabulary),
            ", ".join(v["tag"] for v in vocabulary[:30]),
            len(excluded),
        )
        out = self.out_dir / "tags.json"
        doc = {
            "contract_version": self.contract_version,
            "per_frame": {
                fid: [{"tag": t, "score": round(s, 4)} for t, s in tags]
                for fid, tags in per_frame.items()
            },
            "vocabulary": vocabulary,
            "excluded": excluded,
        }
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"tags": out}

    # --- validation ------------------------------------------------------------------------

    def validate_outputs(self, outputs: dict[str, Path]) -> None:
        doc = json.loads(outputs["tags"].read_text("utf-8"))
        if doc.get("contract_version") != self.contract_version:
            raise ContractError(f"contract_version {doc.get('contract_version')!r}")
        frames = json.loads(self.ws.abs("00_ingest/frames.json").read_text("utf-8"))["frames"]
        frame_ids = {f["frame_id"] for f in frames}
        unknown = set(doc["per_frame"]) - frame_ids
        if unknown:
            raise ContractError(f"per_frame has unknown frame ids {sorted(unknown)[:5]}")
        for fid, tags in doc["per_frame"].items():
            for t in tags:
                if not isinstance(t["tag"], str) or not 0.0 <= t["score"] <= 1.0:
                    raise ContractError(f"bad tag entry in {fid}: {t}")
        vocab = doc["vocabulary"]
        if not vocab:
            raise ContractError(
                "empty vocabulary (no tag passed the filters; add input/vocabulary.txt)"
            )
        tags = [v["tag"] for v in vocab]
        if len(set(tags)) != len(tags):
            raise ContractError("duplicate tags in vocabulary")
        for v in vocab:
            if not v["tag"] or v["source"] not in SOURCES or v["kind"] not in KINDS:
                raise ContractError(f"bad vocabulary entry {v}")
            if v["frames"] < 0 or not (v["max_score"] is None or 0.0 <= v["max_score"] <= 1.0):
                raise ContractError(f"bad vocabulary stats {v}")
        leaked = set(tags) & set(doc["excluded"])
        if leaked:
            raise ContractError(f"excluded tags in vocabulary: {sorted(leaked)}")


# --- pure helpers (unit-tested) ---------------------------------------------------------------


def _repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def sample_frames(frames: list[dict], k: int) -> list[dict]:
    """k frames evenly spread over the sequence (all of them if k <= 0 or k >= len)."""
    if k <= 0 or k >= len(frames):
        return list(frames)
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, k)).astype(int))
    return [frames[i] for i in idx]


def normalize(tag: str) -> str:
    """Lowercase, single spaces, no surrounding punctuation."""
    return re.sub(r"\s+", " ", tag.strip().lower()).strip(" .,;:")


def canonical(tag: str, synonyms: dict[str, str]) -> str:
    t = normalize(tag)
    return synonyms.get(t, t)


def read_terms(path: Path) -> list[str]:
    """One term per line; empty lines and `#` comments are skipped. Missing file = no terms."""
    if not path.is_file():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = normalize(line.split("#", 1)[0])
        if t:
            terms.append(t)
    return terms


def read_synonyms(path: Path) -> dict[str, str]:
    """YAML `canonical: [alias, ...]` -> {alias: canonical}. Missing file = no synonyms."""
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for target, aliases in data.items():
        for alias in aliases or []:
            a, c = normalize(str(alias)), normalize(str(target))
            if a in out and out[a] != c:
                raise ValueError(f"{path}: {a!r} is a synonym of both {out[a]!r} and {c!r}")
            out[a] = c
    return out


def normalize_frame_tags(
    tags: list[tuple[str, float]], synonyms: dict[str, str], threshold: float
) -> list[tuple[str, float]]:
    """Canonical tags with score >= threshold, merged by max score, sorted by score."""
    best: dict[str, float] = {}
    for tag, score in tags:
        if score < threshold:
            continue
        t = canonical(tag, synonyms)
        if t:
            best[t] = max(best.get(t, 0.0), float(score))
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


def aggregate(
    per_frame: dict[str, list[tuple[str, float]]],
    user: list[str],
    blacklist: set[str],
    stuff: set[str],
    *,
    min_frames: int,
    high_score: float,
    keep_stuff: bool,
) -> tuple[list[dict], list[str]]:
    """Vocabulary entries and the sorted list of excluded tags.

    A model tag enters the vocabulary if it occurs in >= min_frames frames or once with a
    score >= high_score, and is neither blacklisted nor (with keep_stuff false) a stuff class.
    User terms always enter (also when blacklisted); their stats come from the model if it saw
    them, otherwise frames 0 and max_score null.
    """
    frames: dict[str, int] = {}
    best: dict[str, float] = {}
    for tags in per_frame.values():
        for t, s in tags:
            frames[t] = frames.get(t, 0) + 1
            best[t] = max(best.get(t, 0.0), s)

    user_set = set(user)
    excluded = {
        t
        for t in frames
        if (t in blacklist or (t in stuff and not keep_stuff)) and t not in user_set
    }
    entries: dict[str, dict] = {}
    for t, n in frames.items():
        if t in excluded or t in user_set:
            continue
        if n >= min_frames or best[t] >= high_score:
            entries[t] = _entry(t, n, best[t], "model", stuff)
    for t in user:
        if t not in entries:
            entries[t] = _entry(t, frames.get(t, 0), best.get(t), "user", stuff)
    vocab = sorted(entries.values(), key=lambda v: (-v["frames"], -(v["max_score"] or 0), v["tag"]))
    return vocab, sorted(excluded)


def _entry(tag: str, n: int, score: float | None, source: str, stuff: set[str]) -> dict:
    return {
        "tag": tag,
        "frames": n,
        "max_score": None if score is None else round(score, 4),
        "source": source,
        "kind": "stuff" if tag in stuff else "object",
    }
