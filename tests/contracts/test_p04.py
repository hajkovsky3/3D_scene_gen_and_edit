"""Contract tests for phase 04_tagging (format of outputs, not quality).

The aggregation logic is tested with a fake tagger that returns fixed tags per frame; the
RAM++ test runs only where the `ram` package, the checkpoint and torch are available.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scenepipe.core.phase import ContractError, register
from scenepipe.core.progress import StageProgress
from scenepipe.core.workspace import Workspace
from scenepipe.phases.p04_tagging.common import (
    REPO_ROOT,
    TaggingBase,
    normalize,
    read_synonyms,
    sample_frames,
)
from scenepipe.phases.p04_tagging.fixed import TaggingFixed

N_FRAMES = 12


@register
class TaggingFake(TaggingBase):
    """Every frame: couch (synonym of sofa), floor (stuff), room (blacklisted), a low-score
    tag; frame 0 also a single high-score tag and frame 1 a single low-score one."""

    impl = "fake"

    def tag_frames(self, frames: list[tuple[str, Path]], stage: StageProgress):
        out = {}
        for i, (fid, path) in enumerate(frames):
            assert path.is_file()
            tags = [("Couch", 0.9), ("sofa", 0.7), ("floor", 0.8), ("room", 0.95), ("cat", 0.3)]
            if i == 0:
                tags.append(("piano", 0.95))
            if i == 1:
                tags.append(("vase", 0.7))
            out[fid] = tags
            stage.update((i + 1) / len(frames))
        return out


def make_ws(tmp_path: Path, user_vocab: str | None = None) -> Workspace:
    ws = Workspace.open(tmp_path, "demo")
    images = ws.root / "00_ingest" / "images"
    images.mkdir(parents=True)
    frames = []
    rng = np.random.default_rng(0)
    for i in range(N_FRAMES):
        fid = f"f{i + 1:06d}"
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), np.uint8)).save(images / f"{fid}.jpg")
        frames.append({"frame_id": fid, "image": f"00_ingest/images/{fid}.jpg"})
    doc = {"contract_version": 1, "frames": frames, "rejected": []}
    (ws.root / "00_ingest" / "frames.json").write_text(json.dumps(doc), encoding="utf-8")
    if user_vocab is not None:
        (ws.root / "input").mkdir()
        (ws.root / "input" / "vocabulary.txt").write_text(user_vocab, encoding="utf-8")
    return ws


@pytest.fixture
def lists(tmp_path) -> dict[str, str]:
    d = tmp_path / "lists"
    d.mkdir()
    (d / "blacklist.txt").write_text("# scenes\nroom\n", encoding="utf-8")
    (d / "stuff.txt").write_text("floor\nwall\n", encoding="utf-8")
    (d / "synonyms.yaml").write_text("sofa: [couch]\n", encoding="utf-8")
    return {
        "blacklist_file": str(d / "blacklist.txt"),
        "stuff_file": str(d / "stuff.txt"),
        "synonyms_file": str(d / "synonyms.yaml"),
    }


def run(phase_cls, ws: Workspace, **config) -> dict:
    phase_cls(ws, config).execute()
    return json.loads((ws.root / "04_tagging" / "tags.json").read_text(encoding="utf-8"))


def by_tag(doc: dict) -> dict[str, dict]:
    return {v["tag"]: v for v in doc["vocabulary"]}


def test_contract(tmp_path, lists):
    ws = make_ws(tmp_path)
    doc = run(TaggingFake, ws, sample_frames=6, **lists)
    assert doc["contract_version"] == 1
    assert len(doc["per_frame"]) == 6
    frames = json.loads((ws.root / "00_ingest" / "frames.json").read_text("utf-8"))["frames"]
    assert set(doc["per_frame"]) <= {f["frame_id"] for f in frames}
    for tags in doc["per_frame"].values():
        names = [t["tag"] for t in tags]
        assert names.count("sofa") == 1 and "couch" not in names  # merged synonyms
        assert "cat" not in names  # below score_threshold
        assert next(t for t in tags if t["tag"] == "sofa")["score"] == 0.9  # max of merged

    vocab = by_tag(doc)
    assert set(vocab) == {"sofa", "floor", "piano"}  # vase: 1 frame, low score
    assert vocab["sofa"] == {
        "tag": "sofa",
        "frames": 6,
        "max_score": 0.9,
        "source": "model",
        "kind": "object",
    }
    assert vocab["floor"]["kind"] == "stuff"
    assert vocab["piano"]["frames"] == 1  # single frame, score >= high_score
    assert doc["excluded"] == ["room"]
    assert doc["vocabulary"][0]["tag"] == "sofa"  # most frames first

    manifest = json.loads((ws.root / "04_tagging" / "manifest.json").read_text("utf-8"))
    assert manifest["phase"] == "04_tagging" and manifest["impl"] == "fake"
    assert manifest["inputs"]["frames"]["path"] == "00_ingest/frames.json"


def test_stuff_excluded_and_min_frames(tmp_path, lists):
    ws = make_ws(tmp_path)
    doc = run(TaggingFake, ws, keep_stuff_classes=False, min_frames=1, high_score=1.0, **lists)
    vocab = by_tag(doc)
    assert "floor" not in vocab and "floor" in doc["excluded"]
    assert {"vase", "piano"} <= set(vocab)  # min_frames 1
    assert len(doc["per_frame"]) == N_FRAMES  # sample_frames 30 > 12 frames


def test_user_vocabulary(tmp_path, lists):
    ws = make_ws(tmp_path, "# extra terms\nLamp\nroom\nCouch\n\n")
    doc = run(TaggingFake, ws, **lists)
    vocab = by_tag(doc)
    assert vocab["lamp"] == {
        "tag": "lamp",
        "frames": 0,
        "max_score": None,
        "source": "user",
        "kind": "object",
    }
    assert vocab["room"]["source"] == "user" and "room" not in doc["excluded"]  # user wins
    assert vocab["sofa"]["source"] == "user" and vocab["sofa"]["frames"] == N_FRAMES
    manifest = json.loads((ws.root / "04_tagging" / "manifest.json").read_text("utf-8"))
    assert manifest["inputs"]["user_vocabulary"]["path"] == "input/vocabulary.txt"


def test_fixed(tmp_path, lists):
    ws = make_ws(tmp_path, "chair\ncouch\nwall\n")
    doc = run(TaggingFixed, ws, **lists)
    assert doc["per_frame"] == {}
    assert [v["tag"] for v in doc["vocabulary"]] == ["chair", "sofa", "wall"]
    assert by_tag(doc)["wall"]["kind"] == "stuff"
    assert all(v["source"] == "user" for v in doc["vocabulary"])


def test_fixed_needs_vocabulary(tmp_path, lists):
    with pytest.raises(FileNotFoundError):
        TaggingFixed(make_ws(tmp_path), lists).execute()


def test_empty_vocabulary_fails(tmp_path, lists):
    ws = make_ws(tmp_path)
    with pytest.raises(ContractError):
        TaggingFake(ws, {**lists, "score_threshold": 0.99, "high_score": 1.0}).execute()
    assert not (ws.root / "04_tagging" / "manifest.json").exists()


def test_helpers(tmp_path):
    frames = [{"frame_id": i} for i in range(100)]
    picked = [f["frame_id"] for f in sample_frames(frames, 5)]
    assert picked == [0, 25, 50, 74, 99]
    assert len(sample_frames(frames, 0)) == 100
    assert normalize("  Coffee   Table. ") == "coffee table"
    bad = tmp_path / "syn.yaml"
    bad.write_text("sofa: [couch]\nbench: [couch]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_synonyms(bad)


def test_repo_lists_consistent():
    """No blacklisted term is also a stuff class, and no alias is itself a blacklisted term."""
    from scenepipe.phases.p04_tagging.common import canonical, read_terms

    syn = read_synonyms(REPO_ROOT / "configs" / "tag_synonyms.yaml")
    black = {canonical(t, syn) for t in read_terms(REPO_ROOT / "configs" / "tag_blacklist.txt")}
    stuff = {canonical(t, syn) for t in read_terms(REPO_ROOT / "configs" / "tag_stuff.txt")}
    assert black and stuff and not black & stuff
    assert not set(syn.values()) & black


RAM_CHECKPOINT = REPO_ROOT / "checkpoints" / "ram_plus" / "ram_plus_swin_large_14m.pth"


@pytest.mark.skipif(
    importlib.util.find_spec("ram") is None or not RAM_CHECKPOINT.is_file(),
    reason="needs the `ram` package and the RAM++ checkpoint (phases/04_tagging.md)",
)
def test_ram_plus(tmp_path):
    from scenepipe.phases.p04_tagging.ram_plus import TaggingRamPlus

    ws = make_ws(tmp_path, "chair\n")  # noise images: the vocabulary may be only the user term
    doc = run(TaggingRamPlus, ws, sample_frames=3, batch_size=2)
    assert len(doc["per_frame"]) == 3
    assert "chair" in by_tag(doc)
