"""Phase 04_tagging, implementation `fixed`: the vocabulary is exactly input/vocabulary.txt
(after synonyms), no model. For controlled experiments. Spec: phases/04_tagging.md"""

from __future__ import annotations

from pathlib import Path

from scenepipe.core.phase import register
from scenepipe.core.progress import StageProgress
from scenepipe.phases.p04_tagging.common import TaggingBase


@register
class TaggingFixed(TaggingBase):
    impl = "fixed"
    impl_version = "1"
    uses_model = False

    def tag_frames(
        self, frames: list[tuple[str, Path]], stage: StageProgress
    ) -> dict[str, list[tuple[str, float]]]:
        return {}
