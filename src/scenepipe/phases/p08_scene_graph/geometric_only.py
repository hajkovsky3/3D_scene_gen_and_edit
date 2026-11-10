"""Phase 08_scene_graph, implementation `geometric_only` (baseline): no VLM or LLM, the
caption is the label of 07 and only geometric relations are computed. Runs on the CPU.
"""

from __future__ import annotations

import logging
from typing import Any

from scenepipe.core.phase import register
from scenepipe.phases.p08_scene_graph.common import SceneGraphBase

log = logging.getLogger(__name__)


@register
class SceneGraphGeometricOnly(SceneGraphBase):
    impl = "geometric_only"
    impl_version = "1"

    def describe(self, nodes: list[dict[str, Any]], ctx: dict[str, Any]) -> None:
        for n in nodes:
            n["caption"] = n["label"] or ""

    def semantic_relations(self, nodes: list[dict], edges: list[dict]) -> list[dict]:
        log.warning("geometric_only: semantic_relations ignored (no LLM)")
        return []
