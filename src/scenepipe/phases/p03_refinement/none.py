"""Phase 03_refinement, implementation `none`: gaussians from 02 unchanged (baseline for
before/after comparisons). Spec: phases/03_refinement.md"""

from __future__ import annotations

from typing import Any

from scenepipe.core.cameras import Camera
from scenepipe.core.gaussians import Gaussians
from scenepipe.core.phase import register
from scenepipe.phases.p03_refinement.common import CRITERIA, RefinementBase


@register
class RefinementNone(RefinementBase):
    impl = "none"
    impl_version = "1"

    def refine(
        self, g: Gaussians, train: list[Camera], radius: float
    ) -> tuple[Gaussians | None, dict[str, int], int, dict[str, Any]]:
        return None, dict.fromkeys(CRITERIA, 0), 0, {}
