"""Gaussian splat PLY I/O. Format: docs/data-contracts.md, section 5.

Numpy only, so every phase environment can read and write the scene geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
MAX_SH_DEGREE = 3
NUM_REST = (MAX_SH_DEGREE + 1) ** 2 - 1  # 15 coefficients per channel in the file

PLY_PROPERTIES: tuple[str, ...] = (
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    *(f"f_rest_{i}" for i in range(3 * NUM_REST)),
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)  # fmt: skip


def num_sh_rest(sh_degree: int) -> int:
    return (sh_degree + 1) ** 2 - 1


@dataclass
class Gaussians:
    """Gaussians in the stored (pre-activation) parametrization.

    means: float32[N,3] world coordinates
    sh_dc: float32[N,3] SH degree-0 coefficients (not RGB)
    sh_rest: float32[N,K,3] SH degrees 1..d, K = (d+1)^2 - 1 (coefficient-major, like gsplat)
    opacity: float32[N] logit
    scales: float32[N,3] log
    quats: float32[N,4] wxyz, not necessarily normalized
    """

    means: np.ndarray
    sh_dc: np.ndarray
    sh_rest: np.ndarray
    opacity: np.ndarray
    scales: np.ndarray
    quats: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.means)
        shapes = {
            "means": (n, 3),
            "sh_dc": (n, 3),
            "opacity": (n,),
            "scales": (n, 3),
            "quats": (n, 4),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name}: shape {getattr(self, name).shape} != {shape}")
        k = self.sh_rest.shape[1] if self.sh_rest.ndim == 3 else -1
        if self.sh_rest.shape != (n, k, 3) or k not in {num_sh_rest(d) for d in range(4)}:
            raise ValueError(f"sh_rest: invalid shape {self.sh_rest.shape}")

    def __len__(self) -> int:
        return len(self.means)

    @property
    def sh_degree(self) -> int:
        return round(np.sqrt(self.sh_rest.shape[1] + 1)) - 1

    def rgb(self) -> np.ndarray:
        """View-independent color in [0, 1] (degree-0 SH only)."""
        return np.clip(0.5 + SH_C0 * self.sh_dc, 0.0, 1.0)

    def with_sh_degree(self, sh_degree: int) -> Gaussians:
        """Copy with sh_rest truncated or zero-padded to the given degree."""
        k = num_sh_rest(sh_degree)
        rest = np.zeros((len(self), k, 3), np.float32)
        m = min(k, self.sh_rest.shape[1])
        rest[:, :m] = self.sh_rest[:, :m]
        return Gaussians(self.means, self.sh_dc, rest, self.opacity, self.scales, self.quats)


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """RGB in [0, 1] -> SH degree-0 coefficient."""
    return (np.asarray(rgb, np.float32) - 0.5) / SH_C0


def write_ply(path: Path, g: Gaussians) -> None:
    """Binary little-endian PLY with exactly PLY_PROPERTIES (missing SH degrees as zeros)."""
    n = len(g)
    rest = g.with_sh_degree(MAX_SH_DEGREE).sh_rest  # [N,15,3]
    cols = [
        g.means,
        np.zeros((n, 3), np.float32),
        g.sh_dc,
        rest.transpose(0, 2, 1).reshape(n, 3 * NUM_REST),  # channel-major: R0..R14, G0.., B0..
        g.opacity[:, None],
        g.scales,
        g.quats,
    ]
    data = np.concatenate([np.asarray(c, np.float32) for c in cols], axis=1)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            *(f"property float {p}" for p in PLY_PROPERTIES),
            "end_header",
        ]
    )
    with open(path, "wb") as f:
        f.write((header + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data, dtype="<f4").tobytes())


def read_ply(path: Path, sh_degree: int = MAX_SH_DEGREE) -> Gaussians:
    """Read a PLY written in the project format; raises ValueError on any other layout."""
    with open(path, "rb") as f:
        header: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break
        body = f.read()
    if header[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError(f"{path}: not a binary little-endian PLY")
    elements = [h for h in header if h.startswith("element ")]
    if len(elements) != 1 or not elements[0].startswith("element vertex "):
        raise ValueError(f"{path}: expected a single 'vertex' element, got {elements}")
    n = int(elements[0].split()[2])
    props = [h.split() for h in header if h.startswith("property ")]
    if [p[1] for p in props] != ["float"] * len(props):
        raise ValueError(f"{path}: all properties must be float32")
    names = tuple(p[2] for p in props)
    if names != PLY_PROPERTIES:
        raise ValueError(f"{path}: properties differ from docs/data-contracts.md section 5")
    if len(body) != n * len(names) * 4:
        raise ValueError(f"{path}: body has {len(body)} bytes, expected {n * len(names) * 4}")
    data = np.frombuffer(body, dtype="<f4").reshape(n, len(names)).astype(np.float32)
    i = 9 + 3 * NUM_REST
    rest = data[:, 9:i].reshape(n, 3, NUM_REST).transpose(0, 2, 1)
    g = Gaussians(
        means=data[:, 0:3].copy(),
        sh_dc=data[:, 6:9].copy(),
        sh_rest=np.ascontiguousarray(rest),
        opacity=data[:, i].copy(),
        scales=data[:, i + 1 : i + 4].copy(),
        quats=data[:, i + 4 : i + 8].copy(),
    )
    return g.with_sh_degree(sh_degree) if sh_degree != MAX_SH_DEGREE else g
