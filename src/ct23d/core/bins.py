from __future__ import annotations

from typing import List

import numpy as np

from .models import IntensityBin, MeshingConfig, ProjectConfig


# ---------------------------------------------------------------------------
# Bin generation
# ---------------------------------------------------------------------------

def generate_uniform_bins(cfg: MeshingConfig) -> List[IntensityBin]:
    """
    Generate uniform intensity bins from MeshingConfig.

    Bins cover the range [min_intensity, max_intensity] with `n_bins` bins.
    Each bin is [low, high) in integer intensity units.

    Example:
        min_intensity = 1, max_intensity = 255, n_bins = 3
        -> [1, 86), [86, 171), [171, 256)
    """
    n = cfg.n_bins
    if n <= 0:
        raise ValueError(f"MeshingConfig.n_bins must be > 0, got {n}.")

    min_i = int(cfg.min_intensity)
    max_i = int(cfg.max_intensity)

    if max_i <= min_i:
        raise ValueError(
            f"MeshingConfig.max_intensity ({max_i}) must be > "
            f"min_intensity ({min_i})."
        )

    # We want n bins that partition [min_i, max_i] as evenly as possible.
    # Use linspace to get boundaries in float and round.
    edges = np.linspace(min_i, max_i, num=n + 1, endpoint=True)
    edges = np.round(edges).astype(int)

    bins: List[IntensityBin] = []
    for idx in range(n):
        low = int(edges[idx])
        high = int(edges[idx + 1])
        # Ensure at least width 1
        if high <= low:
            high = low + 1
        # Ensure minimum intensity is at least 1 (0 is typically background/air)
        if low < 1:
            low = 1
            if high <= low:
                high = low + 1
        bins.append(
            IntensityBin(
                index=idx,
                low=low,
                high=high,
                name=f"bin_{idx:02d}",
                color=None,
                enabled=True,
            )
        )

    return bins


def select_bins(project: ProjectConfig) -> List[IntensityBin]:
    """
    Select the list of bins to use for meshing, given a ProjectConfig.

    Behavior:
      - If project.meshing.use_custom_bins is True:
          -> use project.bins but filter for enabled=True.
      - Else:
          -> generate uniform bins from project.meshing and mark them enabled.

    Returns a NEW list of bins; mutating them will not affect project.bins.
    """
    meshing_cfg = project.meshing

    if meshing_cfg.use_custom_bins and project.bins:
        # Use custom bins but filter and re-index the enabled ones
        enabled_bins = [b for b in project.bins if b.enabled]
        bins: List[IntensityBin] = []
        for new_idx, b in enumerate(enabled_bins):
            # Copy with updated index to ensure a clean 0..N-1 indexing
            bins.append(
                IntensityBin(
                    index=new_idx,
                    low=b.low,
                    high=b.high,
                    name=b.name,
                    color=b.color,
                    enabled=True,
                )
            )
        return bins

    # Default: uniform bins from MeshingConfig
    return generate_uniform_bins(meshing_cfg)


# ---------------------------------------------------------------------------
# Bin utilities
# ---------------------------------------------------------------------------

def describe_bin(bin: IntensityBin) -> str:
    """
    Human-readable description of an intensity bin, for CLI / logs / GUI.

    Example:
        "bin_00: [1, 32)"
        "bone: [200, 256)"
    """
    label = bin.name if bin.name is not None else f"bin_{bin.index:02d}"
    return f"{label}: [{bin.low}, {bin.high})"


def build_bin_mask(
    volume: np.ndarray,
    bin: IntensityBin,
) -> np.ndarray:
    """
    Build a boolean mask for voxels falling into the bin's intensity range.

    Parameters
    ----------
    volume:
        3D grayscale volume [Z, Y, X], integer-valued (e.g. uint8).
    bin:
        IntensityBin specifying [low, high) interval.

    Returns
    -------
    np.ndarray
        Boolean mask [Z, Y, X] where True means voxel belongs to this bin.

    Raises
    ------
    ValueError
        If volume is not 3D.
    """
    if volume.ndim != 3:
        raise ValueError(
            f"build_bin_mask expects a 3D grayscale volume [Z, Y, X], "
            f"got shape {volume.shape}"
        )

    low = bin.low
    high = bin.high
    # bin is [low, high): inclusive low, exclusive high
    mask = (volume >= low) & (volume < high)
    return mask
