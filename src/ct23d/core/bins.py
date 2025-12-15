from __future__ import annotations

from typing import List, Tuple

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


def detect_optimal_bins(
    intensities: np.ndarray,
    min_bins: int = 3,
    max_bins: int = 12,
    min_pixels_per_bin: int = 1000,
) -> Tuple[int, float, float]:
    """
    Detect optimal number of bins and intensity range from histogram distribution.
    
    Uses histogram analysis to find:
    1. Optimal number of bins based on data distribution
    2. Effective min/max intensity range (excluding outliers)
    
    Parameters
    ----------
    intensities : np.ndarray
        Array of intensity values (can be 1D or flattened)
    min_bins : int
        Minimum number of bins to generate
    max_bins : int
        Maximum number of bins to generate
    min_pixels_per_bin : int
        Minimum number of pixels per bin (used to limit bin count)
        
    Returns
    -------
    Tuple[int, float, float]
        (optimal_n_bins, effective_min, effective_max)
    """
    intensities_flat = intensities.flatten() if intensities.ndim > 1 else intensities
    
    # Remove zeros (background/air) for better analysis
    non_zero = intensities_flat[intensities_flat > 0]
    if non_zero.size == 0:
        # All zeros - use full range
        vmin = float(intensities_flat.min())
        vmax = float(intensities_flat.max())
        return min_bins, vmin, vmax
    
    # Use percentiles to exclude outliers (1st and 99th percentiles)
    p1 = np.percentile(non_zero, 1)
    p99 = np.percentile(non_zero, 99)
    
    # Effective range (excluding extreme outliers)
    effective_min = max(1.0, float(p1))  # At least 1
    effective_max = float(p99)
    
    # Calculate histogram to find optimal bin count
    # Use a fine-grained histogram first
    n_hist_bins = min(256, int(effective_max - effective_min + 1))
    if n_hist_bins < 10:
        n_hist_bins = 10
    
    hist, edges = np.histogram(non_zero, bins=n_hist_bins, range=(effective_min, effective_max))
    
    # Find peaks in histogram (local maxima)
    # Simple peak detection: values higher than neighbors
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist) * 0.1:
            peaks.append(i)
    
    # Optimal bin count based on:
    # 1. Number of significant peaks (tissue types)
    # 2. Data spread (std relative to mean)
    # 3. Minimum pixels per bin constraint
    
    n_peaks = len(peaks)
    if n_peaks > 0:
        # Use number of peaks as base, but limit by constraints
        suggested_bins = min(max(n_peaks, min_bins), max_bins)
    else:
        # No clear peaks - use data spread
        std_rel = np.std(non_zero) / (np.mean(non_zero) + 1e-6)
        if std_rel > 0.5:
            suggested_bins = max_bins
        elif std_rel > 0.3:
            suggested_bins = (min_bins + max_bins) // 2
        else:
            suggested_bins = min_bins
    
    # Apply minimum pixels per bin constraint
    total_pixels = non_zero.size
    max_bins_by_pixels = total_pixels // min_pixels_per_bin
    optimal_n_bins = min(suggested_bins, max_bins_by_pixels, max_bins)
    optimal_n_bins = max(optimal_n_bins, min_bins)
    
    return int(optimal_n_bins), effective_min, effective_max


def generate_optimal_bins(
    intensities: np.ndarray,
    min_bins: int = 3,
    max_bins: int = 12,
) -> List[IntensityBin]:
    """
    Generate optimal bins based on intensity distribution analysis.
    
    Parameters
    ----------
    intensities : np.ndarray
        Array of intensity values
    min_bins : int
        Minimum number of bins
    max_bins : int
        Maximum number of bins
        
    Returns
    -------
    List[IntensityBin]
        List of optimally-spaced bins
    """
    n_bins, vmin, vmax = detect_optimal_bins(intensities, min_bins, max_bins)
    
    # Generate uniform bins in the effective range
    vmin_int = int(round(vmin))
    vmax_int = int(round(vmax))
    
    if vmin_int < 1:
        vmin_int = 1
    if vmax_int <= vmin_int:
        vmax_int = vmin_int + 1
    
    edges = np.linspace(vmin_int, vmax_int, num=n_bins + 1, endpoint=True)
    edges = np.round(edges).astype(int)
    
    bins: List[IntensityBin] = []
    for idx in range(n_bins):
        low = int(edges[idx])
        high = int(edges[idx + 1])
        if high <= low:
            high = low + 1
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
