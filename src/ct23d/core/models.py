from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Intensity = int                      # grayscale value, 0–255 for now
Spacing3D = Tuple[float, float, float]  # (z, y, x) in mm
ColorRGB = Tuple[float, float, float]   # normalized 0–1


# ---------------------------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """
    Configuration for preprocessing raw CT slice images.

    This maps to the behavior implemented in:
      - load_and_clean_slices()
      - remove_colored_overlays()
      - remove_bed_and_headrest_sequence()
    from the legacy CT_to_3D.py script. :contentReference[oaicite:0]{index=0}

    All paths are *project-relative* or *config-file-relative*; they are
    interpreted by higher-level helpers in core.io_paths / config_io.
    """

    # --- IO paths ---

    input_dir: Optional[Path] = None
    """
    Directory containing raw CT slice images (JPEG/PNG).
    Expected to be relative to the project root or to the config file.
    """

    processed_dir: Optional[Path] = None
    """
    Directory to store processed slices (after overlay/bed removal).
    If None, a default like input_dir / 'processed_slices' can be used
    by higher-level code.
    """

    use_cache: bool = True
    """
    If True and processed_dir exists with images, reuse them instead of
    regenerating processing.
    Mirrors the 'use_cache' logic in load_and_clean_slices().
    """

    # --- Overlay / text removal ---

    grayscale_tolerance: int = 1
    """
    Max allowed difference between RGB channels to still consider a pixel
    grayscale. Passed to remove_colored_overlays().
    """

    saturation_threshold: float = 0.08
    """
    HSV saturation threshold (0–1) above which pixels are treated as colored
    overlays (numbers, text, markers). Also passed to remove_colored_overlays().
    """

    # --- Bed / headrest handling ---

    remove_bed: bool = True
    """
    If True, apply bed/headrest removal across the stack via
    remove_bed_and_headrest_sequence().
    """

    # --- Additional preprocessing options ---

    remove_non_grayscale: bool = False
    """
    If True, turn all non-grayscale pixels black (set to 0,0,0) based on saturation threshold.
    This is more aggressive than overlay removal - it completely removes color.
    Uses the saturation_threshold parameter to determine what counts as non-grayscale.
    """
    
    remove_overlays: bool = True
    """
    If True, automatically remove colored overlays (text, markers) by converting them to grayscale.
    This is the default behavior that converts colored pixels to grayscale based on saturation threshold.
    If False, preserves original colors (useful for JPEG images that should remain colored).
    """

    object_mask: Optional[np.ndarray] = None
    """
    Optional 2D boolean mask (Y, X) for the selected slice to remove objects.
    This mask will be tracked across the Z-stack using object matching.
    If None, no object removal is applied.
    """
    
    object_mask_slice_index: Optional[int] = None
    """
    Index of the slice (0-based) where object_mask was selected.
    If None and object_mask is provided, defaults to 0 (first slice).
    The mask will be propagated forward and backward from this slice.
    """
    
    rotation: int = 0
    """
    Rotation angle in degrees (0, 90, 180, 270) to apply during preprocessing.
    Positive = clockwise. Rotation is applied when loading images for processing.
    """
    
    crop_objects: Optional[List[dict]] = None
    """
    Optional list of crop objects, each containing:
    - 'mask': 2D boolean mask (Y, X) - pixels inside mask are kept, outside are set to black
    - 'slice_min': int - first slice where crop applies (0-based)
    - 'slice_max': int - last slice where crop applies (0-based, inclusive)
    If None or empty, no cropping is applied.
    """
    
    object_removal_objects: Optional[List[dict]] = None
    """
    Optional list of object removal objects, each containing:
    - 'mask': 2D boolean mask (Y, X) - pixels to remove
    - 'slice_min': int - first slice where removal applies (0-based)
    - 'slice_max': int - last slice where removal applies (0-based, inclusive)
    If None or empty, no object removal is applied.
    """
    
    non_grayscale_slice_ranges: Optional[List[Tuple[int, int]]] = None
    """
    Optional list of (min_slice, max_slice) tuples for non-grayscale removal (0-based, inclusive).
    If None or empty, no non-grayscale removal is applied.
    """
    
    non_body_removal_objects: Optional[List[dict]] = None
    """
    Optional list of non-body removal objects, each containing:
    - 'parameters': dict with 'body_threshold_hu', 'closing_radius_mm', 'min_component_size_vox',
      'outside_only', 'background_fill'
    - 'slice_min': int - first slice where removal applies (0-based)
    - 'slice_max': int - last slice where removal applies (0-based, inclusive)
    If None or empty, no non-body removal is applied.
    """
    
    export_slice_range: Optional[Tuple[int, int]] = None
    """
    Optional tuple (min_slice, max_slice) for export slice range (0-based, inclusive).
    If None, all processed slices are exported.
    """
    
    export_prefix: Optional[str] = None
    """
    Optional prefix for exported filenames. If provided, exported files will be named
    as "{export_prefix}_{index:05d}.{ext}" where index follows the Z order (0, 1, 2, ...).
    If None, original filenames are preserved.
    """
    
    reordered_slice_paths: Optional[List[Path]] = None
    """
    Optional list of slice paths in Z order (sorted by actual Z position from DICOM metadata).
    If provided, preprocessing will use this order instead of alphabetical filename sorting.
    If None, uses alphabetical sorting of filenames.
    """


# ---------------------------------------------------------------------------
# Intensity bin definition
# ---------------------------------------------------------------------------

@dataclass
class IntensityBin:
    """
    One grayscale-intensity bin used to generate a mesh.

    Normally bins are auto-generated from MeshingConfig (min/max/n_bins),
    but GUI presets or YAML configs can define explicit bins.

    Intensity range is [low, high) in grayscale units.
    """

    index: int
    """Zero-based bin index, used for naming and ordering."""

    low: Intensity
    """Inclusive lower bound for grayscale intensity."""

    high: Intensity
    """Exclusive upper bound for grayscale intensity."""

    name: Optional[str] = None
    """
    Optional display name (e.g. 'bone', 'soft tissue').
    Not required by core logic; useful for GUI and presets.
    """

    color: Optional[ColorRGB] = None
    """
    Optional normalized RGB color (0–1) for GUI previews or suggested export.
    The meshing core can still sample real CT colors as in the legacy script.
    """

    enabled: bool = True
    """
    If False, this bin is kept in the config but skipped when generating meshes.
    Handy for toggling bins from the GUI.
    """


# ---------------------------------------------------------------------------
# Meshing configuration
# ---------------------------------------------------------------------------

@dataclass
class MeshingConfig:
    """
    Configuration for volume -> mesh conversion.

    This corresponds to CLI options and parameters used in the legacy script: :contentReference[oaicite:1]{index=1}
      - spacing
      - non_black_threshold
      - min_component_size
      - smoothing_sigma
      - n_bins, min_intensity, max_intensity
      - output directory / prefix
    """

    # --- Geometry / spacing ---

    spacing: Spacing3D = (1.6, 1.0, 1.0)
    """
    Physical voxel spacing as (z, y, x) in millimeters.
    Mirrors the '--spacing Z,Y,X' CLI argument.
    """

    # --- Global mask building (if used) ---

    non_black_threshold: int = 15
    """
    Intensity threshold below which voxels are considered background when
    building a global mask (build_mask()).
    Mirrors '--non-black-threshold'.
    """

    min_component_size: int = 5000
    """
    Minimum connected component size (in voxels) to keep during mask cleaning
    or adaptive filtering. Mirrors '--min-component-size' (but with a more
    conservative default suited to full-body scans).
    """

    enable_component_filtering: bool = False
    """
    If True, remove small connected components smaller than min_component_size.
    If False, skip component filtering (preserves all details, may include noise).
    Default: False (off) for minimal loss.
    """

    smoothing_sigma: float = 1.0
    """
    Sigma for Gaussian smoothing (in voxel units).
    Used in build_mask() and optionally in per-bin masks.
    Mirrors '--smoothing-sigma'.
    """

    enable_smoothing: bool = False
    """
    If True, apply Gaussian smoothing to masks before meshing.
    If False, skip smoothing (preserves sharp edges, may create jagged surfaces).
    Default: False (off) for minimal loss.
    
    Note: In combined export mode, smoothing each bin separately can cause boundary
    misalignment. For best quality in combined mode, consider disabling smoothing or
    using lower sigma values.
    """
    
    smooth_combined_masks: bool = False
    """
    For combined export mode: If True, combine all bin masks before smoothing
    (then separate for color assignment). This preserves bin boundary alignment
    but may blur colors at boundaries. If False, smooth each bin separately
    (may cause boundary gaps/overlaps).
    
    Note: This option only applies when exporting as combined file.
    For separate file export, each bin is always smoothed individually.
    """

    # --- Binning ---

    n_bins: int = 6
    """
    Number of intensity bins when auto-generating uniform bins between
    [min_intensity, max_intensity]. Mirrors '--n-bins'.
    """

    min_intensity: Intensity = 1
    """Minimum grayscale value included in binning. Mirrors '--min-intensity'."""

    max_intensity: Intensity = 255
    """Maximum grayscale value included in binning. Mirrors '--max-intensity'."""

    use_custom_bins: bool = False
    """
    If True, ignore n_bins / min_intensity / max_intensity and use the explicit
    list of IntensityBin objects on ProjectConfig.bins instead.
    """

    # --- Output ---

    output_dir: Optional[Path] = None
    """
    Directory where generated meshes are saved.
    Expected to be project-relative; resolved by callers.
    If None, callers decide a sensible default (e.g. next to input_dir).
    """

    output_prefix: str = "ct_layer"
    """
    Prefix for output PLY filenames, e.g.
      'ct_layer_bin_00_1_32.ply'
    Mirrors '--output-prefix'.
    """

    # --- Behavior flags ---

    adaptive_component_filtering: bool = True
    """
    If True, per-bin masks will use an adaptive min-component-size strategy
    similar to generate_intensity_meshes() in the legacy script
    (based on a fraction of the largest component). :contentReference[oaicite:2]{index=2}
    """

    skip_empty_bins: bool = True
    """
    If True, bins with no voxels simply log/notify and are skipped instead of
    raising errors.
    """


# ---------------------------------------------------------------------------
# Project-level configuration
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """
    Top-level configuration object for a CT23D project.

    This groups together:
      - preprocessing options (PreprocessConfig),
      - meshing options (MeshingConfig),
      - optional explicit intensity bins (List[IntensityBin]).

    This is what we will typically:
      - load/save to YAML,
      - pass around in the GUI,
      - adapt into CLI arguments for the legacy compatibility layer.
    """

    name: str = "CT23D Project"
    """Human-readable project name (for GUI titles, recent-project list, etc.)."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    """Preprocessing options (paths, overlay removal, bed removal, etc.)."""

    meshing: MeshingConfig = field(default_factory=MeshingConfig)
    """Meshing options (spacing, thresholds, binning, outputs)."""

    bins: List[IntensityBin] = field(default_factory=list)
    """
    Optional explicit intensity bins.
    If empty and meshing.use_custom_bins is False, core.bins will generate
    uniform bins from MeshingConfig.
    """

    config_path: Optional[Path] = None
    """
    Optional path to the YAML config file this was loaded from.
    Purely informational; not used by algorithms.
    Stored as project-relative or config-relative by config_io helpers.
    """
