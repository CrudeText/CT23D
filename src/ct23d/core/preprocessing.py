from __future__ import annotations

from pathlib import Path
from typing import List, Callable, Optional

import numpy as np

from .models import PreprocessConfig
from . import images


# Allowed image extensions
_IMAGE_EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]


def _ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def _has_any_images(folder: Path) -> bool:
    """
    Lightweight check to know if a folder already contains images.
    """
    if not folder.is_dir():
        return False

    for ext in _IMAGE_EXTS:
        if any(folder.glob(f"*{ext}")):
            return True
    return False


def _list_image_files(folder: Path) -> List[Path]:
    """
    List image files in a folder, sorted by name.

    Raises a RuntimeError with a clear message if no images are found.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    paths: List[Path] = []
    for ext in _IMAGE_EXTS:
        paths.extend(folder.glob(f"*{ext}"))

    paths = sorted(paths)
    if not paths:
        raise RuntimeError(
            f"No image files with extensions {_IMAGE_EXTS} found in: {folder}"
        )
    return paths


def preprocess_slices(
    cfg: PreprocessConfig,
    progress_cb: Optional[Callable[[str, int, int, int], None]] = None,
) -> Path:
    """
    Main preprocessing entry point.

    - READ raw slices from cfg.input_dir
    - WRITE processed slices to cfg.processed_dir (or <input>/processed_slices)
    - Optional caching: if use_cache=True and processed dir already has images,
      we simply return the processed directory.
    - Optional progress_cb(idx, total) is called during saving, where idx is
      in [1, total].

    Returns
    -------
    Path
        Directory containing processed slices.
    """
    input_dir: Path = cfg.input_dir
    if cfg.processed_dir is None:
        processed_dir = input_dir / "processed_slices"
    else:
        processed_dir = cfg.processed_dir

    input_dir = input_dir.resolve()
    processed_dir = processed_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Cache: if processed directory already has images and caching is enabled, just use it.
    if cfg.use_cache and _has_any_images(processed_dir):
        return processed_dir

    # ------------------------------------------------------------------
    # 1. Load raw slices FROM INPUT DIR
    # ------------------------------------------------------------------
    slice_paths: List[Path] = _list_image_files(input_dir)
    total = len(slice_paths)

    # Load all images as RGB numpy arrays with rotation (with progress)
    rgb_slices = []
    for idx, p in enumerate(slice_paths, start=1):
        if progress_cb is not None:
            # Report progress: phase, current_in_phase, total_in_phase, total_overall
            # If progress_cb raises an exception (e.g., InterruptedError), let it propagate
            progress_cb("loading", idx, total, total)
        rgb_slices.append(images.load_image_rgb(p, rotation=cfg.rotation))
    volume_rgb = np.stack(rgb_slices, axis=0)  # (Z, Y, X, 3)
    
    if progress_cb is not None:
        progress_cb("loading", total, total, total)  # Mark loading as complete

    # ------------------------------------------------------------------
    # 2. Run the actual preprocessing pipeline
    # ------------------------------------------------------------------
    # This phase includes:
    # - Mask propagation (if object_mask provided): tracks selected objects across all slices
    #   by finding overlapping objects in adjacent slices (forward and backward)
    # - Overlay removal: removes colored text/markers from each slice
    # - Non-grayscale removal: turns colored pixels black (if enabled)
    # - Object mask application: removes tracked objects from each slice
    if progress_cb is not None:
        progress_cb("processing", 0, total, total)  # Start processing
    
    # If progress_cb raises an exception during preprocessing, let it propagate
    processed_volume = images.preprocess_volume_rgb(
        volume_rgb,
        grayscale_tolerance=cfg.grayscale_tolerance,
        saturation_threshold=cfg.saturation_threshold,
        remove_bed=cfg.remove_bed,
        remove_non_grayscale=cfg.remove_non_grayscale,
        object_mask=cfg.object_mask,
        object_mask_slice_index=cfg.object_mask_slice_index,
        progress_cb=progress_cb,
    )

    if progress_cb is not None:
        progress_cb("processing", total, total, total)  # Mark processing as complete

    # ------------------------------------------------------------------
    # 3. Save processed slices to processed_dir
    # ------------------------------------------------------------------
    _ensure_dir(processed_dir)

    for idx, (src_path, slice_arr) in enumerate(zip(slice_paths, processed_volume), start=1):
        if progress_cb is not None:
            progress_cb("saving", idx, total, total)

        out_path = processed_dir / src_path.name
        images.save_image_rgb(out_path, slice_arr)
    
    if progress_cb is not None:
        progress_cb("saving", total, total, total)  # Mark saving as complete

    return processed_dir
