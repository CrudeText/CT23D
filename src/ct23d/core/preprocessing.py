from __future__ import annotations

from pathlib import Path
from typing import List, Callable, Optional

import numpy as np

from .models import PreprocessConfig
from . import images


# Allowed image extensions
_IMAGE_EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".dcm", ".dicom"]


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
    # Use reordered paths if provided, otherwise use alphabetical sorting
    if cfg.reordered_slice_paths is not None:
        slice_paths: List[Path] = cfg.reordered_slice_paths
    else:
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
    
    # Apply crop masks if specified (each crop applies to its slice range)
    if cfg.crop_objects:
        for crop_obj in cfg.crop_objects:
            crop_mask = crop_obj.get('mask')
            slice_min = crop_obj.get('slice_min', 0)
            slice_max = crop_obj.get('slice_max', len(volume_rgb) - 1)
            
            if crop_mask is not None:
                # Clamp slice range to valid bounds
                slice_min = max(0, min(slice_min, len(volume_rgb) - 1))
                slice_max = max(slice_min, min(slice_max, len(volume_rgb) - 1))
                
                # Apply crop mask to slices in range
                # crop_mask=True means keep those pixels, False means remove (set to black)
                keep_mask = crop_mask
                for z_idx in range(slice_min, slice_max + 1):
                    if z_idx < len(volume_rgb):
                        # Set pixels outside the crop mask to black
                        volume_rgb[z_idx][~keep_mask] = [0, 0, 0]
    
    # Apply non-body removal if specified (3D body mask computation)
    if cfg.non_body_removal_objects:
        for non_body_obj in cfg.non_body_removal_objects:
            slice_min = non_body_obj.get('slice_min', 0)
            slice_max = non_body_obj.get('slice_max', len(volume_rgb) - 1)
            params = non_body_obj.get('parameters', {})
            
            # Clamp slice range to valid bounds
            slice_min = max(0, min(slice_min, len(volume_rgb) - 1))
            slice_max = max(slice_min, min(slice_max, len(volume_rgb) - 1))
            
            # Convert RGB volume to grayscale for body mask computation
            if volume_rgb.ndim == 4 and volume_rgb.shape[3] == 3:
                # Use max channel (better for medical imaging)
                volume_gray = np.max(volume_rgb, axis=3)
            else:
                volume_gray = volume_rgb
            
            # Try to get spacing from DICOM files, otherwise use default
            spacing = (1.0, 1.0, 1.0)  # Default: (sz, sy, sx) in mm
            if slice_paths:
                try:
                    first_path = slice_paths[0]
                    if images._is_dicom_file(first_path):
                        # Try to get spacing from DICOM
                        try:
                            import pydicom
                            ds = pydicom.dcmread(str(first_path), stop_before_pixels=True)
                            if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing is not None:
                                pixel_spacing = ds.PixelSpacing
                                if len(pixel_spacing) >= 2:
                                    spacing_y = float(pixel_spacing[0])
                                    spacing_x = float(pixel_spacing[1])
                                    # Try to get slice thickness
                                    spacing_z = 1.0
                                    if hasattr(ds, 'SliceThickness') and ds.SliceThickness is not None:
                                        spacing_z = float(ds.SliceThickness)
                                    spacing = (spacing_z, spacing_y, spacing_x)  # (sz, sy, sx)
                        except Exception:
                            pass  # Use default spacing
                except Exception:
                    pass  # Use default spacing
            
            # Get parameters
            body_threshold_hu = params.get('body_threshold_hu', -300.0)
            closing_radius_mm = params.get('closing_radius_mm', 8.0)
            min_component_size_vox = params.get('min_component_size_vox', 1000)
            outside_only = params.get('outside_only', True)
            background_fill_hu = params.get('background_fill', -1024.0)
            
            # Try to convert HU values to raw intensity if DICOM
            # Get HU conversion parameters if available
            hu_conv = None
            if slice_paths:
                try:
                    first_path = slice_paths[0]
                    if images._is_dicom_file(first_path):
                        hu_conv = images.get_dicom_hu_conversion(first_path)
                except Exception:
                    pass
            
            # Convert body_threshold from HU to raw intensity if needed
            if hu_conv is not None:
                # DICOM with HU conversion available - use HU threshold directly
                body_threshold = body_threshold_hu
            else:
                # No HU conversion - convert HU threshold to approximate raw intensity
                if body_threshold_hu < 0:
                    body_threshold = 200.0  # Approximate for -300 HU in raw uint16
                else:
                    body_threshold = float(body_threshold_hu)
            
            # Compute 3D body mask for the slice range
            volume_slice_range = volume_gray[slice_min:slice_max + 1]
            
            # Convert volume to HU for mask computation if DICOM
            if hu_conv is not None:
                slope, intercept = hu_conv
                volume_slice_range = volume_slice_range.astype(np.float32) * slope + intercept
            
            # Compute body mask
            body_mask_3d = images.compute_body_mask_3d(
                volume_slice_range,
                spacing=spacing,
                body_threshold=body_threshold,
                closing_radius_mm=closing_radius_mm,
                min_component_size_vox=min_component_size_vox,
                outside_only=outside_only,
            )
            
            # Apply non-body removal: set non-body pixels to background fill
            # Convert background fill from HU to raw intensity if needed
            if hu_conv is not None:
                # DICOM: convert HU to raw intensity: HU = raw * slope + intercept
                # So: raw = (HU - intercept) / slope
                slope, intercept = hu_conv
                background_fill_raw = (background_fill_hu - intercept) / slope
            else:
                # No HU conversion: background_fill is already in raw intensity domain
                # If negative, it's likely meant to be black (0)
                background_fill_raw = background_fill_hu if background_fill_hu >= 0 else 0.0
            
            # Clip to appropriate dtype range and convert to int
            if volume_rgb.dtype == np.uint8:
                fill_val = int(np.clip(background_fill_raw, 0, 255))
            elif volume_rgb.dtype == np.uint16:
                fill_val = int(np.clip(background_fill_raw, 0, 65535))
            else:
                fill_val = float(background_fill_raw)
            
            fill_rgb = np.array([fill_val, fill_val, fill_val], dtype=volume_rgb.dtype)
            
            # Apply removal to RGB volume
            for local_z_idx, global_z_idx in enumerate(range(slice_min, slice_max + 1)):
                if global_z_idx < len(volume_rgb):
                    non_body_mask = ~body_mask_3d[local_z_idx]
                    volume_rgb[global_z_idx][non_body_mask] = fill_rgb
    
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
        remove_non_grayscale=cfg.remove_non_grayscale,  # Legacy flag
        remove_overlays=cfg.remove_overlays,  # Control automatic overlay removal (grayscale conversion)
        object_mask=cfg.object_mask,  # Legacy support
        object_mask_slice_index=cfg.object_mask_slice_index,  # Legacy support
        non_grayscale_slice_ranges=cfg.non_grayscale_slice_ranges,  # New: slice ranges for non-grayscale removal
        object_removal_objects=cfg.object_removal_objects,  # New: objects with slice ranges
        progress_cb=progress_cb,
    )

    if progress_cb is not None:
        progress_cb("processing", total, total, total)  # Mark processing as complete

    # ------------------------------------------------------------------
    # 3. Save processed slices to processed_dir
    # ------------------------------------------------------------------
    _ensure_dir(processed_dir)

    # Determine which slices to export
    if cfg.export_slice_range is not None:
        min_slice, max_slice = cfg.export_slice_range
        # Clamp to valid range
        min_slice = max(0, min(min_slice, len(processed_volume) - 1))
        max_slice = max(min_slice, min(max_slice, len(processed_volume) - 1))
        export_indices = list(range(min_slice, max_slice + 1))
    else:
        # Export all slices
        export_indices = list(range(len(processed_volume)))
    
    export_total = len(export_indices)
    
    for export_idx, z_idx in enumerate(export_indices, start=1):
        if progress_cb is not None:
            progress_cb("saving", export_idx, export_total, export_total)

        slice_arr = processed_volume[z_idx]
        
        # Determine output filename
        if cfg.export_prefix is not None:
            # Use prefix + sequential index based on Z order (0-based: 0, 1, 2, ...)
            src_path = slice_paths[z_idx]
            ext = src_path.suffix
            # Sequential numbering follows the Z order (0, 1, 2, ...)
            sequential_index = export_idx - 1  # Convert from 1-based to 0-based
            out_filename = f"{cfg.export_prefix}_{sequential_index:05d}{ext}"
            out_path = processed_dir / out_filename
        else:
            # Preserve original filename
            src_path = slice_paths[z_idx]
            out_path = processed_dir / src_path.name
        
        # If original was DICOM, pass it as reference to preserve metadata
        reference_dicom = src_path if images._is_dicom_file(src_path) else None
        images.save_image_rgb(out_path, slice_arr, reference_dicom=reference_dicom)
    
    if progress_cb is not None:
        progress_cb("saving", export_total, export_total, export_total)  # Mark saving as complete

    return processed_dir
