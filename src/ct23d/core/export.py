"""
Export functions for combining multiple bin meshes into single or multiple files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .models import IntensityBin, MeshingConfig
from . import meshing as meshing_mod
from . import volume as volmod


def combine_meshes(
    vertices_list: List[np.ndarray],
    faces_list: List[np.ndarray],
    colors_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine multiple meshes into a single mesh.
    
    Parameters
    ----------
    vertices_list : List[np.ndarray]
        List of vertex arrays, each shape [N, 3]
    faces_list : List[np.ndarray]
        List of face arrays, each shape [M, 3]
    colors_list : List[np.ndarray]
        List of color arrays, each shape [N, 3] (uint8 RGB)
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Combined (vertices, faces, colors)
    """
    if not vertices_list:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Combine vertices
    combined_verts = np.vstack(vertices_list)
    
    # Combine colors
    combined_colors = np.vstack(colors_list)
    
    # Combine faces with offset indices
    combined_faces = []
    vertex_offset = 0
    
    for verts, faces in zip(vertices_list, faces_list):
        if faces.size > 0:
            # Offset face indices by current vertex count
            offset_faces = faces + vertex_offset
            combined_faces.append(offset_faces)
        vertex_offset += verts.shape[0]
    
    if combined_faces:
        combined_faces = np.vstack(combined_faces)
    else:
        combined_faces = np.array([]).reshape(0, 3)
    
    return combined_verts, combined_faces, combined_colors


def _estimate_file_size_from_meshes(
    all_vertices: List[np.ndarray],
    all_faces: List[np.ndarray],
    format_name: str,
    export_colors: bool,
    export_opacity: bool,
    stl_binary: bool,
    use_optimized: bool = True,  # New parameter: use optimized format estimates
) -> int:
    """
    Estimate total file size in bytes from actual mesh data.
    
    This is much more accurate than voxel-based estimates since it uses
    the actual vertex and face counts from extracted meshes.
    
    Parameters
    ----------
    use_optimized : bool
        If True, uses optimized PLY format estimates (4 decimal places).
        If False, uses standard PLY format estimates (6 decimal places).
    """
    format_upper = format_name.upper()
    
    # Calculate total vertices and faces
    total_vertices = sum(v.shape[0] for v in all_vertices)
    total_faces = sum(f.shape[0] for f in all_faces)
    
    if format_upper == "STL":
        if stl_binary:
            # Binary STL: 80 byte header + 4 bytes (face count) + 50 bytes per face
            return 80 + 4 + (total_faces * 50)
        else:
            # ASCII STL: header + face data (rough estimate)
            # Each face: ~100 bytes
            return 100 + (total_faces * 100)
    else:  # PLY
        # PLY header: ~300 bytes
        header_size = 300
        
        # Vertex line size depends on properties and format
        # IMPORTANT: Optimized PLY ALWAYS exports colors (from bins), regardless of export_colors setting
        if use_optimized:
            # Optimized format uses 4 decimal places and ALWAYS includes colors from bins
            if export_opacity:
                vertex_line_size = 50  # "x.4f y.4f z.4f r g b a\n" (colors always present, opacity optional)
            else:
                vertex_line_size = 45  # "x.4f y.4f z.4f r g b\n" (colors always present)
        else:
            # Standard format uses 6 decimal places
            if export_colors and export_opacity:
                vertex_line_size = 50  # "x.6f y.6f z.6f r g b a\n"
            elif export_colors or export_opacity:
                vertex_line_size = 45  # "x.6f y.6f z.6f r g b\n" or "x.6f y.6f z.6f a\n"
            else:
                vertex_line_size = 30  # "x.6f y.6f z.6f\n"
        
        # Face line: ~18 bytes "3 v1 v2 v3\n"
        face_line_size = 18
        
        return header_size + (total_vertices * vertex_line_size) + (total_faces * face_line_size)


def _calculate_file_size_estimate(
    volume_gray: np.ndarray,
    enabled_bins: List[IntensityBin],
    config: MeshingConfig,
    volume_color: Optional[np.ndarray],
    export_mode: str,
    format_name: str,
    export_colors: bool,
    export_opacity: bool,
    stl_binary: bool,
    phase_progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
) -> int:
    """
    Calculate estimated file size by quickly extracting meshes for all bins.
    
    This function builds masks, extracts meshes, and calculates file size
    based on actual vertex/face counts. It's more accurate than voxel-based
    estimates but takes longer (still faster than full export since we skip
    file writing).
    
    For PLY format, uses the optimized meshing algorithm to match actual export.
    
    Returns estimated total file size in bytes.
    """
    from . import bins as binsmod
    
    # For PLY format, use optimized meshing algorithm to get accurate vertex/face counts
    use_optimized = (format_name.upper() == "PLY")
    
    if use_optimized and export_mode == "combined":
        # Use optimized combined mask approach for more accurate estimation
        from . import meshing_optimized as opt_meshing
        
        if phase_progress_callback:
            phase_progress_callback("Calculating file size", 0, 1, 0)
        
        # Build combined mask (same as optimized export)
        combined_mask = opt_meshing.build_combined_bin_mask(volume_gray, enabled_bins)
        
        if np.count_nonzero(combined_mask) == 0:
            if phase_progress_callback:
                phase_progress_callback("Calculating file size", 1, 1, 0)
            return 0
        
        # Apply same filtering/smoothing as export
        if config.enable_component_filtering:
            combined_mask = meshing_mod.clean_bin_mask(combined_mask, config)
        if config.enable_smoothing:
            combined_mask = meshing_mod.smooth_mask(combined_mask, config.smoothing_sigma)
        
        if np.count_nonzero(combined_mask) == 0:
            if phase_progress_callback:
                phase_progress_callback("Calculating file size", 1, 1, 0)
            return 0
        
        # Extract mesh using optimized algorithm (no colors needed for size estimation)
        try:
            verts, faces, _ = opt_meshing.extract_optimized_mesh(
                volume_gray, combined_mask, enabled_bins, config, None, progress_callback=None
            )
            if verts.size > 0 and faces.size > 0:
                # Calculate size using optimized format
                # IMPORTANT: Optimized PLY ALWAYS exports colors (from bins), regardless of export_colors setting
                # So we pass export_colors=True for the size calculation, but export_opacity is checked correctly
                estimated_size = _estimate_file_size_from_meshes(
                    [verts], [faces], format_name, export_colors=True, export_opacity=export_opacity, stl_binary=stl_binary, use_optimized=True
                )
                if phase_progress_callback:
                    phase_progress_callback("Calculating file size", 1, 1, estimated_size)
                return estimated_size
        except Exception:
            # If optimized extraction fails, fall back to standard method
            pass
    
    # Standard estimation method (for STL or if optimized fails)
    total = len(enabled_bins)
    all_vertices = []
    all_faces = []
    
    if phase_progress_callback:
        phase_progress_callback("Calculating file size", 0, total, 0)
    
    for i, bin_ in enumerate(enabled_bins):
        if phase_progress_callback:
            phase_progress_callback("Calculating file size", i + 1, total, 0)
        
        # Build mask
        bin_mask = binsmod.build_bin_mask(volume_gray, bin_)
        
        if config.skip_empty_bins and np.count_nonzero(bin_mask) == 0:
            continue
        
        # Clean if enabled
        if config.enable_component_filtering:
            bin_mask_clean = meshing_mod.clean_bin_mask(bin_mask, config)
        else:
            bin_mask_clean = bin_mask
        
        if config.skip_empty_bins and np.count_nonzero(bin_mask_clean) == 0:
            continue
        
        # Smooth if enabled
        if config.enable_smoothing:
            bin_mask_smooth = meshing_mod.smooth_mask(bin_mask_clean, config.smoothing_sigma)
        else:
            bin_mask_smooth = bin_mask_clean
        
        if config.skip_empty_bins and np.count_nonzero(bin_mask_smooth) == 0:
            continue
        
        # Extract mesh to get vertex/face counts (don't need colors for size estimation)
        try:
            verts, faces, _ = meshing_mod.extract_mesh(bin_mask_smooth, None, config)
            if verts.size > 0 and faces.size > 0:
                all_vertices.append(verts)
                all_faces.append(faces)
        except Exception:
            # If extraction fails, skip this bin
            continue
    
    if not all_vertices:
        if phase_progress_callback:
            phase_progress_callback("Calculating file size", total, total, 0)
        return 0
    
    # Calculate estimated file size
    if export_mode == "combined":
        # For combined mode, combine all meshes first
        combined_verts, combined_faces, _ = combine_meshes(all_vertices, all_faces, [np.zeros((v.shape[0], 3), dtype=np.uint8) for v in all_vertices])
        # IMPORTANT: For optimized PLY, colors are ALWAYS exported (from bins)
        # For standard format, respect export_colors setting
        colors_always_exported = use_optimized or export_colors
        estimated_size = _estimate_file_size_from_meshes(
            [combined_verts], [combined_faces], format_name, export_colors=colors_always_exported, export_opacity=export_opacity, stl_binary=stl_binary, use_optimized=use_optimized
        )
    else:
        # For separate mode, calculate size for each file and sum
        total_size = 0
        # IMPORTANT: For optimized PLY, colors are ALWAYS exported (from bins)
        colors_always_exported = use_optimized or export_colors
        for verts, faces in zip(all_vertices, all_faces):
            total_size += _estimate_file_size_from_meshes(
                [verts], [faces], format_name, export_colors=colors_always_exported, export_opacity=export_opacity, stl_binary=stl_binary, use_optimized=use_optimized
            )
        estimated_size = total_size
    
    if phase_progress_callback:
        # Pass estimated size as overall_total so it can be stored for later use
        phase_progress_callback("Calculating file size", total, total, estimated_size)
    
    return estimated_size


def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human-readable string."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"


def export_bins_to_meshes(
    volume: np.ndarray,
    bins: Sequence[IntensityBin],
    config: MeshingConfig,
    output_dir: Path,
    filename_prefix: str = "ct_export",
    export_mode: str = "separate",  # "separate" or "combined"
    format_name: str = "PLY",  # "PLY", "OBJ", "STL"
    opacity: Optional[float] = None,
    stl_binary: bool = True,  # For STL format: True=binary, False=ASCII
    progress_callback: Optional[Callable[[int, int], None]] = None,
    phase_progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
    bytes_written_callback: Optional[Callable[[int], None]] = None,  # New: track actual bytes written
    estimated_file_size: Optional[int] = None,  # New: pre-calculated estimated size
) -> List[Path]:
    """
    Export bins to PLY meshes with bin colors and optional opacity.
    
    Uses optimized meshing algorithm for better performance and file size.
    
    Parameters
    ----------
    volume : np.ndarray
        Volume data, shape (Z, Y, X, 3) for RGB or (Z, Y, X) for grayscale
    bins : Sequence[IntensityBin]
        List of intensity bins to export
    config : MeshingConfig
        Meshing configuration
    output_dir : Path
        Directory where meshes will be saved
    filename_prefix : str
        Prefix for output filenames
    export_mode : str
        "separate" - one file per bin, "combined" - all bins in one file
    opacity : Optional[float]
        Opacity value (0.0 to 1.0) to apply to all meshes
    progress_callback : Optional[callable]
        Optional callback function(current, total) for progress updates
    phase_progress_callback : Optional[callable]
        Optional callback function(phase, current, phase_total, overall_total) for phase-aware progress
    bytes_written_callback : Optional[callable]
        Optional callback function(bytes_written) to track actual bytes written to files
    estimated_file_size : Optional[int]
        Pre-calculated estimated file size in bytes (for progress tracking)
        
    Returns
    -------
    List[Path]
        Paths to generated PLY files
    """
    # Add opacity to config temporarily
    if opacity is not None:
        config.opacity = opacity
    
    # Use optimized meshing algorithm for PLY format (faster, better file size)
    # For STL or other formats, fall back to old implementation
    if format_name.upper() == "PLY":
        from . import meshing_optimized as opt_meshing
        
        # Wrap progress callbacks to match expected format
        def wrapped_phase_progress_cb(phase: str, current: int, phase_total: int, overall_total: int) -> None:
            # The optimized meshing uses different phase names, adapt them
            if phase_progress_callback:
                # Map optimized phases to expected phases
                phase_map = {
                    "Building combined mask": "Building masks",
                    "Extracting mesh": "Extracting meshes",
                    "Saving file": "Saving files",
                    "Processing bins": "Building masks",  # First phase of separate mode
                }
                mapped_phase = phase_map.get(phase, phase)
                # Clamp values to prevent Qt overflow
                MAX_INT = 2147483647
                current_clamped = min(current, MAX_INT)
                phase_total_clamped = min(phase_total, MAX_INT)
                overall_total_clamped = min(overall_total, MAX_INT)
                phase_progress_callback(mapped_phase, current_clamped, phase_total_clamped, overall_total_clamped)
        
        try:
            return opt_meshing.export_optimized_mesh(
                volume=volume,
                bins=bins,
                config=config,
                output_dir=output_dir,
                filename_prefix=filename_prefix,
                export_mode=export_mode,
                opacity=opacity,
                progress_callback=progress_callback,
                phase_progress_callback=wrapped_phase_progress_cb,
                bytes_written_callback=bytes_written_callback,
            )
        except Exception as e:
            # If optimized meshing fails, fall back to original implementation
            import warnings
            warnings.warn(f"Optimized meshing failed, falling back to standard implementation: {e}")
            # Continue to fallback implementation below
    
    # Fallback to original implementation for non-PLY formats or if optimization disabled
    # Convert to intensity if needed (using max channel for better preservation
    # of information from colored overlays in medical imaging)
    if volume.ndim == 4 and volume.shape[-1] == 3:
        volume_gray = volmod.to_intensity_max(volume)
        volume_color = volume
    elif volume.ndim == 3:
        # Preserve dtype (don't cast uint16 to uint8 for DICOM data)
        volume_gray = volume
        volume_color = None
    else:
        raise ValueError(f"Volume must be 3D or 4D, got shape {volume.shape}")
    
    enabled_bins = [b for b in bins if b.enabled]
    if not enabled_bins:
        return []
    
    # Track actual bytes written (cumulative total)
    total_bytes_written = [0]  # Use list to allow modification in nested functions
    
    def track_bytes_written(new_total_bytes: int) -> None:
        """Update total bytes written and notify callback."""
        total_bytes_written[0] = new_total_bytes
        if bytes_written_callback:
            bytes_written_callback(new_total_bytes)
    
    if export_mode == "combined":
        # Generate all meshes and combine them into a single file
        all_vertices = []
        all_faces = []
        all_colors = []
        
        from . import bins as binsmod
        total = len(enabled_bins)
        
        # Phase-aware progress for combined mode
        phase = "Building masks"
        phase_current = 0
        phase_total = total
        overall_current = 0
        overall_total = total * 3  # 3 phases: build, extract, save
        
        if phase_progress_callback:
            phase_progress_callback(phase, 0, phase_total, overall_total)
        
        # Phase 1: Build all masks first
        bin_masks = []
        for i, bin_ in enumerate(enabled_bins):
            if progress_callback:
                progress_callback(i, total)
            
            if phase_progress_callback:
                phase_current = i + 1
                overall_current = i + 1
                phase_progress_callback(phase, phase_current, phase_total, overall_total)
            
            # Build mask for this bin
            bin_mask = binsmod.build_bin_mask(volume_gray, bin_)
            
            if phase_progress_callback:
                phase_progress_callback(phase, i + 1, phase_total, overall_total)
            
            if config.skip_empty_bins and np.count_nonzero(bin_mask) == 0:
                bin_masks.append(None)
                continue
            
            # Clean
            if config.enable_component_filtering:
                bin_mask_clean = meshing_mod.clean_bin_mask(bin_mask, config)
            else:
                bin_mask_clean = bin_mask
            if config.skip_empty_bins and np.count_nonzero(bin_mask_clean) == 0:
                bin_masks.append(None)
                continue
            
            # Smooth
            if config.enable_smoothing:
                bin_mask_smooth = meshing_mod.smooth_mask(bin_mask_clean, config.smoothing_sigma)
            else:
                bin_mask_smooth = bin_mask_clean
            if config.skip_empty_bins and np.count_nonzero(bin_mask_smooth) == 0:
                bin_masks.append(None)
                continue
            
            bin_masks.append(bin_mask_smooth)
        
        if phase_progress_callback:
            phase_progress_callback(phase, phase_total, phase_total, overall_total)
        
        # Phase 2: Extract all meshes
        if phase_progress_callback:
            phase = "Extracting meshes"
            phase_current = 0
            phase_total = total
            overall_current = total
            phase_progress_callback(phase, 0, phase_total, overall_total)
        
        for i, (bin_, bin_mask_smooth) in enumerate(zip(enabled_bins, bin_masks)):
            if bin_mask_smooth is None:
                continue
            
            def extract_progress_cb(current: int, total: int) -> None:
                if phase_progress_callback:
                    extraction_progress = current / max(total, 1)
                    phase_current_scaled = i + extraction_progress
                    overall_current_scaled = total + phase_current_scaled
                    phase_progress_callback(phase, int(phase_current_scaled), phase_total, overall_total)
            
            verts, faces, colors = meshing_mod.extract_mesh(bin_mask_smooth, volume_color, config, progress_callback=extract_progress_cb)
            
            if verts.size == 0 or faces.size == 0:
                continue
            
            if phase_progress_callback:
                phase_current = i + 1
                overall_current = total + i + 1
                phase_progress_callback(phase, phase_current, phase_total, overall_total)
            
            # Apply bin color
            if bin_.color is not None:
                r, g, b = bin_.color
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                bin_color_rgb = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
                colors = np.tile(bin_color_rgb, (verts.shape[0], 1))
            elif colors is None:
                colors = np.full((verts.shape[0], 3), 128, dtype=np.uint8)
            
            all_vertices.append(verts)
            all_faces.append(faces)
            all_colors.append(colors)
        
        if not all_vertices:
            return []
        
        if phase_progress_callback:
            phase_progress_callback(phase, phase_total, phase_total, overall_total)
        
        # Combine all meshes
        combined_verts, combined_faces, combined_colors = combine_meshes(
            all_vertices, all_faces, all_colors
        )
        
        # Switch to saving files phase
        if phase_progress_callback:
            phase = "Saving files"
            phase_current = 0
            phase_total = estimated_file_size if estimated_file_size else 1
            overall_current = total * 2
            phase_progress_callback(phase, 0, phase_total, overall_total)
        
        # Save combined mesh
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        format_upper = format_name.upper()
        
        # Create progress callback for file saving based on bytes
        def file_save_progress_cb(bytes_written: int, total_bytes: int) -> None:
            # bytes_written is cumulative bytes written so far for this file
            # Track it directly
            track_bytes_written(bytes_written)
            if phase_progress_callback and estimated_file_size:
                # Progress based on actual bytes / estimated bytes
                # Use bytes_written as current progress, estimated_file_size as total
                phase_progress_callback(phase, bytes_written, estimated_file_size, estimated_file_size)
        
        if format_upper == "STL":
            out_path = output_dir / f"{filename_prefix}_combined.stl"
            meshing_mod.save_mesh_stl(out_path, combined_verts, combined_faces, binary=stl_binary, progress_callback=file_save_progress_cb)
        else:  # Default to PLY
            out_path = output_dir / f"{filename_prefix}_combined.ply"
            meshing_mod.save_mesh_ply(out_path, combined_verts, combined_faces, combined_colors, opacity=opacity, progress_callback=file_save_progress_cb)
        
        # Update final bytes written from actual file size
        actual_file_size = out_path.stat().st_size
        track_bytes_written(actual_file_size)
        
        if progress_callback:
            progress_callback(total, total)
        
        if phase_progress_callback:
            phase_progress_callback(phase, estimated_file_size if estimated_file_size else actual_file_size, estimated_file_size if estimated_file_size else actual_file_size, estimated_file_size if estimated_file_size else actual_file_size)
        
        return [out_path]
    
    else:  # export_mode == "separate"
        # Use existing function which exports each bin separately
        format_upper = format_name.upper()
        if format_upper not in ("PLY", "STL"):
            raise ValueError(
                f"Format '{format_name}' is not yet implemented. "
                "Currently only PLY and STL formats are supported for separate file export."
            )
        
        config.output_dir = output_dir
        config.output_prefix = filename_prefix
        return meshing_mod.generate_meshes_for_bins(
            volume_gray=volume_gray,
            bins=enabled_bins,
            cfg=config,
            volume_color=volume_color,
            progress_callback=progress_callback,
            phase_progress_callback=phase_progress_callback,
            export_format=format_upper,
            stl_binary=stl_binary,
            bytes_written_callback=bytes_written_callback,
            estimated_file_size=estimated_file_size,
        )