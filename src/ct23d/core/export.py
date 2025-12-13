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


def export_bins_to_meshes(
    volume: np.ndarray,
    bins: Sequence[IntensityBin],
    config: MeshingConfig,
    output_dir: Path,
    filename_prefix: str = "ct_export",
    export_mode: str = "separate",  # "separate" or "combined"
    format_name: str = "PLY",  # "PLY", "OBJ", "STL"
    opacity: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    phase_progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
) -> List[Path]:
    """
    Export bins to PLY meshes with bin colors and optional opacity.
    
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
        
    Returns
    -------
    List[Path]
        Paths to generated PLY files
    """
    # Add opacity to config temporarily
    if opacity is not None:
        config.opacity = opacity
    
    # Convert to grayscale if needed
    if volume.ndim == 4 and volume.shape[-1] == 3:
        volume_gray = volmod.to_grayscale(volume)
        volume_color = volume
    elif volume.ndim == 3:
        volume_gray = volume.astype(np.uint8)
        volume_color = None
    else:
        raise ValueError(f"Volume must be 3D or 4D, got shape {volume.shape}")
    
    enabled_bins = [b for b in bins if b.enabled]
    if not enabled_bins:
        return []
    
    if export_mode == "combined":
        # Generate all meshes and combine them into a single file
        all_vertices = []
        all_faces = []
        all_colors = []
        
        from . import bins as binsmod
        total = len(enabled_bins)
        
        for i, bin_ in enumerate(enabled_bins):
            if progress_callback:
                progress_callback(i, total)
            
            # Build mask for this bin
            bin_mask = binsmod.build_bin_mask(volume_gray, bin_)
            
            if config.skip_empty_bins and np.count_nonzero(bin_mask) == 0:
                continue
            
            # Clean and smooth
            bin_mask_clean = meshing_mod.clean_bin_mask(bin_mask, config)
            if config.skip_empty_bins and np.count_nonzero(bin_mask_clean) == 0:
                continue
            
            bin_mask_smooth = meshing_mod.smooth_mask(bin_mask_clean, config.smoothing_sigma)
            if config.skip_empty_bins and np.count_nonzero(bin_mask_smooth) == 0:
                continue
            
            # Extract mesh
            verts, faces, colors = meshing_mod.extract_mesh(bin_mask_smooth, volume_color, config)
            
            if verts.size == 0 or faces.size == 0:
                continue
            
            # Apply bin color (override volume colors if bin color is set)
            if bin_.color is not None:
                # Colors are stored as sRGB-compatible RGB values (0-255) for Blender
                r, g, b = bin_.color
                # Ensure values are in [0, 1] range and convert to sRGB uint8
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                # Convert to sRGB uint8 (0-255) - these are already in sRGB color space
                bin_color_rgb = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
                colors = np.tile(bin_color_rgb, (verts.shape[0], 1))
            elif colors is None:
                # If no colors from volume and no bin color, use default gray
                colors = np.full((verts.shape[0], 3), 128, dtype=np.uint8)
            
            all_vertices.append(verts)
            all_faces.append(faces)
            all_colors.append(colors)
        
        if not all_vertices:
            return []
        
        # Combine all meshes
        combined_verts, combined_faces, combined_colors = combine_meshes(
            all_vertices, all_faces, all_colors
        )
        
        # Save combined mesh
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{filename_prefix}_combined.ply"
        meshing_mod.save_mesh_ply(out_path, combined_verts, combined_faces, combined_colors, opacity=opacity)
        
        if progress_callback:
            progress_callback(total, total)
        
        return [out_path]
    
    else:  # export_mode == "separate"
        # Use existing function which exports each bin separately
        # Currently only PLY is implemented
        if format_name.upper() != "PLY":
            raise ValueError(
                f"Format '{format_name}' is not yet implemented. "
                "Currently only PLY format is supported for separate file export."
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
        )

