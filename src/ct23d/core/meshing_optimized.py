"""
Optimized meshing algorithm focused on:
- File size optimization
- Minimal information loss
- Fast execution
- Accurate voxel spacing
- Direct bin color assignment
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from skimage import measure

from .models import IntensityBin, MeshingConfig
from . import bins as binsmod


def build_combined_bin_mask(
    volume_gray: np.ndarray,
    bins: Sequence[IntensityBin],
) -> np.ndarray:
    """
    Build a single combined mask for all enabled bins efficiently.
    
    Only pixels within enabled bin ranges are included.
    This is more efficient than building separate masks for each bin.
    
    Parameters
    ----------
    volume_gray : np.ndarray
        3D grayscale volume [Z, Y, X]
    bins : Sequence[IntensityBin]
        List of intensity bins (only enabled ones are processed)
        
    Returns
    -------
    np.ndarray
        Boolean mask [Z, Y, X] where True means pixel is in any enabled bin
    """
    enabled_bins = [b for b in bins if b.enabled]
    if not enabled_bins:
        return np.zeros_like(volume_gray, dtype=bool)
    
    # Build combined mask: True if pixel is in ANY enabled bin
    combined_mask = np.zeros_like(volume_gray, dtype=bool)
    
    for bin_ in enabled_bins:
        # Build mask for this bin range [low, high)
        bin_mask = binsmod.build_bin_mask(volume_gray, bin_)
        combined_mask |= bin_mask  # Union with combined mask
    
    return combined_mask


def extract_optimized_mesh(
    volume_gray: np.ndarray,
    combined_mask: np.ndarray,
    bins: Sequence[IntensityBin],
    cfg: MeshingConfig,
    volume_color: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract mesh from combined bin mask with optimized color assignment.
    
    This function:
    1. Extracts a single mesh from the combined mask using marching cubes
    2. Assigns colors based on bin colors directly (no volume sampling if bin.color is set)
    3. Applies voxel spacing correctly
    4. Optimizes for minimal information loss
    
    Parameters
    ----------
    volume_gray : np.ndarray
        3D grayscale volume [Z, Y, X] for determining which bin each vertex belongs to
    combined_mask : np.ndarray
        Boolean mask [Z, Y, X] of all enabled bins combined
    bins : Sequence[IntensityBin]
        List of intensity bins (only enabled ones)
    cfg : MeshingConfig
        Meshing configuration with voxel spacing
    volume_color : Optional[np.ndarray]
        Optional 4D [Z, Y, X, 3] uint8 volume for sampling colors (only if bin.color is None)
    progress_callback : Optional[Callable]
        Optional progress callback(current, total)
        
    Returns
    -------
    (vertices, faces, colors)
        vertices: [N, 3] float32 array in physical units (mm)
        faces: [M, 3] int32 array of vertex indices
        colors: [N, 3] uint8 array of RGB colors
    """
    if progress_callback:
        progress_callback(0, 4)  # Step 0: Starting
    
    # Step 1: Extract mesh using marching cubes
    vol_f = combined_mask.astype(np.float32)
    
    if progress_callback:
        progress_callback(1, 4)  # Step 1: Running marching cubes
    
    # Run marching cubes in voxel coordinates (spacing=1,1,1)
    verts_vox, faces, _normals, _values = measure.marching_cubes(
        vol_f,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
    )
    
    if verts_vox.size == 0 or faces.size == 0:
        # Empty mesh
        return (
            np.array([]).reshape(0, 3).astype(np.float32),
            np.array([]).reshape(0, 3).astype(np.int32),
            np.array([]).reshape(0, 3).astype(np.uint8),
        )
    
    if progress_callback:
        progress_callback(2, 4)  # Step 2: Assigning colors
    
    # Step 2: Assign colors based on bin colors
    enabled_bins = [b for b in bins if b.enabled]
    n_verts = verts_vox.shape[0]
    colors = np.zeros((n_verts, 3), dtype=np.uint8)
    
    # Round vertex voxel coordinates to nearest integer for sampling
    z_idx = np.clip(np.round(verts_vox[:, 0]).astype(int), 0, volume_gray.shape[0] - 1)
    y_idx = np.clip(np.round(verts_vox[:, 1]).astype(int), 0, volume_gray.shape[1] - 1)
    x_idx = np.clip(np.round(verts_vox[:, 2]).astype(int), 0, volume_gray.shape[2] - 1)
    
    # Sample intensities at vertex locations
    vertex_intensities = volume_gray[z_idx, y_idx, x_idx]
    
    # Assign colors: for each vertex, find which bin it belongs to and use that bin's color
    for bin_ in enabled_bins:
        # Find vertices that belong to this bin [low, high)
        in_bin = (vertex_intensities >= bin_.low) & (vertex_intensities < bin_.high)
        
        if np.any(in_bin):
            if bin_.color is not None:
                # Use bin color directly (no volume sampling needed)
                r, g, b = bin_.color
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                bin_color_rgb = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
                colors[in_bin] = bin_color_rgb
            elif volume_color is not None:
                # Sample from volume color if bin doesn't have a color
                # Use the same indices
                colors[in_bin] = volume_color[z_idx[in_bin], y_idx[in_bin], x_idx[in_bin]].astype(np.uint8)
            else:
                # Default gray if no bin color and no volume color
                colors[in_bin] = [128, 128, 128]
    
    # Handle vertices not in any bin (shouldn't happen if mask is correct, but be safe)
    no_bin = np.all(colors == 0, axis=1)
    if np.any(no_bin):
        colors[no_bin] = [128, 128, 128]  # Default gray
    
    if progress_callback:
        progress_callback(3, 4)  # Step 3: Applying voxel spacing
    
    # Step 3: Scale to physical units using cfg.spacing = (z, y, x)
    sz, sy, sx = cfg.spacing
    scale = np.array([sz, sy, sx], dtype=np.float32)
    verts_phys = verts_vox * scale[None, :]
    
    if progress_callback:
        progress_callback(4, 4)  # Step 4: Complete
    
    return (
        verts_phys.astype(np.float32),
        faces.astype(np.int32),
        colors.astype(np.uint8),
    )


def save_mesh_ply_optimized(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    opacity: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Save mesh as optimized PLY file with minimal file size.
    
    Optimizations:
    - Reduced precision for vertices (4 decimal places instead of 6)
    - Efficient batch writing
    - Minimal header
    
    Parameters
    ----------
    path : Path
        Destination file path
    vertices : np.ndarray
        Array [N, 3] of float vertices
    faces : np.ndarray
        Array [M, 3] of int vertex indices
    colors : np.ndarray
        Array [N, 3] of uint8 RGB colors
    opacity : Optional[float]
        Optional opacity value (0.0 to 1.0)
    progress_callback : Optional[Callable]
        Optional callback(bytes_written, total_bytes) for progress
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]
    # Opacity is always applied per-vertex when provided (and < 1.0)
    # If opacity is None or >= 1.0, don't add alpha channel
    has_opacity_in_file = opacity is not None and opacity < 1.0
    alpha_value = int(opacity * 255) if has_opacity_in_file else None
    
    # Calculate header size accurately
    header_lines_list = [
        "ply\n",
        "format ascii 1.0\n",
        f"element vertex {n_verts}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "property uchar red\n",  # Colors are ALWAYS exported in optimized PLY
        "property uchar green\n",
        "property uchar blue\n",
    ]
    if has_opacity_in_file:
        header_lines_list.append("property uchar alpha\n")
    header_lines_list.extend([
        f"element face {n_faces}\n",
        "property list uchar int vertex_indices\n",
        "end_header\n",
    ])
    header_size = sum(len(line.encode('utf-8')) for line in header_lines_list)
    
    # Calculate actual file size by sampling lines
    # Sample vertices to get average line length
    sample_verts = min(100, n_verts) if n_verts > 0 else 0
    total_vertex_bytes = 0
    if sample_verts > 0:
        for i in range(sample_verts):
            v = vertices[i]
            c = colors[i]
            if has_opacity_in_file:
                line = f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])} {alpha_value}\n"
            else:
                line = f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
            total_vertex_bytes += len(line.encode('utf-8'))
        avg_vertex_line_size = total_vertex_bytes / sample_verts
    else:
        # Fallback: estimate based on format
        # "x.4f y.4f z.4f r g b [a]\n" - typically 40-50 bytes
        avg_vertex_line_size = 48 if has_opacity_in_file else 43
    
    # Sample faces to get average line length
    sample_faces = min(100, n_faces) if n_faces > 0 else 0
    total_face_bytes = 0
    if sample_faces > 0:
        for i in range(sample_faces):
            a, b, c = faces[i]
            face_line = f"3 {int(a)} {int(b)} {int(c)}\n"
            total_face_bytes += len(face_line.encode('utf-8'))
        avg_face_line_size = total_face_bytes / sample_faces
    else:
        avg_face_line_size = 18  # "3 v1 v2 v3\n" - roughly 18 bytes
    
    estimated_total_size = int(header_size + (n_verts * avg_vertex_line_size) + (n_faces * avg_face_line_size))
    
    if progress_callback:
        progress_callback(0, estimated_total_size)
    
    bytes_written = 0
    
    with path.open("w", encoding="utf-8") as f:
        # Write header
        header_lines = [
            "ply\n",
            "format ascii 1.0\n",
            f"element vertex {n_verts}\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
            "property uchar red\n",
            "property uchar green\n",
            "property uchar blue\n",
        ]
        if has_opacity:
            header_lines.append("property uchar alpha\n")
        header_lines.extend([
            f"element face {n_faces}\n",
            "property list uchar int vertex_indices\n",
            "end_header\n",
        ])
        
        for line in header_lines:
            f.write(line)
            bytes_written += len(line.encode('utf-8'))
        
        if progress_callback:
            progress_callback(bytes_written, estimated_total_size)
        
        # Write vertices in batches (reduced precision: 4 decimals)
        batch_size = max(1000, n_verts // 100)  # Update every 1% or every 1000 vertices
        for i in range(n_verts):
            v = vertices[i]
            c = colors[i]
            
            if has_opacity_in_file:
                # Write vertex with colors and alpha: "x y z r g b a\n"
                line = f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])} {alpha_value}\n"
                bytes_written += len(line.encode('utf-8'))
            else:
                # Write vertex with colors only: "x y z r g b\n"
                # Colors are ALWAYS exported in optimized PLY (from bins or default gray)
                line = f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
                bytes_written += len(line.encode('utf-8'))
            
            f.write(line)
            
            if progress_callback and (i + 1) % batch_size == 0:
                progress_callback(bytes_written, estimated_total_size)
        
        # Write faces in batches
        face_batch_size = max(1000, n_faces // 100)
        for i, (a, b, c) in enumerate(faces):
            face_line = f"3 {int(a)} {int(b)} {int(c)}\n"
            f.write(face_line)
            bytes_written += len(face_line.encode('utf-8'))
            
            if progress_callback and (i + 1) % face_batch_size == 0:
                progress_callback(bytes_written, estimated_total_size)
        
        if progress_callback:
            progress_callback(estimated_total_size, estimated_total_size)
        
        # Flush and sync
        f.flush()
        import os
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass


def export_optimized_mesh(
    volume: np.ndarray,
    bins: Sequence[IntensityBin],
    config: MeshingConfig,
    output_dir: Path,
    filename_prefix: str = "ct_export",
    export_mode: str = "combined",  # "separate" or "combined"
    opacity: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    phase_progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
    bytes_written_callback: Optional[Callable[[int], None]] = None,
) -> List[Path]:
    """
    Optimized mesh export function.
    
    Key optimizations:
    1. Builds single combined mask for all enabled bins (only processes relevant pixels)
    2. Extracts single mesh (faster than per-bin extraction)
    3. Assigns bin colors directly (no unnecessary volume sampling)
    4. Applies voxel spacing correctly
    5. Optimized PLY writing with reduced precision
    
    Parameters
    ----------
    volume : np.ndarray
        Volume data, shape (Z, Y, X, 3) for RGB or (Z, Y, X) for grayscale
    bins : Sequence[IntensityBin]
        List of intensity bins to export
    config : MeshingConfig
        Meshing configuration with voxel spacing
    output_dir : Path
        Directory where meshes will be saved
    filename_prefix : str
        Prefix for output filenames
    export_mode : str
        "separate" - one file per bin, "combined" - all bins in one file
    opacity : Optional[float]
        Opacity value (0.0 to 1.0) to apply to all meshes
    progress_callback : Optional[Callable]
        Optional callback(current, total) for progress
    phase_progress_callback : Optional[Callable]
        Optional callback(phase, current, phase_total, overall_total) for phase-aware progress
    bytes_written_callback : Optional[Callable]
        Optional callback(bytes_written) to track actual bytes written
        
    Returns
    -------
    List[Path]
        Paths to generated PLY files
    """
    from . import volume as volmod
    
    # Convert to grayscale and color volumes
    if volume.ndim == 4 and volume.shape[-1] == 3:
        volume_gray = volmod.to_intensity_max(volume)
        volume_color = volume
    elif volume.ndim == 3:
        volume_gray = volume
        volume_color = None
    else:
        raise ValueError(f"Volume must be 3D or 4D, got shape {volume.shape}")
    
    enabled_bins = [b for b in bins if b.enabled]
    if not enabled_bins:
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if export_mode == "combined":
        # Combined mode: single mesh with all bins
        
        if phase_progress_callback:
            phase_progress_callback("Building combined mask", 0, 1, 0)
        
        # Build combined mask (only pixels in enabled bins)
        combined_mask = build_combined_bin_mask(volume_gray, bins)
        
        if np.count_nonzero(combined_mask) == 0:
            # Empty mask - no pixels in any enabled bin
            return []
        
        # Apply optional cleaning/filtering (only if enabled, to minimize loss)
        if config.enable_component_filtering:
            from . import meshing as meshing_mod
            combined_mask = meshing_mod.clean_bin_mask(combined_mask, config)
        
        if config.enable_smoothing:
            from . import meshing as meshing_mod
            combined_mask = meshing_mod.smooth_mask(combined_mask, config.smoothing_sigma)
        
        if phase_progress_callback:
            phase_progress_callback("Building combined mask", 1, 1, 0)
            phase_progress_callback("Extracting mesh", 0, 1, 1)
        
        # Extract optimized mesh
        def extract_progress_cb(current: int, total: int) -> None:
            if phase_progress_callback:
                # Scale to overall progress
                phase_progress_callback("Extracting mesh", current, total, 1 + current)
        
        verts, faces, colors = extract_optimized_mesh(
            volume_gray,
            combined_mask,
            bins,
            config,
            volume_color,
            progress_callback=extract_progress_cb,
        )
        
        if verts.size == 0 or faces.size == 0:
            return []
        
        if phase_progress_callback:
            phase_progress_callback("Extracting mesh", 1, 1, 2)
            phase_progress_callback("Saving file", 0, 1, 2)
        
        # Save combined mesh
        out_path = output_dir / f"{filename_prefix}_combined.ply"
        
        # Estimate file size for progress (use same calculation as save_mesh_ply_optimized)
        # Colors are always exported in optimized PLY, opacity is optional
        has_opacity_est = opacity is not None and opacity < 1.0
        # Sample-based estimation for accuracy
        sample_verts = min(100, verts.shape[0]) if verts.shape[0] > 0 else 0
        if sample_verts > 0:
            total_bytes = 0
            alpha_val = int(opacity * 255) if has_opacity_est else None
            for i in range(sample_verts):
                v = verts[i]
                c = colors[i]
                if has_opacity_est:
                    line = f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])} {alpha_val}\n"
                else:
                    line = f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
                total_bytes += len(line.encode('utf-8'))
            avg_vertex_size = total_bytes / sample_verts
        else:
            avg_vertex_size = 48 if has_opacity_est else 43
        
        # Header size calculation
        header_lines = [
            "ply\n", "format ascii 1.0\n", f"element vertex {verts.shape[0]}\n",
            "property float x\n", "property float y\n", "property float z\n",
            "property uchar red\n", "property uchar green\n", "property uchar blue\n",
        ]
        if has_opacity_est:
            header_lines.append("property uchar alpha\n")
        header_lines.extend([
            f"element face {faces.shape[0]}\n",
            "property list uchar int vertex_indices\n",
            "end_header\n",
        ])
        header_size = sum(len(line.encode('utf-8')) for line in header_lines)
        
        # Face line size (sample-based)
        sample_faces = min(100, faces.shape[0]) if faces.shape[0] > 0 else 0
        if sample_faces > 0:
            total_face_bytes = sum(len(f"3 {int(a)} {int(b)} {int(c)}\n".encode('utf-8')) for a, b, c in faces[:sample_faces])
            avg_face_size = total_face_bytes / sample_faces
        else:
            avg_face_size = 18
        
        estimated_file_size = int(header_size + (verts.shape[0] * avg_vertex_size) + (faces.shape[0] * avg_face_size))
        
        if phase_progress_callback:
            phase_progress_callback("Saving files", 0, estimated_file_size, 2)
        
        # Create progress callback for file saving
        def file_save_progress_cb(bytes_written: int, total_bytes: int) -> None:
            if bytes_written_callback:
                bytes_written_callback(bytes_written)
            if phase_progress_callback:
                phase_progress_callback("Saving files", bytes_written, total_bytes, 2 + bytes_written)
        
        save_mesh_ply_optimized(
            out_path,
            verts,
            faces,
            colors,
            opacity=opacity,
            progress_callback=file_save_progress_cb,
        )
        
        if phase_progress_callback:
            actual_size = out_path.stat().st_size
            phase_progress_callback("Saving files", actual_size, actual_size, 2 + actual_size)
        
        return [out_path]
    
    else:
        # Separate mode: one file per bin
        outputs = []
        total = len(enabled_bins)
        
        if phase_progress_callback:
            phase_progress_callback("Processing bins", 0, total, 0)
        
        for i, bin_ in enumerate(enabled_bins):
            if phase_progress_callback:
                phase_progress_callback("Processing bins", i, total, i)
            
            # Build mask for this bin only
            bin_mask = binsmod.build_bin_mask(volume_gray, bin_)
            
            if np.count_nonzero(bin_mask) == 0:
                continue
            
            # Apply optional cleaning/filtering
            if config.enable_component_filtering:
                from . import meshing as meshing_mod
                bin_mask = meshing_mod.clean_bin_mask(bin_mask, config)
            
            if config.enable_smoothing:
                from . import meshing as meshing_mod
                bin_mask = meshing_mod.smooth_mask(bin_mask, config.smoothing_sigma)
            
            if np.count_nonzero(bin_mask) == 0:
                continue
            
            # Extract mesh (only for this bin)
            # Create a temporary bins list with just this bin
            temp_bins = [bin_]
            verts, faces, colors = extract_optimized_mesh(
                volume_gray,
                bin_mask,
                temp_bins,
                config,
                volume_color,
                progress_callback=None,  # Don't need progress for individual bins
            )
            
            if verts.size == 0 or faces.size == 0:
                continue
            
            # Save file
            low_int = int(round(bin_.low))
            high_int = int(round(bin_.high))
            out_path = output_dir / f"{filename_prefix}_bin_{bin_.index:02d}_{low_int}_{high_int}.ply"
            
            def file_save_progress_cb(bytes_written: int, total_bytes: int) -> None:
                if bytes_written_callback:
                    bytes_written_callback(bytes_written)
            
            save_mesh_ply_optimized(
                out_path,
                verts,
                faces,
                colors,
                opacity=opacity,
                progress_callback=file_save_progress_cb,
            )
            
            outputs.append(out_path)
            
            if phase_progress_callback:
                phase_progress_callback("Processing bins", i + 1, total, i + 1)
        
        return outputs

