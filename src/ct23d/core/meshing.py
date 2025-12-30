from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening
from skimage import measure, morphology

from .models import MeshingConfig, IntensityBin
from . import volume as volmod
from . import bins as binsmod


# ---------------------------------------------------------------------------
# Global / per-bin mask building
# ---------------------------------------------------------------------------


def build_global_mask(
    volume_color: np.ndarray,
    cfg: MeshingConfig,
) -> np.ndarray:
    """
    Build a cleaned binary mask from a color or grayscale volume.

    Steps (inspired by the legacy CT_to_3D.py 'build_mask' logic):
      1. Convert to grayscale if needed.
      2. Gaussian smoothing to reduce noise.
      3. Intensity threshold to separate foreground from background.
      4. Morphological closing and opening to remove specks and holes.
      5. Remove small connected components (< min_component_size).

    Parameters
    ----------
    volume_color:
        4D volume [Z, Y, X, 3] or 3D [Z, Y, X], typically uint8.
    cfg:
        MeshingConfig providing thresholds and parameters.

    Returns
    -------
    np.ndarray
        Boolean mask [Z, Y, X] where True means foreground.
    """
    if volume_color.ndim == 4 and volume_color.shape[-1] == 3:
        # Use max intensity instead of luminance to preserve information
        # from colored overlays in medical imaging
        gray = volmod.to_intensity_max(volume_color)
    elif volume_color.ndim == 3:
        # Preserve dtype (don't cast uint16 to uint8)
        gray = volume_color
    else:
        raise ValueError(
            f"build_global_mask expects [Z, Y, X] or [Z, Y, X, 3], got {volume_color.shape}"
        )

    # Smooth to reduce noise
    smooth = gaussian_filter(gray.astype(np.float32), sigma=cfg.smoothing_sigma)

    # Threshold: foreground = intensities above non_black_threshold
    # Scale threshold based on dtype: uint16 data needs higher threshold
    if gray.dtype == np.uint16:
        # Scale uint8 threshold (15) to uint16 range (0-65535)
        # 15/255 ≈ 0.059, so for uint16: 0.059 * 65535 ≈ 3848
        threshold = float(cfg.non_black_threshold) * (65535.0 / 255.0)
    else:
        threshold = float(cfg.non_black_threshold)
    mask = smooth > threshold

    # Morphological cleanup
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)

    # Remove small connected components
    mask = morphology.remove_small_objects(mask, min_size=cfg.min_component_size)

    if np.count_nonzero(mask) == 0:
        dtype_info = f" (dtype: {gray.dtype}, threshold used: {threshold:.1f})"
        raise RuntimeError(
            f"Global mask is empty after cleaning.{dtype_info}\n"
            "Try lowering 'non_black_threshold' or 'min_component_size'. "
            "For uint16 DICOM data, the threshold is automatically scaled from the uint8 default."
        )

    return mask


def clean_bin_mask(
    bin_mask: np.ndarray,
    cfg: MeshingConfig,
) -> np.ndarray:
    """
    Clean a per-bin mask via connected-component analysis.
    
    This function removes small isolated regions (noise) from the mask.
    It should only be called when enable_component_filtering is True.
    
    If cfg.adaptive_component_filtering is True:
      - Find the largest connected component size L.
      - Compute an effective min size:
           eff_min_size = max(cfg.min_component_size, int(0.01 * L))
      - Remove components smaller than eff_min_size.

    Otherwise:
      - Remove components smaller than cfg.min_component_size.
    
    Note: This function is only called when cfg.enable_component_filtering is True.
    If you're seeing surface loss with component filtering disabled, check:
    1. That enable_component_filtering is actually False in your config
    2. That Gaussian smoothing isn't removing the surfaces (see smooth_mask docstring)
    """
    if bin_mask.ndim != 3:
        raise ValueError(
            f"clean_bin_mask expects a 3D mask [Z, Y, X], got {bin_mask.shape}"
        )

    labels = measure.label(bin_mask, connectivity=1)
    if labels.max() == 0:
        # no components; return all False
        return np.zeros_like(bin_mask, dtype=bool)

    # Compute component sizes
    props = measure.regionprops(labels)
    sizes = np.array([p.area for p in props], dtype=np.int64)
    largest = int(sizes.max())

    if cfg.adaptive_component_filtering:
        eff_min_size = max(cfg.min_component_size, int(0.01 * largest))
    else:
        eff_min_size = cfg.min_component_size

    cleaned = morphology.remove_small_objects(labels, min_size=eff_min_size) > 0
    return cleaned


def smooth_mask(
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Smooth a binary mask using a Gaussian filter and re-threshold at 0.5.

    This is mainly to make the isosurface from marching cubes smoother.
    
    WARNING: Gaussian smoothing can remove small objects and thin connections!
    - Small isolated objects may disappear if their blurred values drop below 0.5
    - Thin connections between larger objects can be broken
    - Use with caution if you need to preserve fine details or small structures
    
    If you're experiencing surface loss, try:
    1. Disabling Gaussian smoothing (set enable_smoothing=False)
    2. Reducing the sigma value (smaller sigma = less blur = less surface loss)
    3. Disabling component filtering if small objects are being removed

    Parameters
    ----------
    mask:
        Boolean mask [Z, Y, X].
    sigma:
        Standard deviation for Gaussian smoothing (in voxels).
        Larger values = more blur = smoother surfaces but more surface loss.

    Returns
    -------
    np.ndarray
        Boolean mask [Z, Y, X] after smoothing and thresholding.
    """
    if sigma <= 0.0:
        return mask

    smoothed = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return smoothed > 0.5


# ---------------------------------------------------------------------------
# Mesh extraction & saving
# ---------------------------------------------------------------------------


def extract_mesh(
    mask: np.ndarray,
    volume_color: Optional[np.ndarray],
    cfg: MeshingConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract a surface mesh from a binary mask using marching cubes.

    Strategy:
      - Use marching cubes on the binary mask (float, level=0.5).
      - Use voxel coordinates (spacing = 1,1,1) to sample colors if a color
        volume is provided.
      - Scale vertices by cfg.spacing to get real-world coordinates.

    Parameters
    ----------
    mask:
        Boolean mask [Z, Y, X].
    volume_color:
        Optional 4D [Z, Y, X, 3] uint8 volume to sample vertex colors from.
        If None, no colors are returned.
    cfg:
        MeshingConfig providing voxel spacing.

    Returns
    -------
    (vertices, faces, colors)
        vertices:
            Array [N, 3] of vertex coordinates in physical units.
        faces:
            Array [M, 3] of int indices.
        colors:
            Optional array [N, 3] of uint8 RGB colors, or None.
    """
    if mask.ndim != 3:
        raise ValueError(
            f"extract_mesh expects a 3D mask [Z, Y, X], got {mask.shape}"
        )

    # marching_cubes expects values, not booleans
    vol_f = mask.astype(np.float32)
    
    if progress_callback:
        progress_callback(0, 3)  # Step 1: Starting marching cubes

    # Run marching cubes in voxel coordinates (spacing=1,1,1)
    verts_vox, faces, _normals, _values = measure.marching_cubes(
        vol_f,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
    )
    
    if progress_callback:
        progress_callback(1, 3)  # Step 2: Marching cubes complete

    # Sample colors if volume_color is provided (can be slow for many vertices)
    colors = None
    if volume_color is not None:
        if volume_color.ndim != 4 or volume_color.shape[-1] != 3:
            raise ValueError(
                f"volume_color must be [Z, Y, X, 3], got {volume_color.shape}"
            )

        zmax, ymax, xmax, _ = volume_color.shape
        # verts_vox are in (z, y, x)
        # Sample colors in batches for progress updates if there are many vertices
        n_verts = verts_vox.shape[0]
        if progress_callback and n_verts > 10000:
            # For large meshes, sample colors in batches and update progress
            batch_size = max(5000, n_verts // 20)  # Update every 5% or every 5000 vertices
            colors_list = []
            for batch_start in range(0, n_verts, batch_size):
                batch_end = min(batch_start + batch_size, n_verts)
                batch_verts = verts_vox[batch_start:batch_end]
                z_idx = np.clip(np.round(batch_verts[:, 0]).astype(int), 0, zmax - 1)
                y_idx = np.clip(np.round(batch_verts[:, 1]).astype(int), 0, ymax - 1)
                x_idx = np.clip(np.round(batch_verts[:, 2]).astype(int), 0, xmax - 1)
                batch_colors = volume_color[z_idx, y_idx, x_idx].astype(np.uint8)
                colors_list.append(batch_colors)
                # Update progress (step 2 + progress through color sampling)
                if progress_callback:
                    color_progress = batch_end / n_verts
                    progress_callback(2 + int(color_progress), 3)
            colors = np.vstack(colors_list)
        else:
            z_idx = np.clip(np.round(verts_vox[:, 0]).astype(int), 0, zmax - 1)
            y_idx = np.clip(np.round(verts_vox[:, 1]).astype(int), 0, ymax - 1)
            x_idx = np.clip(np.round(verts_vox[:, 2]).astype(int), 0, xmax - 1)
            colors = volume_color[z_idx, y_idx, x_idx].astype(np.uint8)
    
    if progress_callback:
        progress_callback(3, 3)  # Step 3: Colors sampled

    # Scale to physical units using cfg.spacing = (z, y, x)
    sz, sy, sx = cfg.spacing
    scale = np.array([sz, sy, sx], dtype=np.float32)
    verts_phys = verts_vox * scale[None, :]

    return verts_phys.astype(np.float32), faces.astype(np.int32), colors


def save_mesh_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None,
    opacity: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Save a mesh as an ASCII PLY file with optional per-vertex RGB colors and opacity.

    Parameters
    ----------
    path:
        Destination file path.
    vertices:
        Array [N, 3] of float vertices.
    faces:
        Array [M, 3] of int vertex indices.
    colors:
        Optional array [N, 3] of uint8 RGB colors.
        If None, colors are omitted from the PLY.
    opacity:
        Optional opacity value (0.0 to 1.0) applied to all vertices.
        If provided, an alpha channel is added to the PLY file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]

    has_colors = colors is not None
    has_opacity = opacity is not None
    
    if has_colors:
        if colors.shape != (n_verts, 3):
            raise ValueError(
                f"colors must have shape (N, 3) matching vertices, "
                f"got {colors.shape} for {n_verts} vertices."
            )
    
    if opacity is not None:
        if not (0.0 <= opacity <= 1.0):
            raise ValueError(f"opacity must be between 0.0 and 1.0, got {opacity}")
        alpha_value = int(opacity * 255)

    # Estimate file size for progress tracking
    # Header: ~300 bytes
    # Vertex line: ~40-50 bytes (with colors/opacity), ~30 bytes (without)
    # Face line: ~15-20 bytes
    header_size = 300
    if has_colors and has_opacity:
        vertex_line_size = 50
    elif has_colors or has_opacity:
        vertex_line_size = 45
    else:
        vertex_line_size = 30
    face_line_size = 18
    
    estimated_total_size = header_size + (n_verts * vertex_line_size) + (n_faces * face_line_size)
    
    if progress_callback:
        progress_callback(0, estimated_total_size)  # Start: header
    
    with path.open("w", encoding="utf-8") as f:
        bytes_written = 0
        
        # Header
        header_lines = [
            "ply\n",
            "format ascii 1.0\n",
            f"element vertex {n_verts}\n",
            "property float x\n",
            "property float y\n",
            "property float z\n",
        ]
        if has_colors:
            header_lines.extend([
                "property uchar red\n",
                "property uchar green\n",
                "property uchar blue\n",
            ])
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
            # The callback will check for cancellation and raise InterruptedError if needed
            progress_callback(bytes_written, estimated_total_size)
        
        # Vertices - write in batches and report progress
        # Use smaller batch size for more frequent progress updates (prevents UI freezing)
        batch_size = max(500, n_verts // 200)  # Update progress every 0.5% or every 500 vertices
        if has_colors and has_opacity:
            for i, ((x, y, z), (r, g, b)) in enumerate(zip(vertices, colors)):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {alpha_value}\n")
                bytes_written += 50  # Approximate
                if progress_callback and (i + 1) % batch_size == 0:
                    # The callback will check for cancellation and raise InterruptedError if needed
                    progress_callback(bytes_written, estimated_total_size)
        elif has_colors:
            for i, ((x, y, z), (r, g, b)) in enumerate(zip(vertices, colors)):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
                bytes_written += 45  # Approximate
                if progress_callback and (i + 1) % batch_size == 0:
                    # The callback will check for cancellation and raise InterruptedError if needed
                    progress_callback(bytes_written, estimated_total_size)
        elif has_opacity:
            for i, (x, y, z) in enumerate(vertices):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {alpha_value}\n")
                bytes_written += 40  # Approximate
                if progress_callback and (i + 1) % batch_size == 0:
                    # The callback will check for cancellation and raise InterruptedError if needed
                    progress_callback(bytes_written, estimated_total_size)
        else:
            for i, (x, y, z) in enumerate(vertices):
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
                bytes_written += 30  # Approximate
                if progress_callback and (i + 1) % batch_size == 0:
                    # The callback will check for cancellation and raise InterruptedError if needed
                    progress_callback(bytes_written, estimated_total_size)

        # Faces - write in batches and report progress
        # Use smaller batch size for more frequent progress updates (prevents UI freezing)
        face_batch_size = max(500, n_faces // 200)  # Update progress every 0.5% or every 500 faces
        for i, (a, b, c) in enumerate(faces):
            f.write(f"3 {int(a)} {int(b)} {int(c)}\n")
            bytes_written += 18  # Approximate
            if progress_callback and (i + 1) % face_batch_size == 0:
                # The callback will check for cancellation and raise InterruptedError if needed
                progress_callback(bytes_written, estimated_total_size)
        
        if progress_callback:
            # The callback will check for cancellation and raise InterruptedError if needed
            progress_callback(estimated_total_size, estimated_total_size)  # Complete
        
        # Explicitly flush and sync the file to ensure it's written and released on Windows
        f.flush()
        import os
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            # Some file systems don't support fsync, or file might not have fileno
            pass
        # The 'with' statement will close the file here, releasing the handle


def save_mesh_stl(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    binary: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Save a mesh as an STL file (binary or ASCII format).
    
    STL format does not support colors or opacity - it only stores geometry.
    Binary format is more compact and faster to write/read.
    
    Parameters
    ----------
    path:
        Destination file path.
    vertices:
        Array [N, 3] of float vertices.
    faces:
        Array [M, 3] of int vertex indices.
    binary:
        If True, save as binary STL (default, faster and smaller).
        If False, save as ASCII STL (human-readable).
    progress_callback:
        Optional callback function(bytes_written, total_bytes) for progress updates.
    """
    import struct
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]
    
    # Estimate file size for progress tracking
    if binary:
        # Binary STL: 80 byte header + 4 bytes (face count) + 50 bytes per face
        estimated_total_size = 80 + 4 + (n_faces * 50)
    else:
        # ASCII STL: header + face data
        estimated_total_size = 100 + (n_faces * 100)  # Rough estimate
    
    if progress_callback:
        progress_callback(0, estimated_total_size)
    
    if binary:
        # Binary STL format
        with path.open("wb") as f:
            # 80-byte header (usually contains description, but we'll leave it empty)
            header = b"CT23D STL Export" + b"\x00" * (80 - 16)
            f.write(header)
            bytes_written = len(header)
            
            if progress_callback:
                progress_callback(bytes_written, estimated_total_size)
            
            # Number of triangles (4 bytes, unsigned int)
            f.write(struct.pack("<I", n_faces))
            bytes_written += 4
            
            if progress_callback:
                progress_callback(bytes_written, estimated_total_size)
            
            # For each triangle:
            # - Normal vector (3 floats, 12 bytes) - we'll compute it
            # - Vertex 1 (3 floats, 12 bytes)
            # - Vertex 2 (3 floats, 12 bytes)
            # - Vertex 3 (3 floats, 12 bytes)
            # - Attribute byte count (2 bytes, usually 0)
            # Use batches for progress updates
            batch_size = max(500, n_faces // 200)  # Update every 0.5% or every 500 faces
            for i, face in enumerate(faces):
                # Get triangle vertices
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                
                # Compute normal vector (cross product of two edges)
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                else:
                    normal = np.array([0.0, 0.0, 1.0])  # Default normal if degenerate
                
                # Write normal (3 floats)
                f.write(struct.pack("<fff", float(normal[0]), float(normal[1]), float(normal[2])))
                
                # Write vertices (3 floats each)
                f.write(struct.pack("<fff", float(v0[0]), float(v0[1]), float(v0[2])))
                f.write(struct.pack("<fff", float(v1[0]), float(v1[1]), float(v1[2])))
                f.write(struct.pack("<fff", float(v2[0]), float(v2[1]), float(v2[2])))
                
                # Attribute byte count (usually 0)
                f.write(struct.pack("<H", 0))
                
                bytes_written += 50  # Each face is 50 bytes
                
                # Update progress periodically
                if progress_callback and (i + 1) % batch_size == 0:
                    progress_callback(bytes_written, estimated_total_size)
            
            if progress_callback:
                progress_callback(estimated_total_size, estimated_total_size)
            
            # Explicitly flush and sync the file to ensure it's written and released on Windows
            f.flush()
            import os
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                # Some file systems don't support fsync, or file might not have fileno
                pass
    else:
        # ASCII STL format
        with path.open("w", encoding="utf-8") as f:
            f.write("solid CT23D_STL_Export\n")
            bytes_written = len("solid CT23D_STL_Export\n")
            
            if progress_callback:
                progress_callback(bytes_written, estimated_total_size)
            
            batch_size = max(500, n_faces // 200)  # Update every 0.5% or every 500 faces
            for i, face in enumerate(faces):
                # Get triangle vertices
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                
                # Compute normal vector
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                else:
                    normal = np.array([0.0, 0.0, 1.0])
                
                # Write facet
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
                
                # Estimate bytes written (rough approximation)
                bytes_written += 100  # Approximate per face
                
                # Update progress periodically
                if progress_callback and (i + 1) % batch_size == 0:
                    progress_callback(bytes_written, estimated_total_size)
            
            f.write("endsolid CT23D_STL_Export\n")
            bytes_written += len("endsolid CT23D_STL_Export\n")
            
            if progress_callback:
                progress_callback(estimated_total_size, estimated_total_size)
            
            # Explicitly flush and sync the file to ensure it's written and released on Windows
            f.flush()
            import os
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                # Some file systems don't support fsync, or file might not have fileno
                pass


# ---------------------------------------------------------------------------
# High-level meshing over bins
# ---------------------------------------------------------------------------


def generate_meshes_for_bins(
    volume_gray: np.ndarray,
    bins: Sequence[IntensityBin],
    cfg: MeshingConfig,
    *,
    volume_color: Optional[np.ndarray] = None,
    global_mask: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    phase_progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
    export_format: str = "PLY",
    stl_binary: bool = True,  # For STL format: True=binary, False=ASCII
) -> List[Path]:
    """
    Generate meshes for a list of intensity bins.

    Workflow:
      1. For each enabled bin:
         - Build a bin mask from `volume_gray`.
         - Apply global_mask (if provided).
         - Clean small components (adaptive if enabled).
         - Smooth the mask.
         - Run marching cubes to get a mesh.
         - Sample per-vertex colors if volume_color is given.
         - Save as PLY in cfg.output_dir.

    Parameters
    ----------
    volume_gray:
        3D grayscale volume [Z, Y, X], e.g. from volume.to_grayscale().
    bins:
        Sequence of IntensityBin objects (typically from bins.select_bins()).
    cfg:
        MeshingConfig containing spacing, thresholds, output directory, etc.
    volume_color:
        Optional [Z, Y, X, 3] RGB volume for sampling per-vertex colors.
        If None, meshes are saved without colors.
    global_mask:
        Optional boolean [Z, Y, X] mask to restrict meshing to a region
        (e.g. from build_global_mask()).

    Returns
    -------
    list of Path
        Paths to the generated PLY files.
    """
    if volume_gray.ndim != 3:
        raise ValueError(
            f"generate_meshes_for_bins expects a 3D grayscale volume [Z, Y, X], "
            f"got shape {volume_gray.shape}"
        )

    if cfg.output_dir is None:
        raise ValueError(
            "MeshingConfig.output_dir must be set before generating meshes."
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    
    enabled_bins = [b for b in bins if b.enabled]
    total = len(enabled_bins)
    
    # Phase-aware progress: Building masks, Extracting meshes, Saving files
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
        # Check for cancellation before each bin
        if phase_progress_callback:
            # Check if we should stop - this will be checked in the callback
            pass
        
        if progress_callback:
            progress_callback(i, total)
        
        if phase_progress_callback:
            phase_current = i + 1
            overall_current = i + 1
            # The callback will check for cancellation and raise InterruptedError if needed
            phase_progress_callback(phase, phase_current, phase_total, overall_total)

        # 1) Build mask for this bin
        bin_mask = binsmod.build_bin_mask(volume_gray, bin_)

        # Skip empty if configured
        if cfg.skip_empty_bins and np.count_nonzero(bin_mask) == 0:
            bin_masks.append(None)
            continue

        # 2) Apply global mask if provided
        if global_mask is not None:
            if global_mask.shape != bin_mask.shape:
                raise ValueError(
                    "global_mask shape does not match volume shape: "
                    f"{global_mask.shape} vs {bin_mask.shape}"
                )
            bin_mask &= global_mask

        # After applying global mask, maybe it's empty
        if cfg.skip_empty_bins and np.count_nonzero(bin_mask) == 0:
            bin_masks.append(None)
            continue

        # 3) Clean small components (if enabled)
        if cfg.enable_component_filtering:
            bin_mask_clean = clean_bin_mask(bin_mask, cfg)
        else:
            bin_mask_clean = bin_mask

        if cfg.skip_empty_bins and np.count_nonzero(bin_mask_clean) == 0:
            bin_masks.append(None)
            continue

        # 4) Smooth mask (if enabled)
        if cfg.enable_smoothing:
            bin_mask_smooth = smooth_mask(bin_mask_clean, cfg.smoothing_sigma)
        else:
            bin_mask_smooth = bin_mask_clean

        if cfg.skip_empty_bins and np.count_nonzero(bin_mask_smooth) == 0:
            bin_masks.append(None)
            continue
        
        bin_masks.append(bin_mask_smooth)
    
    # Mark building masks phase as complete
    if phase_progress_callback:
        phase_progress_callback(phase, phase_total, phase_total, overall_total)
    
    # Phase 2: Extract all meshes
    if phase_progress_callback:
        phase = "Extracting meshes"
        phase_current = 0
        phase_total = total
        overall_current = total
        phase_progress_callback(phase, 0, phase_total, overall_total)
    
    # Store extracted meshes for saving phase
    extracted_meshes = []
    for i, (bin_, bin_mask_smooth) in enumerate(zip(enabled_bins, bin_masks)):
        if bin_mask_smooth is None:
            extracted_meshes.append(None)
            continue
        
        # 5) Extract mesh (with progress callback to prevent freezing)
        # Update progress during extraction
        def extract_progress_cb(current: int, total: int) -> None:
            # Report progress during mesh extraction
            if phase_progress_callback:
                # Scale extraction progress to overall progress
                # We're in phase 2 (extracting), starting at overall_current = total
                extraction_progress = current / max(total, 1)
                phase_current_scaled = i + extraction_progress
                overall_current_scaled = total + phase_current_scaled
                phase_progress_callback(phase, int(phase_current_scaled), phase_total, overall_total)
        
        verts, faces, colors = extract_mesh(bin_mask_smooth, volume_color, cfg, progress_callback=extract_progress_cb)
        
        if phase_progress_callback:
            phase_current = i + 1
            overall_current = total + i + 1
            # The callback will check for cancellation and raise InterruptedError if needed
            phase_progress_callback(phase, phase_current, phase_total, overall_total)

        if verts.size == 0 or faces.size == 0:
            # No geometry produced (e.g. extremely small bin) – skip
            extracted_meshes.append(None)
            continue

        # 6) Apply bin color if specified (override volume colors)
        if bin_.color is not None:
            # Use bin color for all vertices
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
        
        extracted_meshes.append((bin_, verts, faces, colors))
    
    # Mark extracting meshes phase as complete
    if phase_progress_callback:
        phase_progress_callback(phase, phase_total, phase_total, overall_total)
    
    # Phase 3: Save all files
    if phase_progress_callback:
        phase = "Saving files"
        phase_current = 0
        phase_total = len([m for m in extracted_meshes if m is not None])
        overall_current = total * 2
        phase_progress_callback(phase, 0, phase_total, overall_total)
    
    saved_count = 0
    for mesh_data in extracted_meshes:
        if mesh_data is None:
            continue
        
        bin_, verts, faces, colors = mesh_data
        
        # 7) Save mesh (round intensity values to integers)
        low_int = int(round(bin_.low))
        high_int = int(round(bin_.high))
        # Get file extension based on export format
        format_upper = export_format.upper()
        if format_upper == "STL":
            ext = "stl"
        else:  # Default to PLY
            ext = "ply"
        fname = (
            f"{cfg.output_prefix}_bin_{bin_.index:02d}"
            f"_{low_int}_{high_int}.{ext}"
        )
        out_path = output_dir / fname
        # Get opacity from config if available, otherwise None
        opacity = getattr(cfg, 'opacity', None)
        
        # Create progress callback for file saving based on file size
        def file_save_progress_cb(bytes_written: int, total_bytes: int) -> None:
            if phase_progress_callback:
                # Calculate progress within this file (0.0 to 1.0)
                file_progress = bytes_written / max(total_bytes, 1)
                # Calculate overall progress: previous files + this file's progress
                # Each file contributes 1 to phase_total, so we need to scale file_progress
                files_completed = saved_count
                files_total = phase_total
                if files_total > 0:
                    overall_file_progress = (files_completed + file_progress) / files_total
                    phase_current = int(overall_file_progress * files_total)
                    overall_current = total * 2 + phase_current
                    phase_progress_callback(phase, phase_current, phase_total, overall_total)
        
        if format_upper == "STL":
            save_mesh_stl(out_path, verts, faces, binary=stl_binary, progress_callback=file_save_progress_cb)
        else:  # PLY format
            save_mesh_ply(out_path, verts, faces, colors, opacity=opacity, progress_callback=file_save_progress_cb)
        outputs.append(out_path)
        saved_count += 1
        
        if phase_progress_callback:
            phase_current = saved_count
            overall_current = total * 2 + saved_count
            phase_progress_callback(phase, phase_current, phase_total, overall_total)
    
    if progress_callback:
        progress_callback(total, total)
    
    # Mark saving phase as complete
    if phase_progress_callback:
        phase_progress_callback(phase, phase_total, phase_total, overall_total)

    return outputs


def generate_meshes_from_volume(
    volume: np.ndarray,
    config: MeshingConfig,
    output_dir: Path,
    filename_prefix: str = "ct_layer",
    bins: Optional[Sequence[IntensityBin]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    phase_progress_callback: Optional[Callable[[str, int, int, int], None]] = None,
) -> List[Path]:
    """
    High-level function to generate meshes from a volume with custom bins.
    
    This is a convenience wrapper that:
    1. Converts the volume to grayscale
    2. Calls generate_meshes_for_bins with the provided bins
    
    Parameters
    ----------
    volume : np.ndarray
        Volume data, shape (Z, Y, X, 3) for RGB or (Z, Y, X) for grayscale
    config : MeshingConfig
        Meshing configuration
    output_dir : Path
        Directory where meshes will be saved
    filename_prefix : str
        Prefix for output filenames
    bins : Optional[Sequence[IntensityBin]]
        Custom bins to use. If None, generates uniform bins from config.
    progress_callback : Optional[Callable[[int, int], None]]
        Optional callback function(current, total) for progress updates
        
    Returns
    -------
    List[Path]
        Paths to generated PLY files
    """
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
    
    # Get bins - use provided bins or generate uniform bins
    from . import bins as binsmod
    
    if bins is None:
        bins = binsmod.generate_uniform_bins(config)
    
    # Set output_dir in config
    config.output_dir = output_dir
    config.output_prefix = filename_prefix
    
    # Use the existing generate_meshes_for_bins function
    return generate_meshes_for_bins(
        volume_gray=volume_gray,
        bins=bins,
        cfg=config,
        volume_color=volume_color,
        progress_callback=progress_callback,
        phase_progress_callback=phase_progress_callback,
    )