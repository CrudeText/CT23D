#!/usr/bin/env python3
"""
Simple script to visualize NRRD files using PyVista.

This script allows you to verify that NRRD files contain the expected data
by displaying the volume as a 3D mesh and showing volume statistics.

Usage:
    python scripts/view_nrrd.py <path_to_nrrd_file>
    
Or from project root:
    python -m scripts.view_nrrd <path_to_nrrd_file>
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import numpy as np
    import pyvista as pv
    from ct23d.core.volume import load_volume_nrrd
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install required packages:")
    print("  pip install pyvista pynrrd")
    sys.exit(1)


def visualize_nrrd(nrrd_path: Path) -> None:
    """Load and visualize an NRRD file."""
    print(f"Loading NRRD file: {nrrd_path}")
    print("-" * 60)
    
    # Load the volume
    try:
        volume = load_volume_nrrd(nrrd_path)
    except Exception as e:
        print(f"Error loading NRRD file: {e}")
        sys.exit(1)
    
    # Print volume information
    print(f"Shape (Z, Y, X): {volume.data.shape}")
    print(f"Data type: {volume.data.dtype}")
    print(f"Spacing (sx, sy, sz): {volume.spacing}")
    print(f"Origin (ox, oy, oz): {volume.origin}")
    print(f"Intensity kind: {volume.intensity_kind}")
    print(f"Min value: {volume.data.min()}")
    print(f"Max value: {volume.data.max()}")
    print(f"Mean value: {volume.data.mean():.2f}")
    print(f"Non-zero voxels: {np.count_nonzero(volume.data):,} / {volume.data.size:,}")
    
    if volume.provenance:
        print(f"\nProvenance:")
        for key, value in volume.provenance.items():
            if key == "intensity_bins" and isinstance(value, list):
                print(f"  {key}: {len(value)} bins")
                for i, bin_data in enumerate(value):
                    bin_info = f"    Bin {i}: low={bin_data.get('low')}, high={bin_data.get('high')}"
                    if bin_data.get('color'):
                        color = bin_data.get('color')
                        bin_info += f", color={color}"
                    print(bin_info)
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "-" * 60)
    print("Creating 3D visualization...")
    print("Controls:")
    print("  - Left click + drag: Rotate")
    print("  - Right click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - 'q' or close window: Quit")
    print("-" * 60)
    
    # Create a mesh using threshold of 200
    data_min = float(volume.data.min())
    data_max = float(volume.data.max())
    
    threshold = 200.0
    
    # Get color from bin that contains the threshold
    mesh_color = 'white'  # Default
    matching_bin_info = None
    if volume.provenance:
        intensity_bins = volume.provenance.get("intensity_bins")
        if intensity_bins and isinstance(intensity_bins, list) and len(intensity_bins) > 0:
            # Find bin that contains threshold and get its color
            for bin_data in intensity_bins:
                if isinstance(bin_data, dict):
                    low = bin_data.get("low")
                    high = bin_data.get("high")
                    if low is not None and high is not None and low <= threshold < high:
                        matching_bin_info = f"Bin [low={low}, high={high}]"
                        color = bin_data.get("color")
                        if color and isinstance(color, (list, tuple)) and len(color) >= 3:
                            # Color is in 0-1 range, PyVista accepts this
                            r, g, b = color[0], color[1], color[2]
                            # Ensure values are in 0-1 range
                            if r > 1.0 or g > 1.0 or b > 1.0:
                                r, g, b = r / 255.0, g / 255.0, b / 255.0
                            mesh_color = (r, g, b)
                            print(f"Using color from {matching_bin_info}: RGB({r:.3f}, {g:.3f}, {b:.3f})")
                            break
    
    print(f"\nUsing threshold: {threshold:.2f} (range: {data_min:.2f} to {data_max:.2f})")
    if matching_bin_info:
        print(f"Note: This threshold matches {matching_bin_info} - the entire mesh will be this bin's color.")
    print("Note: With a single threshold, the entire mesh has one uniform color.")
    if volume.provenance and volume.provenance.get("intensity_bins"):
        print("To see multiple bin colors, you would need to generate separate meshes for each bin.")
    print("Creating mesh... (this may take a moment for large volumes)")
    
    try:
        from skimage import measure
        
        # Create binary mask using threshold (same as original, but with smarter default)
        mask = (volume.data >= threshold).astype(np.float32)
        
        # Run marching cubes
        verts_vox, faces, _normals, _values = measure.marching_cubes(
            mask,
            level=0.5,
            spacing=(1.0, 1.0, 1.0),
        )
        
        if len(faces) == 0:
            print("Warning: No mesh was generated. Try adjusting the threshold.")
            print("You can modify the threshold in the script (around line 80).")
            return
        
        # Scale vertices by spacing
        sx, sy, sz = volume.spacing
        verts_physical = verts_vox.copy()
        verts_physical[:, 0] *= sz  # Z
        verts_physical[:, 1] *= sy  # Y
        verts_physical[:, 2] *= sx  # X
        
        # Convert to PyVista format (x, y, z)
        verts_pyvista = np.column_stack([
            verts_physical[:, 2],  # x
            verts_physical[:, 1],  # y
            verts_physical[:, 0],  # z
        ])
        
        # Convert faces to VTK format
        n_faces = len(faces)
        faces_vtk = np.empty((n_faces, 4), dtype=np.int32)
        faces_vtk[:, 0] = 3
        faces_vtk[:, 1:] = faces
        faces_vtk = faces_vtk.flatten()
        
        # Create mesh
        mesh = pv.PolyData(verts_pyvista, faces_vtk)
        mesh = mesh.compute_normals()
        
        print(f"Mesh created: {len(mesh.points):,} vertices, {mesh.n_cells:,} faces")
        
        # Create plotter and display
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color=mesh_color, smooth_shading=True)
        plotter.add_axes()
        plotter.background_color = 'black'
        plotter.reset_camera()
        plotter.show()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/view_nrrd.py <path_to_nrrd_file>")
        sys.exit(1)
    
    nrrd_path = Path(sys.argv[1])
    
    if not nrrd_path.exists():
        print(f"Error: File not found: {nrrd_path}")
        sys.exit(1)
    
    if not nrrd_path.suffix.lower() in ('.nrrd', '.nhdr'):
        print(f"Warning: File does not have .nrrd or .nhdr extension: {nrrd_path}")
    
    visualize_nrrd(nrrd_path)


if __name__ == "__main__":
    main()

