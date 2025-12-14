from __future__ import annotations

from typing import Optional, List, Callable

import numpy as np
try:
    import pyqtgraph.opengl as gl
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from ct23d.core.models import IntensityBin


class Histogram3DGLView(QFrame):
    """
    True 3D histogram view using OpenGL:
    - X axis: Slice number
    - Y axis: Intensity
    - Z axis: Pixel count (height)
    
    Displays as a 3D surface plot that can be rotated and zoomed.
    """

    def __init__(self, parent: Optional[QFrame] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_OPENGL:
            # Show error message if OpenGL is not available
            error_label = QLabel(
                "3D visualization requires OpenGL support.\n"
                "Please install PyOpenGL: pip install PyOpenGL PyOpenGL_accelerate"
            )
            error_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_label)
            self._gl_view = None
            return

        # Create OpenGL view widget
        self._gl_view = gl.GLViewWidget(self)
        layout.addWidget(self._gl_view)

        # Set up the view
        self._gl_view.setCameraPosition(distance=200, elevation=30, azimuth=45)
        self._gl_view.setBackgroundColor('k')  # Black background

        # Store data
        self._count_data: Optional[np.ndarray] = None
        self._vmin: float = 0.0
        self._vmax: float = 255.0
        self._num_slices: int = 0
        self._n_bins: int = 256
        self._surface_item: Optional[gl.GLSurfacePlotItem] = None
        self._bin_planes: List[gl.GLGridItem] = []
        self._base_colors: Optional[np.ndarray] = None
        self._z_normalized: Optional[np.ndarray] = None
        self._axes_items: List = []  # Store axis line items

    def clear(self) -> None:
        """Clear the 3D histogram plot."""
        if self._gl_view is None:
            return
        
        # Remove axes items first
        for axis_item in self._axes_items:
            self._gl_view.removeItem(axis_item)
        self._axes_items.clear()
        
        self._gl_view.clear()
        self._count_data = None
        self._surface_item = None
        self._bin_planes.clear()

    def set_histogram_3d(
        self,
        volume: np.ndarray,
        n_bins: int = 256,
        value_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        Compute and display a 3D histogram from a volume.
        
        Parameters
        ----------
        volume : np.ndarray
            Volume data, shape (Z, Y, X, 3) for RGB or (Z, Y, X) for grayscale
        n_bins : int
            Number of intensity bins
        value_range : Optional[tuple[float, float]]
            Intensity range (vmin, vmax). If None, computed from data.
        """
        if self._gl_view is None:
            return
            
        if volume is None or volume.size == 0:
            self.clear()
            return

        arr = np.asarray(volume)
        
        # Convert RGB to grayscale if needed
        if arr.ndim == 4 and arr.shape[-1] == 3:
            grayscale = arr.mean(axis=-1)  # (Z, Y, X)
        else:
            grayscale = arr  # (Z, Y, X)
        
        num_slices = grayscale.shape[0]
        self._num_slices = num_slices
        self._n_bins = n_bins
        
        # Determine intensity range
        if value_range is None:
            # Exclude zeros to avoid huge spike
            non_zero = grayscale[grayscale > 0]
            if non_zero.size > 0:
                vmin = float(non_zero.min())
                vmax = float(non_zero.max())
            else:
                vmin = float(grayscale.min())
                vmax = float(grayscale.max())
        else:
            vmin, vmax = value_range
        
        self._vmin = vmin
        self._vmax = vmax
        
        # Compute histogram for each slice
        slice_histograms = []
        intensity_bin_edges = np.linspace(vmin, vmax, n_bins + 1)
        intensity_bin_centers = (intensity_bin_edges[:-1] + intensity_bin_edges[1:]) / 2
        
        # Process slices
        from PySide6.QtWidgets import QApplication
        batch_size = max(50, num_slices // 10)
        
        for z_idx in range(num_slices):
            slice_data = grayscale[z_idx]
            slice_data_clipped = np.clip(slice_data, vmin, vmax)
            hist, _ = np.histogram(slice_data_clipped, bins=intensity_bin_edges)
            slice_histograms.append(hist)
            
            if (z_idx + 1) % batch_size == 0:
                QApplication.processEvents()
        
        histograms_2d = np.array(slice_histograms)  # (num_slices, n_bins)
        
        # Store data
        self._count_data = histograms_2d
        
        # Clear previous plot
        self._gl_view.clear()
        self._surface_item = None
        self._bin_planes.clear()
        
        # Prepare data for 3D surface plot
        # GLSurfacePlotItem expects:
        # - x: 1D array of length M (slice indices)
        # - y: 1D array of length N (intensity bin centers)
        # - z: 2D array of shape (M, N) (pixel counts)
        
        x = np.arange(num_slices, dtype=float)  # Slice numbers (1D array)
        y = intensity_bin_centers.astype(float)  # Intensity values (1D array)
        
        # Z is the pixel counts, shape (num_slices, n_bins)
        Z = histograms_2d.astype(float)  # (num_slices, n_bins)
        
        # Normalize Z for better visualization (log scale to handle large variations)
        Z_log = np.log1p(Z)  # log(1+x) to handle zeros
        
        # Create color map based on Z values (height)
        # Use a colormap that goes from dark blue (low) to bright yellow (high)
        Z_normalized = (Z_log - Z_log.min()) / (Z_log.max() - Z_log.min() + 1e-10)
        
        # Create color array (R, G, B, A) for each point
        # Start with default colormap, will be tinted by bins in update_bins
        colors = np.zeros((num_slices, n_bins, 4), dtype=float)
        colors[:, :, 0] = Z_normalized  # Red component
        colors[:, :, 1] = 0.5 + 0.5 * Z_normalized  # Green component
        colors[:, :, 2] = 1.0 - 0.5 * Z_normalized  # Blue component
        colors[:, :, 3] = 0.8  # Alpha
        
        # Store for later bin tinting
        self._base_colors = colors.copy()
        self._z_normalized = Z_normalized
        
        # Scale Z to reasonable height (normalize to 0-50 range for visibility)
        Z_scaled = Z_log * (50.0 / (Z_log.max() + 1e-10))
        
        # Verify shapes match expectations
        assert len(x) == num_slices, f"x length {len(x)} != num_slices {num_slices}"
        assert len(y) == n_bins, f"y length {len(y)} != n_bins {n_bins}"
        assert Z_scaled.shape == (num_slices, n_bins), f"Z shape {Z_scaled.shape} != ({num_slices}, {n_bins})"
        
        # Create 3D surface plot
        # GLSurfacePlotItem expects 1D x and y arrays, and 2D z array
        self._surface_item = gl.GLSurfacePlotItem(
            x=x,  # 1D array of slice indices
            y=y,  # 1D array of intensity bin centers
            z=Z_scaled,  # 2D array of shape (len(x), len(y))
            colors=colors,
            shader='balloon',  # Shader for smooth appearance
            computeNormals=True,
        )
        self._gl_view.addItem(self._surface_item)
        
        # Add axes using GLLinePlotItem for visibility
        # X-axis (slice number) - horizontal line at origin
        z_max = np.max(Z_scaled)
        
        # X-axis line (red) - along X direction at Y=vmin, Z=0
        # Create more points for smoother line
        x_points = np.array([[i, vmin, 0] for i in range(0, num_slices + 1, max(1, num_slices // 20))])
        x_axis_line = gl.GLLinePlotItem(
            pos=x_points,
            color=(1.0, 0.0, 0.0, 1.0),  # Red, fully opaque
            width=3  # Thicker for visibility
        )
        self._gl_view.addItem(x_axis_line)
        self._axes_items.append(x_axis_line)
        
        # Y-axis line (green) - along Y direction at X=0, Z=0
        y_points = np.array([[0, vmin + i * (vmax - vmin) / 20, 0] for i in range(21)])
        y_axis_line = gl.GLLinePlotItem(
            pos=y_points,
            color=(0.0, 1.0, 0.0, 1.0),  # Green, fully opaque
            width=3  # Thicker for visibility
        )
        self._gl_view.addItem(y_axis_line)
        self._axes_items.append(y_axis_line)
        
        # Z-axis line (blue) - along Z direction at X=0, Y=vmin
        z_points = np.array([[0, vmin, i * z_max / 20] for i in range(21)])
        z_axis_line = gl.GLLinePlotItem(
            pos=z_points,
            color=(0.0, 0.0, 1.0, 1.0),  # Blue, fully opaque
            width=3  # Thicker for visibility
        )
        self._gl_view.addItem(z_axis_line)
        self._axes_items.append(z_axis_line)
        
        # Auto-fit the view
        self._gl_view.setCameraPosition(distance=max(num_slices, n_bins, z_max) * 2, elevation=30, azimuth=45)

    def update_bins(self, bins: List[IntensityBin]) -> None:
        """
        Update the 3D histogram to tint the surface with bin colors.
        
        Parameters
        ----------
        bins : List[IntensityBin]
            List of intensity bins to display
        """
        if self._gl_view is None or self._count_data is None or self._surface_item is None:
            return
        
        if not bins or not hasattr(self, '_base_colors') or self._base_colors is None:
            return
        
        # Get bin colors and create tinted surface
        num_slices = self._count_data.shape[0]
        n_bins = self._count_data.shape[1]
        
        # Create tinted colors based on bins
        tinted_colors = self._base_colors.copy()
        
        # For each bin, tint the corresponding intensity range
        for bin_obj in bins:
            if not bin_obj.enabled:
                continue
            
            # Get bin color
            if bin_obj.color is not None:
                r, g, b = bin_obj.color
                bin_color = np.array([r, g, b, 1.0])
            else:
                bin_color = np.array([1.0, 1.0, 0.0, 1.0])  # Yellow default
            
            # Find which intensity bins fall within this bin's range
            y = np.linspace(self._vmin, self._vmax, n_bins)
            bin_mask = (y >= bin_obj.low) & (y <= bin_obj.high)
            
            # Tint the colors in this range (blend with bin color)
            for slice_idx in range(num_slices):
                for intensity_idx in range(n_bins):
                    if bin_mask[intensity_idx]:
                        # Blend base color with bin color (70% bin, 30% base)
                        base = tinted_colors[slice_idx, intensity_idx, :3]
                        tinted = 0.7 * bin_color[:3] + 0.3 * base
                        tinted_colors[slice_idx, intensity_idx, :3] = tinted
                        tinted_colors[slice_idx, intensity_idx, 3] = 0.8  # Alpha
        
        # Update surface colors by recreating the surface item
        # Store current camera position
        try:
            camera_opts = self._gl_view.opts
            camera_distance = camera_opts.get('distance', max(num_slices, n_bins) * 2)
            camera_elevation = camera_opts.get('elevation', 30)
            camera_azimuth = camera_opts.get('azimuth', 45)
        except:
            camera_distance = max(num_slices, n_bins) * 2
            camera_elevation = 30
            camera_azimuth = 45
        
        # Remove old surface
        self._gl_view.removeItem(self._surface_item)
        
        # Recompute Z data
        Z = self._count_data.astype(float)
        Z_log = np.log1p(Z)
        z_data = Z_log * (50.0 / (Z_log.max() + 1e-10))
        
        # Get x and y
        x_data = np.arange(num_slices, dtype=float)
        y_data = np.linspace(self._vmin, self._vmax, n_bins)
        
        # Create new surface with tinted colors
        self._surface_item = gl.GLSurfacePlotItem(
            x=x_data,
            y=y_data,
            z=z_data,
            colors=tinted_colors,
            shader='balloon',
            computeNormals=True,
        )
        self._gl_view.addItem(self._surface_item)
        
        # Restore camera position
        self._gl_view.setCameraPosition(distance=camera_distance, elevation=camera_elevation, azimuth=camera_azimuth)

