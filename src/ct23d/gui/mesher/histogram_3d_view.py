from __future__ import annotations

from typing import Optional, List, Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QFrame, QVBoxLayout
from PySide6 import QtCore
from PySide6.QtGui import QColor

from ct23d.core.models import IntensityBin


class Histogram3DView(QFrame):
    """
    3D histogram view showing:
    - X axis: Slice number
    - Y axis: Intensity
    - Z axis: Pixel count (shown as color/intensity in 2D heatmap)
    
    Since PyQtGraph doesn't have native 3D support, we display this as a 2D heatmap
    where the color represents the pixel count (Z axis).
    """

    def __init__(self, parent: Optional[QFrame] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Use PyQtGraph's PlotWidget for 2D heatmap representation
        self._plot = pg.PlotWidget(self)
        layout.addWidget(self._plot)

        self._plot.setLabel("bottom", "Slice Number")
        self._plot.setLabel("left", "Intensity")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Store data for potential future use
        self._intensity_data: Optional[np.ndarray] = None
        self._slice_data: Optional[np.ndarray] = None
        self._count_data: Optional[np.ndarray] = None
        self._vmin: float = 0.0
        self._vmax: float = 255.0
        self._num_slices: int = 0
        self._bin_lines: List[pg.InfiniteLine] = []
        self._bin_labels: List[pg.TextItem] = []
        self.bin_boundary_changed: Optional[Callable[[int, str, float], None]] = None

    def clear(self) -> None:
        """Clear the 3D histogram plot."""
        self._plot.clear()
        self._intensity_data = None
        self._slice_data = None
        self._count_data = None
        self._bin_lines.clear()
        self._bin_labels.clear()

    def set_histogram_3d(
        self,
        volume: np.ndarray,
        n_bins: int = 256,
        value_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        Compute and display a 3D histogram from a volume.
        
        Displays as a 2D heatmap where:
        - X axis: Slice numbers
        - Y axis: Intensity bins
        - Color/intensity: Pixel count (Z axis)
        
        Parameters
        ----------
        volume : np.ndarray
            Volume data, shape (Z, Y, X, 3) for RGB or (Z, Y, X) for grayscale
        n_bins : int
            Number of intensity bins
        value_range : Optional[tuple[float, float]]
            Intensity range (vmin, vmax). If None, computed from data.
        """
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
        # Result: (num_slices, n_bins) - pixel counts per slice per intensity bin
        slice_histograms = []
        intensity_bin_edges = np.linspace(vmin, vmax, n_bins + 1)
        intensity_bin_centers = (intensity_bin_edges[:-1] + intensity_bin_edges[1:]) / 2
        
        for z_idx in range(num_slices):
            slice_data = grayscale[z_idx]
            # Compute histogram for ALL pixels (including zeros)
            # This ensures we see slices even if they have mostly zeros
            # Use the same bin edges for all slices to ensure consistency
            # Note: numpy.histogram uses [left, right) for all bins except the last which is [left, right]
            # We need to ensure values at vmax are included in the last bin
            # Clip values to [vmin, vmax] to ensure they're in range
            slice_data_clipped = np.clip(slice_data, vmin, vmax)
            # Use the same bin edges for all slices
            hist, _ = np.histogram(slice_data_clipped, bins=intensity_bin_edges)
            slice_histograms.append(hist)
        
        histograms_2d = np.array(slice_histograms)  # (num_slices, n_bins)
        
        # Store data
        self._count_data = histograms_2d
        
        # Clear previous plot
        self._plot.clear()
        
        # Create 2D heatmap where:
        # - Rows (Y axis) = Intensity bins
        # - Cols (X axis) = Slice numbers
        # - Color = Pixel count (Z axis)
        # Transpose so that X = slice number, Y = intensity
        img_data = histograms_2d.T  # (n_bins, num_slices) - transposed
        
        # Normalize for better visualization (log scale to handle large variations)
        img_data_log = np.log1p(img_data)  # log(1+x) to handle zeros
        
        # Create image item
        img_item = pg.ImageItem(img_data_log)
        self._plot.addItem(img_item)
        
        # Set up axes (already set in __init__, but ensure they're correct)
        self._plot.setLabel("bottom", "Slice Number")
        self._plot.setLabel("left", "Intensity")
        
        # Set axis ranges - image item uses (x, y, width, height)
        # For ImageItem, the rect maps the image rectangle to data coordinates
        # Image has shape (n_bins, num_slices) = (rows, cols)
        # We want: column j (0 to num_slices-1) maps to slice j (integer)
        # And: row i (0 to n_bins-1) maps to intensity bin i
        
        # ImageItem maps pixel boundaries, not centers
        # Column 0 spans from x = -0.5 to x = 0.5 (center at 0)
        # Column num_slices-1 spans from x = num_slices-1.5 to x = num_slices-0.5 (center at num_slices-1)
        # So the rect should span from x = -0.5 to x = num_slices - 0.5
        
        # For Y axis (intensity), we want row 0 (lowest bin) at y = vmin, row n_bins-1 at y = vmax
        # Row 0 spans from y = vmin to y = vmin + bin_width (center at vmin + 0.5*bin_width)
        # Row n_bins-1 spans from y = vmax - bin_width to y = vmax (center at vmax - 0.5*bin_width)
        
        # Calculate bin width for intensity
        intensity_bin_width = (vmax - vmin) / n_bins if n_bins > 0 else 1.0
        
        # Set rect: (x_min, y_min, width, height)
        # x_min = -0.5 so column 0 center is at x=0
        # y_min = vmin so row 0 bottom edge is at y=vmin
        # width = num_slices so column num_slices-1 right edge is at x=num_slices-0.5
        # height = vmax - vmin so row n_bins-1 top edge is at y=vmax
        img_item.setRect(QtCore.QRectF(
            -0.5, vmin,  # x, y position (left edge of column 0, bottom edge of row 0)
            num_slices, vmax - vmin  # width, height (total span)
        ))
        
        # Auto-zoom to fit the data (show integer slice numbers 0 to num_slices-1)
        # X range: -0.5 to num_slices-0.5 (so slice centers are at integers)
        # Y range: vmin to vmax (full intensity range)
        self._plot.setXRange(-0.5, num_slices - 0.5, padding=0)
        self._plot.setYRange(vmin, vmax, padding=0)
        
        # Add color bar to show pixel count scale
        # Note: PyQtGraph's colorbar might need different API
        try:
            self._plot.addColorBar(img_item, colorMap='viridis', label='Pixel Count (log scale)')
        except Exception:
            # Fallback if colorbar API is different
            pass
    
    def update_bins(self, bins: List[IntensityBin]) -> None:
        """
        Update the histogram to show bin boundaries with colored lines and labels.
        
        Parameters
        ----------
        bins : List[IntensityBin]
            List of intensity bins to display on the histogram
        """
        # Remove existing bin lines and labels
        for line in self._bin_lines:
            self._plot.removeItem(line)
        for label in self._bin_labels:
            self._plot.removeItem(label)
        self._bin_lines.clear()
        self._bin_labels.clear()
        
        if not bins:
            return
        
        # Generate unique colors for each bin if not already set
        import colorsys
        num_bins = len(bins)
        
        for i, bin_obj in enumerate(bins):
            if not bin_obj.enabled:
                continue
            
            # Get color for this bin (use bin color if set, otherwise generate unique color)
            if bin_obj.color is not None:
                r, g, b = bin_obj.color
                color = QColor(int(r * 255), int(g * 255), int(b * 255))
            else:
                # Generate a unique color using HSV
                hue = (i / max(num_bins, 1)) * 0.8  # Use 0-0.8 range for vibrant colors
                saturation = 0.8
                value = 0.9
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                color = QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            
            # Draw lower boundary (low value) - make it draggable
            line_low = pg.InfiniteLine(
                pos=bin_obj.low,
                angle=0,  # Horizontal line
                pen=pg.mkPen(color=color, width=2),
                movable=True,
                bounds=[self._vmin, self._vmax]
            )
            # Store bin's table row index (from bin_obj.index) and which boundary (low/high) for callback
            # Use the bin's index attribute which corresponds to the table row
            bin_table_row = bin_obj.index if hasattr(bin_obj, 'index') and bin_obj.index is not None else i
            line_low.bin_table_row = bin_table_row
            line_low.bin_low_value = bin_obj.low  # Store original values for matching
            line_low.bin_high_value = bin_obj.high
            line_low.boundary_type = 'low'
            line_low.sigPositionChanged.connect(lambda line: self._on_line_moved(line))
            self._plot.addItem(line_low)
            self._bin_lines.append(line_low)
            
            # Draw upper boundary (high value) - make it draggable
            line_high = pg.InfiniteLine(
                pos=bin_obj.high,
                angle=0,  # Horizontal line
                pen=pg.mkPen(color=color, width=2),
                movable=True,
                bounds=[self._vmin, self._vmax]
            )
            line_high.bin_table_row = bin_table_row
            line_high.bin_low_value = bin_obj.low
            line_high.bin_high_value = bin_obj.high
            line_high.boundary_type = 'high'
            line_high.sigPositionChanged.connect(lambda line: self._on_line_moved(line))
            self._plot.addItem(line_high)
            self._bin_lines.append(line_high)
            
            # Add label with bin ID - use index from bin object (0-based), display as 1-based
            bin_id = bin_obj.index if hasattr(bin_obj, 'index') and bin_obj.index is not None else i
            label_text = f"Bin {bin_id + 1}"  # Display as 1-based to match table (bin_1, bin_2, etc.)
            
            # Position label at the middle of the bin range, at the left edge of the plot
            mid_intensity = (bin_obj.low + bin_obj.high) / 2
            label = pg.TextItem(
                text=label_text,
                color=color,
                anchor=(0, 0.5),  # Left-aligned, vertically centered
                border=pg.mkPen(color=color, width=1),
                fill=pg.mkBrush((0, 0, 0, 200))  # Semi-transparent black background
            )
            # Position at left edge of plot
            label.setPos(-0.5, mid_intensity)
            self._plot.addItem(label)
            self._bin_labels.append(label)
    
    def _on_line_moved(self, line: pg.InfiniteLine) -> None:
        """Called when a bin boundary line is dragged."""
        # Get the new position (round to integer)
        new_pos = int(round(line.value()))
        bin_table_row = getattr(line, 'bin_table_row', None)
        boundary_type = line.boundary_type
        
        # Call callback to parent to update table
        if hasattr(self, 'bin_boundary_changed') and self.bin_boundary_changed is not None:
            if bin_table_row is not None:
                self.bin_boundary_changed(bin_table_row, boundary_type, float(new_pos))
    
    def set_bin_boundary_callback(self, callback) -> None:
        """Set callback for when bin boundaries are moved."""
        self.bin_boundary_changed = callback
