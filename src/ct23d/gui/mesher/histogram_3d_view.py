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

        # Swap axes: intensity horizontal (bottom), slice number vertical (left)
        self._plot.setLabel("bottom", "Intensity")
        self._plot.setLabel("left", "Slice Number")
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
        self.range_line_changed: Optional[Callable[[str, float], None]] = None
        self._range_min_line: Optional[pg.InfiniteLine] = None
        self._range_max_line: Optional[pg.InfiniteLine] = None
        self._range_min_label: Optional[pg.TextItem] = None
        self._range_max_label: Optional[pg.TextItem] = None

    def clear(self) -> None:
        """Clear the 3D histogram plot."""
        if self._range_min_line is not None:
            self._plot.removeItem(self._range_min_line)
            self._range_min_line = None
        if self._range_max_line is not None:
            self._plot.removeItem(self._range_max_line)
            self._range_max_line = None
        if self._range_min_label is not None:
            self._plot.removeItem(self._range_min_label)
            self._range_min_label = None
        if self._range_max_label is not None:
            self._plot.removeItem(self._range_max_label)
            self._range_max_label = None
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
        progress_callback: Optional[Callable[[int, int], None]] = None,
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
        progress_callback : Optional[Callable[[int, int], None]]
            Optional callback function(current, total) called during slice processing
            to report progress. Called once per slice processed.
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
        # Use slightly extended range to ensure vmax is included in last bin
        # numpy.histogram uses [left, right) for all bins except last which is [left, right]
        intensity_bin_edges = np.linspace(vmin, vmax, n_bins + 1)
        # Ensure the last edge includes vmax by making it slightly larger
        intensity_bin_edges[-1] = vmax + 1e-6
        intensity_bin_centers = (intensity_bin_edges[:-1] + intensity_bin_edges[1:]) / 2
        
        # Process slices - process events more frequently to keep UI responsive
        from PySide6.QtWidgets import QApplication
        batch_size = max(10, num_slices // 20)  # Update every 5% or every 10 slices (more frequent)
        
        for z_idx in range(num_slices):
            slice_data = grayscale[z_idx]
            # Compute histogram for ALL pixels (including zeros)
            # This ensures we see slices even if they have mostly zeros
            # Use the same bin edges for all slices to ensure consistency
            # Clip values to [vmin, vmax+epsilon] to ensure vmax is included
            slice_data_clipped = np.clip(slice_data, vmin, vmax + 1e-6)
            # Use the same bin edges for all slices
            hist, _ = np.histogram(slice_data_clipped, bins=intensity_bin_edges)
            slice_histograms.append(hist)
            
            # Update progress callback if provided
            if progress_callback is not None:
                progress_callback(z_idx + 1, num_slices)
            
            # Process events more frequently to keep UI responsive
            if (z_idx + 1) % batch_size == 0:
                QApplication.processEvents()
        
        histograms_2d = np.array(slice_histograms)  # (num_slices, n_bins)
        
        # Store data
        self._count_data = histograms_2d
        
        # Clear previous plot (this removes bin lines, but we'll re-add them in update_bins)
        self._plot.clear()
        # Clear the bin_lines list since the items are no longer in the plot
        self._bin_lines.clear()
        self._bin_labels.clear()
        
        # Create 2D heatmap where:
        # - X axis (horizontal) = Intensity bins
        # - Y axis (vertical) = Slice numbers
        # - Color = Pixel count (Z axis)
        # PyQtGraph ImageItem displays data as (rows, cols) where:
        #   - rows = vertical (Y) axis = slice numbers
        #   - cols = horizontal (X) axis = intensity bins
        # Our data is (num_slices, n_bins) which is (Y, X), so we need to transpose
        # to get (intensity bins, slice numbers) = (X, Y), then set rect accordingly
        # Actually, wait - let's check orientation. ImageItem maps [row, col] where row is Y (vertical)
        # and col is X (horizontal). So (num_slices, n_bins) = (rows, cols) = (Y, X) is correct.
        # But if axes appear inverted, we need to transpose to (n_bins, num_slices) = (X, Y)
        # and adjust the rect mapping
        img_data = histograms_2d.T  # Transpose: (n_bins, num_slices) - rows (Y) are intensity, cols (X) are slices
        
        # Normalize for better visualization (log scale to handle large variations)
        img_data_log = np.log1p(img_data)  # log(1+x) to handle zeros
        
        # Create image item
        img_item = pg.ImageItem(img_data_log)
        self._plot.addItem(img_item)
        
        # Set up axes (will be swapped after transpose)
        # Initial labels, will be corrected after rect setup
        self._plot.setLabel("bottom", "Slice Number")
        self._plot.setLabel("left", "Intensity")
        
        # Set axis ranges - image item uses (x, y, width, height)
        # After transpose, image has shape (n_bins, num_slices) = (rows, cols)
        # We want: X axis (horizontal) = Intensity bins, Y axis (vertical) = Slice numbers
        # So: columns (X) should map to slices, rows (Y) should map to intensity
        # But ImageItem maps [row, col] to [y, x] where row=Y (vertical) and col=X (horizontal)
        # With transposed data (n_bins, num_slices):
        #   - rows (Y/vertical) = intensity bins (0 to n_bins-1)
        #   - cols (X/horizontal) = slice numbers (0 to num_slices-1)
        # So we need to map:
        #   - X axis (horizontal/cols): 0 to num_slices-1 maps to slice 0 to num_slices-1
        #   - Y axis (vertical/rows): 0 to n_bins-1 maps to intensity vmin to vmax
        
        # Calculate bin width for intensity
        intensity_bin_width = (vmax - vmin) / n_bins if n_bins > 0 else 1.0
        
        # Set rect: (x_min, y_min, width, height)
        # x_min = -0.5 so column 0 (first slice) left edge is at x=-0.5
        # y_min = vmin so row 0 (lowest intensity) bottom edge is at y=vmin
        # width = num_slices so column num_slices-1 right edge is at x=num_slices-0.5
        # height = vmax - vmin so row n_bins-1 top edge is at y=vmax
        img_item.setRect(QtCore.QRectF(
            -0.5, vmin,  # x, y position (left edge of slice 0, bottom edge of intensity vmin)
            num_slices, vmax - vmin  # width, height (slice span, intensity span)
        ))
        
        # Auto-zoom to fit the data
        # X range: -0.5 to num_slices-0.5 (so slice centers are at integers)
        # Y range: vmin to vmax (full intensity range) with small padding
        self._plot.setXRange(-0.5, max(num_slices - 0.5, 0.5), padding=0.02)
        self._plot.setYRange(vmin, vmax, padding=0.02)
        
        # User wants: X axis = Intensity, Y axis = Slice Number (swapped labels, not graph)
        # So swap the labels to match what user expects to see
        self._plot.setLabel("bottom", "Intensity")
        self._plot.setLabel("left", "Slice Number")
        
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
        if not bins:
            # Remove all lines if no bins
            for line in self._bin_lines:
                self._plot.removeItem(line)
            for label in self._bin_labels:
                self._plot.removeItem(label)
            self._bin_lines.clear()
            self._bin_labels.clear()
            return
        
        # Try to update existing lines instead of removing/recreating them
        # This prevents lines from disappearing during dragging
        bins_dict = {}
        for bin_obj in bins:
            if bin_obj.enabled and hasattr(bin_obj, 'index') and bin_obj.index is not None:
                bins_dict[bin_obj.index] = bin_obj
        
        # Update existing lines that match bins
        lines_to_remove = []
        for line in self._bin_lines:
            bin_table_row = getattr(line, 'bin_table_row', None)
            boundary_type = getattr(line, 'boundary_type', None)
            
            if bin_table_row is not None and bin_table_row in bins_dict:
                bin_obj = bins_dict[bin_table_row]
                # Update line position if it changed
                expected_pos = bin_obj.low if boundary_type == 'low' else bin_obj.high
                if abs(line.value() - expected_pos) > 0.5:  # Only update if significantly different
                    line.blockSignals(True)
                    line.setValue(expected_pos)
                    line.blockSignals(False)
            else:
                # Line doesn't match any bin, mark for removal
                lines_to_remove.append(line)
        
        # Remove lines that don't match any bin
        for line in lines_to_remove:
            self._plot.removeItem(line)
            self._bin_lines.remove(line)
        
        # Remove labels that don't match
        labels_to_remove = []
        for label in self._bin_labels:
            # Labels don't have direct bin reference, so we'll recreate them
            labels_to_remove.append(label)
        
        for label in labels_to_remove:
            self._plot.removeItem(label)
            self._bin_labels.remove(label)
        
        # Now add missing lines and labels for bins that don't have them yet
        existing_line_keys = set()
        for line in self._bin_lines:
            bin_table_row = getattr(line, 'bin_table_row', None)
            boundary_type = getattr(line, 'boundary_type', None)
            if bin_table_row is not None and boundary_type is not None:
                existing_line_keys.add((bin_table_row, boundary_type))
        
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
            
            bin_table_row = bin_obj.index if hasattr(bin_obj, 'index') and bin_obj.index is not None else i
            
            # Add low boundary line if it doesn't exist
            # Bin boundaries should be HORIZONTAL (angle=0) since intensity is on Y-axis (vertical) after transpose
            if (bin_table_row, 'low') not in existing_line_keys:
                # Ensure position is within bounds
                pos = max(self._vmin, min(bin_obj.low, self._vmax))
                line_low = pg.InfiniteLine(
                    pos=pos,
                    angle=0,  # Horizontal line (across intensity axis, which is now vertical)
                    pen=pg.mkPen(color=color, width=3, style=QtCore.Qt.SolidLine),  # Thicker, more visible
                    movable=True,
                    bounds=[self._vmin, self._vmax]
                )
                line_low.setZValue(10)  # Draw on top of heatmap
                line_low.bin_table_row = bin_table_row
                line_low.bin_low_value = bin_obj.low
                line_low.bin_high_value = bin_obj.high
                line_low.boundary_type = 'low'
                line_low.sigPositionChanged.connect(lambda line: self._on_line_moved(line))
                self._plot.addItem(line_low)
                self._bin_lines.append(line_low)
            
            # Add high boundary line if it doesn't exist
            if (bin_table_row, 'high') not in existing_line_keys:
                # Ensure position is within bounds
                pos = max(self._vmin, min(bin_obj.high, self._vmax))
                line_high = pg.InfiniteLine(
                    pos=pos,
                    angle=0,  # Horizontal line (across intensity axis, which is now vertical)
                    pen=pg.mkPen(color=color, width=3, style=QtCore.Qt.SolidLine),  # Thicker, more visible
                    movable=True,
                    bounds=[self._vmin, self._vmax]
                )
                line_high.setZValue(10)  # Draw on top of heatmap
                line_high.bin_table_row = bin_table_row
                line_high.bin_low_value = bin_obj.low
                line_high.bin_high_value = bin_obj.high
                line_high.boundary_type = 'high'
                line_high.sigPositionChanged.connect(lambda line: self._on_line_moved(line))
                self._plot.addItem(line_high)
                self._bin_lines.append(line_high)
            
            # Add label on the left side of the plot (Y-axis side, since intensity is now vertical)
            # Position at the middle of the bin's intensity range (Y position) and at slice 0 (X position)
            bin_id = bin_obj.index if hasattr(bin_obj, 'index') and bin_obj.index is not None else i
            label_text = f"Bin {bin_id + 1}"
            mid_intensity = (bin_obj.low + bin_obj.high) / 2
            label = pg.TextItem(
                text=label_text,
                color=color,
                anchor=(1, 0.5),  # Right-aligned, vertically centered (so it appears on left Y-axis side)
                border=pg.mkPen(color=color, width=1),
                fill=pg.mkBrush((0, 0, 0, 200))
            )
            # Position: X = -0.5 (left edge, slice 0), Y = mid_intensity (intensity position)
            label.setPos(-0.5, mid_intensity)
            self._plot.addItem(label)
            self._bin_labels.append(label)
    
    def _on_line_moved(self, line: pg.InfiniteLine) -> None:
        """Called when a bin boundary line is dragged."""
        # Get the new position (round to integer)
        new_pos = int(round(line.value()))
        # Ensure minimum is at least 1
        if new_pos < 1:
            new_pos = 1
            line.blockSignals(True)
            line.setValue(1.0)
            line.blockSignals(False)
        bin_table_row = getattr(line, 'bin_table_row', None)
        boundary_type = line.boundary_type
        
        # Call callback to parent to update table
        if hasattr(self, 'bin_boundary_changed') and self.bin_boundary_changed is not None:
            if bin_table_row is not None:
                self.bin_boundary_changed(bin_table_row, boundary_type, float(new_pos))
    
    def set_bin_boundary_callback(self, callback) -> None:
        """Set callback for when bin boundaries are moved."""
        self.bin_boundary_changed = callback
    
    def set_range_line_callback(self, callback) -> None:
        """Set callback for when range lines (min/max) are moved."""
        self.range_line_changed = callback
    
    def update_range_lines(self, min_val: Optional[int], max_val: Optional[int]) -> None:
        """Update the range lines (min/max) displayed on the histogram."""
        # Remove existing range lines and labels if any
        if self._range_min_line is not None:
            self._plot.removeItem(self._range_min_line)
            self._range_min_line = None
        if self._range_max_line is not None:
            self._plot.removeItem(self._range_max_line)
            self._range_max_line = None
        if self._range_min_label is not None:
            self._plot.removeItem(self._range_min_label)
            self._range_min_label = None
        if self._range_max_label is not None:
            self._plot.removeItem(self._range_max_label)
            self._range_max_label = None
        
        # Only create lines if values are provided
        if min_val is None or max_val is None or self._vmax <= self._vmin:
            return
        
        # Create min range line (vertical, dashed, distinct color)
        # User wants vertical lines for intensity range (since labels show intensity on X axis)
        # But intensity is actually on Y axis in the data, so we need vertical lines (angle=90) 
        # positioned at the intensity value (Y position)
        self._range_min_line = pg.InfiniteLine(
            pos=float(min_val),
            angle=90,  # Vertical line (since intensity is on Y axis in the plot)
            pen=pg.mkPen(color='cyan', width=2, style=QtCore.Qt.DashLine),
            movable=True,
            bounds=[self._vmin, self._vmax]
        )
        self._range_min_line.setZValue(20)  # Above bin lines
        self._range_min_line.line_type = 'min'
        self._range_min_line.sigPositionChanged.connect(lambda line: self._on_range_line_moved(line))
        self._plot.addItem(self._range_min_line)
        
        # Create min label at top of plot (centered on X axis)
        if self._plot.getViewBox() is not None:
            y_range = self._plot.getViewBox().viewRange()[1]
            y_max = y_range[1]
            x_range = self._plot.getViewBox().viewRange()[0]
            x_center = (x_range[0] + x_range[1]) / 2
        else:
            y_max = self._vmax
            x_center = float(self._num_slices) / 2 if self._num_slices > 0 else 50.0
        self._range_min_label = pg.TextItem(
            text=f"Min: {min_val}",
            color='cyan',
            anchor=(0.5, 0),  # Center-aligned, top-anchored
            border=pg.mkPen(color='cyan', width=1),
            fill=pg.mkBrush((0, 0, 0, 200))
        )
        self._range_min_label.setPos(x_center, y_max * 0.96)  # Top of plot, centered horizontally
        self._plot.addItem(self._range_min_label)
        
        # Create max range line (vertical, dashed, distinct color)
        self._range_max_line = pg.InfiniteLine(
            pos=float(max_val),
            angle=90,  # Vertical line (since intensity is on Y axis in the plot)
            pen=pg.mkPen(color='magenta', width=2, style=QtCore.Qt.DashLine),
            movable=True,
            bounds=[self._vmin, self._vmax]
        )
        self._range_max_line.setZValue(20)  # Above bin lines
        self._range_max_line.line_type = 'max'
        self._range_max_line.sigPositionChanged.connect(lambda line: self._on_range_line_moved(line))
        self._plot.addItem(self._range_max_line)
        
        # Create max label at top of plot (higher position to avoid overlap with min)
        self._range_max_label = pg.TextItem(
            text=f"Max: {max_val}",
            color='magenta',
            anchor=(0.5, 0),  # Center-aligned, top-anchored
            border=pg.mkPen(color='magenta', width=1),
            fill=pg.mkBrush((0, 0, 0, 200))
        )
        self._range_max_label.setPos(x_center, y_max * 0.98)  # Higher position at top of plot
        self._plot.addItem(self._range_max_label)
    
    def _on_range_line_moved(self, line: pg.InfiniteLine) -> None:
        """Called when a range line is dragged."""
        new_pos = int(round(line.value()))
        line_type = getattr(line, 'line_type', None)
        
        # Ensure within bounds
        if new_pos < self._vmin:
            new_pos = int(round(self._vmin))
            line.blockSignals(True)
            line.setValue(float(new_pos))
            line.blockSignals(False)
        elif new_pos > self._vmax:
            new_pos = int(round(self._vmax))
            line.blockSignals(True)
            line.setValue(float(new_pos))
            line.blockSignals(False)
        
        # Call callback to update controls
        if self.range_line_changed is not None and line_type is not None:
            self.range_line_changed(line_type, float(new_pos))
        
        # Update label position (intensity is on Y axis, so line moves vertically)
        # Labels are positioned at top of plot, so we only update the text, not the position
        if line_type == 'min' and self._range_min_label is not None:
            self._range_min_label.setText(f"Min: {new_pos}")
        elif line_type == 'max' and self._range_max_label is not None:
            self._range_max_label.setText(f"Max: {new_pos}")
