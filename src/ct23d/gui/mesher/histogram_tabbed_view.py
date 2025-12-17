from __future__ import annotations

from typing import Optional, List, Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QFrame, QVBoxLayout, QTabWidget, QWidget
from PySide6 import QtCore
from PySide6.QtGui import QColor

from ct23d.core.models import IntensityBin
from ct23d.gui.mesher.histogram_3d_view import Histogram3DView


class AggregatedHistogramView(QFrame):
    """
    Aggregated histogram view showing intensity distribution across all slices.
    - X axis: Intensity
    - Y axis: Pixel count (aggregated across all slices)
    """
    
    def __init__(self, parent: Optional[QFrame] = None) -> None:
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Use PyQtGraph's PlotWidget for histogram
        self._plot = pg.PlotWidget(self)
        layout.addWidget(self._plot)
        
        self._plot.setLabel("bottom", "Intensity")
        self._plot.setLabel("left", "Pixel Count")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Store data
        self._histogram_data: Optional[np.ndarray] = None
        self._bin_edges: Optional[np.ndarray] = None
        self._vmin: float = 0.0
        self._vmax: float = 255.0
        self._bin_lines: List[pg.InfiniteLine] = []
        self._bin_labels: List[pg.TextItem] = []
        self.bin_boundary_changed: Optional[Callable[[int, str, float], None]] = None
        self.range_line_changed: Optional[Callable[[str, float], None]] = None
        self._range_min_line: Optional[pg.InfiniteLine] = None
        self._range_max_line: Optional[pg.InfiniteLine] = None
        self._range_min_label: Optional[pg.TextItem] = None
        self._range_max_label: Optional[pg.TextItem] = None
    
    def clear(self) -> None:
        """Clear the histogram plot."""
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
        self._histogram_data = None
        self._bin_edges = None
        self._bin_lines.clear()
        self._bin_labels.clear()
    
    def set_aggregated_histogram(
        self,
        volume: np.ndarray,
        n_bins: int = 256,
        value_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        Compute and display aggregated histogram from a volume.
        
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
        
        # Flatten all slices into a single array
        intensities = grayscale.ravel()
        
        # Filter out intensity 0 pixels (background/air)
        intensities_filtered = intensities[intensities > 0]
        
        if intensities_filtered.size == 0:
            # Fallback: if everything is zero, use all intensities
            intensities_filtered = intensities
        
        # Determine intensity range
        if value_range is None:
            vmin = float(intensities_filtered.min())
            vmax = float(intensities_filtered.max())
        else:
            vmin, vmax = value_range
        
        # Ensure minimum is at least 1
        if vmin < 1:
            vmin = 1.0
        
        self._vmin = vmin
        self._vmax = vmax
        
        # Compute aggregated histogram (only on filtered intensities)
        # Ensure vmax is included by extending the last bin edge slightly
        bin_edges = np.linspace(vmin, vmax, n_bins + 1)
        bin_edges[-1] = vmax + 1e-6  # Ensure vmax is included in last bin
        # Clip filtered intensities to ensure they're in range
        intensities_clipped = np.clip(intensities_filtered, vmin, vmax + 1e-6)
        hist, edges = np.histogram(intensities_clipped, bins=bin_edges)
        
        # Store data
        self._histogram_data = hist
        self._bin_edges = edges
        
        # Clear previous plot (this removes bin lines, but we'll re-add them in update_bins)
        # Store bin lines before clearing so we can check if they need to be re-added
        self._plot.clear()
        # Clear the bin_lines list since the items are no longer in the plot
        self._bin_lines.clear()
        self._bin_labels.clear()
        
        # Plot histogram as step plot (PyQtGraph style)
        # For stepMode=True, we need len(x) = len(y) + 1
        # x = bin edges (n_bins + 1 values)
        # y = bin counts (n_bins values)
        x = edges  # Bin edges (n_bins + 1 values)
        y = hist  # Bin counts (n_bins values)
        
        # Create step plot (linear scale - no log mode)
        self._plot.plot(
            x, y,
            stepMode=True,
            fillLevel=0,
            brush=(100, 150, 255, 100),
            pen=pg.mkPen(color=(100, 150, 255), width=1),
        )
        
        # Set axis ranges - auto-zoom to fit data
        self._plot.setXRange(vmin, vmax, padding=0.02)
        if hist.size > 0 and hist.max() > 0:
            y_max = float(hist.max()) * 1.05  # 5% padding
        else:
            y_max = 1.0
        self._plot.setYRange(0, y_max, padding=0.02)
    
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
        
        # Remove all labels (we'll recreate them)
        for label in self._bin_labels:
            self._plot.removeItem(label)
        self._bin_labels.clear()
        
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
                hue = (i / max(num_bins, 1)) * 0.8
                saturation = 0.8
                value = 0.9
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                color = QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            
            bin_table_row = bin_obj.index if hasattr(bin_obj, 'index') and bin_obj.index is not None else i
            
            # Add low boundary line if it doesn't exist
            # In aggregated histogram, intensity is on X-axis, so lines should be vertical (angle=90)
            if (bin_table_row, 'low') not in existing_line_keys:
                # Ensure position is within bounds
                pos = max(self._vmin, min(bin_obj.low, self._vmax))
                line_low = pg.InfiniteLine(
                    pos=pos,
                    angle=90,  # Vertical line (across intensity axis)
                    pen=pg.mkPen(color=color, width=3, style=QtCore.Qt.SolidLine),  # Thicker, more visible
                    movable=True,
                    bounds=[self._vmin, self._vmax]
                )
                line_low.setZValue(10)  # Draw on top of histogram
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
                    angle=90,  # Vertical line (across intensity axis)
                    pen=pg.mkPen(color=color, width=3, style=QtCore.Qt.SolidLine),  # Thicker, more visible
                    movable=True,
                    bounds=[self._vmin, self._vmax]
                )
                line_high.setZValue(10)  # Draw on top of histogram
                line_high.bin_table_row = bin_table_row
                line_high.bin_low_value = bin_obj.low
                line_high.bin_high_value = bin_obj.high
                line_high.boundary_type = 'high'
                line_high.sigPositionChanged.connect(lambda line: self._on_line_moved(line))
                self._plot.addItem(line_high)
                self._bin_lines.append(line_high)
            
            # Add label at the top of the plot
            bin_id = bin_obj.index if hasattr(bin_obj, 'index') and bin_obj.index is not None else i
            label_text = f"Bin {bin_id + 1}"
            mid_intensity = (bin_obj.low + bin_obj.high) / 2
            # Get current Y range from plot
            if self._plot.getViewBox() is not None:
                y_range = self._plot.getViewBox().viewRange()[1]
                y_max = y_range[1]
            else:
                # Fallback: use histogram data max
                if self._histogram_data is not None and self._histogram_data.size > 0:
                    y_max = float(self._histogram_data.max()) * 1.05
                else:
                    y_max = 1000.0
            label = pg.TextItem(
                text=label_text,
                color=color,
                anchor=(0.5, 1),  # Center-aligned, bottom-anchored
                border=pg.mkPen(color=color, width=1),
                fill=pg.mkBrush((0, 0, 0, 200))
            )
            label.setPos(mid_intensity, y_max * 0.95)
            self._plot.addItem(label)
            self._bin_labels.append(label)
    
    def _on_line_moved(self, line: pg.InfiniteLine) -> None:
        """Called when a bin boundary line is dragged."""
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
        if hasattr(self, '_range_min_label') and self._range_min_label is not None:
            self._plot.removeItem(self._range_min_label)
            self._range_min_label = None
        if hasattr(self, '_range_max_label') and self._range_max_label is not None:
            self._plot.removeItem(self._range_max_label)
            self._range_max_label = None
        
        # Only create lines if values are provided
        if min_val is None or max_val is None or self._vmax <= self._vmin:
            return
        
        # Create min range line (vertical, dashed, distinct color)
        self._range_min_line = pg.InfiniteLine(
            pos=float(min_val),
            angle=90,  # Vertical line
            pen=pg.mkPen(color='cyan', width=2, style=QtCore.Qt.DashLine),
            movable=True,
            bounds=[self._vmin, self._vmax]
        )
        self._range_min_line.setZValue(20)  # Above bin lines
        self._range_min_line.line_type = 'min'
        self._range_min_line.sigPositionChanged.connect(lambda line: self._on_range_line_moved(line))
        self._plot.addItem(self._range_min_line)
        
        # Create min label (positioned above the line, slightly offset)
        if self._plot.getViewBox() is not None:
            y_range = self._plot.getViewBox().viewRange()[1]
            y_max = y_range[1]
        else:
            if self._histogram_data is not None and self._histogram_data.size > 0:
                y_max = float(self._histogram_data.max()) * 1.05
            else:
                y_max = 1000.0
        self._range_min_label = pg.TextItem(
            text=f"Min: {min_val}",
            color='cyan',
            anchor=(0.5, 0),  # Center-aligned, top-anchored (above line)
            border=pg.mkPen(color='cyan', width=1),
            fill=pg.mkBrush((0, 0, 0, 200))
        )
        self._range_min_label.setPos(float(min_val), y_max * 0.96)  # Lower position
        self._plot.addItem(self._range_min_label)
        
        # Create max range line (vertical, dashed, distinct color)
        self._range_max_line = pg.InfiniteLine(
            pos=float(max_val),
            angle=90,  # Vertical line
            pen=pg.mkPen(color='magenta', width=2, style=QtCore.Qt.DashLine),
            movable=True,
            bounds=[self._vmin, self._vmax]
        )
        self._range_max_line.setZValue(20)  # Above bin lines
        self._range_max_line.line_type = 'max'
        self._range_max_line.sigPositionChanged.connect(lambda line: self._on_range_line_moved(line))
        self._plot.addItem(self._range_max_line)
        
        # Create max label (positioned above the line, different Y position than min)
        self._range_max_label = pg.TextItem(
            text=f"Max: {max_val}",
            color='magenta',
            anchor=(0.5, 0),  # Center-aligned, top-anchored (above line)
            border=pg.mkPen(color='magenta', width=1),
            fill=pg.mkBrush((0, 0, 0, 200))
        )
        self._range_max_label.setPos(float(max_val), y_max * 0.98)  # Higher position
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
        
        # Store current label positions before callback (which might recreate labels)
        min_label_y = self._range_min_label.pos().y() if self._range_min_label is not None else None
        max_label_y = self._range_max_label.pos().y() if self._range_max_label is not None else None
        
        # Update label position BEFORE callback (to prevent position loss)
        if line_type == 'min' and self._range_min_label is not None:
            y_pos = min_label_y if min_label_y is not None else self._range_min_label.pos().y()
            self._range_min_label.setPos(float(new_pos), y_pos)
            self._range_min_label.setText(f"Min: {new_pos}")
        elif line_type == 'max' and self._range_max_label is not None:
            y_pos = max_label_y if max_label_y is not None else self._range_max_label.pos().y()
            self._range_max_label.setPos(float(new_pos), y_pos)
            self._range_max_label.setText(f"Max: {new_pos}")
        
        # Call callback to update controls AFTER updating label position
        if self.range_line_changed is not None and line_type is not None:
            self.range_line_changed(line_type, float(new_pos))


class HistogramTabbedView(QFrame):
    """
    Tabbed histogram view with:
    - Tab 1: Aggregated Histogram (all slices combined)
    - Tab 2: Slice-by-Slice Heatmap (2D heatmap)
    - Tab 3: 3D Surface Plot (true 3D visualization)
    """
    
    def __init__(self, parent: Optional[QFrame] = None) -> None:
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self._tabs = QTabWidget(self)
        layout.addWidget(self._tabs)
        
        # Tab 1: Aggregated Histogram
        self._aggregated_view = AggregatedHistogramView(self)
        self._tabs.addTab(self._aggregated_view, "Aggregated Histogram")
        
        # Tab 2: Slice-by-Slice Heatmap (2D heatmap)
        self._heatmap_view = Histogram3DView(self)
        self._tabs.addTab(self._heatmap_view, "Slice-by-Slice Heatmap")
        
        # 3D Surface Plot removed - too slow
        self._gl_view = None
        
        # Store callback
        self.bin_boundary_changed: Optional[Callable[[int, str, float], None]] = None
    
    def clear(self) -> None:
        """Clear all histogram views."""
        self._aggregated_view.clear()
        self._heatmap_view.clear()
        if self._gl_view is not None:
            self._gl_view.clear()
    
    def set_histogram_3d(
        self,
        volume: np.ndarray,
        n_bins: int = 256,
        value_range: Optional[tuple[float, float]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Set histogram data for all views.
        
        Parameters
        ----------
        volume : np.ndarray
            Volume data
        n_bins : int
            Number of intensity bins
        value_range : Optional[tuple[float, float]]
            Intensity range (vmin, vmax)
        """
        from PySide6.QtWidgets import QApplication
        
        # Update aggregated view (fast, should complete quickly)
        try:
            self._aggregated_view.set_aggregated_histogram(volume, n_bins, value_range)
            QApplication.processEvents()
        except Exception as e:
            print(f"Error updating aggregated histogram: {e}")
        
        # Update heatmap view (can be slow for large volumes)
        try:
            self._heatmap_view.set_histogram_3d(volume, n_bins, value_range, progress_callback=progress_callback)
            QApplication.processEvents()
        except Exception as e:
            print(f"Error updating heatmap histogram: {e}")
    
    def update_bins(self, bins: List[IntensityBin]) -> None:
        """
        Update bin boundaries in both views.
        
        Parameters
        ----------
        bins : List[IntensityBin]
            List of intensity bins
        """
        self._aggregated_view.update_bins(bins)
        self._heatmap_view.update_bins(bins)
        if self._gl_view is not None:
            self._gl_view.update_bins(bins)
    
    def set_bin_boundary_callback(self, callback) -> None:
        """Set callback for when bin boundaries are moved."""
        self.bin_boundary_changed = callback
        self._aggregated_view.set_bin_boundary_callback(callback)
        self._heatmap_view.set_bin_boundary_callback(callback)
    
    def set_range_line_callback(self, callback) -> None:
        """Set callback for when range lines (min/max) are moved."""
        self._aggregated_view.set_range_line_callback(callback)
        self._heatmap_view.set_range_line_callback(callback)
    
    def update_range_lines(self, min_val: Optional[int], max_val: Optional[int]) -> None:
        """Update the range lines (min/max) on both histogram views."""
        self._aggregated_view.update_range_lines(min_val, max_val)
        self._heatmap_view.update_range_lines(min_val, max_val)

