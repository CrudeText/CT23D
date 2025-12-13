from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QDoubleSpinBox,
    QSpinBox,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QComboBox,
    QCheckBox,
)
from PySide6.QtGui import QColor

from ct23d.core import images
from ct23d.core.models import IntensityBin, MeshingConfig
from ct23d.core import meshing as meshing_core
from ct23d.core import export as export_core
from ct23d.gui.status import StatusController
from ct23d.gui.workers import FunctionWorker, WorkerBase
from PySide6.QtCore import Signal
from ct23d.gui.mesher.histogram_3d_view import Histogram3DView
from ct23d.gui.mesher.slice_preview import SlicePreviewWidget


class MesherWizard(QWidget):
    """
    Meshing tab widget.

    - Select processed slices directory
    - Load volume + compute histogram in a background thread
    - Configure spacing, intensity bins, output directory, prefix
    - Generate meshes in a background thread, with progress
    """

    def __init__(self, status: StatusController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.status = status

        self._processed_dir: Optional[Path] = None
        self._default_processed_dir: Optional[Path] = None  # Default from preprocessing
        self._output_dir: Optional[Path] = None
        self._volume: Optional[np.ndarray] = None
        self._current_range: Tuple[float, float] = (0.0, 255.0)

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Top: processed directory selector
        top_row = QHBoxLayout()
        self.processed_label = QLabel("Processed slices directory: (none)")
        self.btn_select_processed = QPushButton("Select Folder (or use default)")
        self.btn_select_processed.clicked.connect(self._on_select_processed_dir)
        top_row.addWidget(self.processed_label, stretch=1)
        top_row.addWidget(self.btn_select_processed)
        main_layout.addLayout(top_row)

        # Meshing parameters group
        params_group = QGroupBox("Meshing parameters")
        params_layout = QGridLayout(params_group)

        # Spacing
        params_layout.addWidget(QLabel("Spacing (mm) Z / Y / X"), 0, 0)
        self.spin_spacing_z = QDoubleSpinBox()
        self.spin_spacing_y = QDoubleSpinBox()
        self.spin_spacing_x = QDoubleSpinBox()
        for sp in (self.spin_spacing_z, self.spin_spacing_y, self.spin_spacing_x):
            sp.setRange(0.001, 1000.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.1)
        self.spin_spacing_z.setValue(1.6)
        self.spin_spacing_y.setValue(1.0)
        self.spin_spacing_x.setValue(1.0)

        params_layout.addWidget(self.spin_spacing_z, 0, 1)
        params_layout.addWidget(self.spin_spacing_y, 0, 2)
        params_layout.addWidget(self.spin_spacing_x, 0, 3)

        # Number of bins
        params_layout.addWidget(QLabel("Number of bins:"), 1, 0)
        self.spin_num_bins = QSpinBox()
        self.spin_num_bins.setRange(1, 1024)
        self.spin_num_bins.setValue(6)
        params_layout.addWidget(self.spin_num_bins, 1, 1)

        # Intensity range (integers)
        params_layout.addWidget(QLabel("Intensity range min / max:"), 1, 2)
        self.spin_int_min = QSpinBox()
        self.spin_int_max = QSpinBox()
        for sp in (self.spin_int_min, self.spin_int_max):
            sp.setRange(0, 255)
            sp.setSingleStep(1)
        self.spin_int_min.setValue(1)
        self.spin_int_max.setValue(255)

        params_layout.addWidget(self.spin_int_min, 1, 3)
        params_layout.addWidget(self.spin_int_max, 1, 4)

        main_layout.addWidget(params_group)
        
        # Compute histogram button (moved here)
        self.btn_load_volume = QPushButton("Compute histogram")
        self.btn_load_volume.clicked.connect(self.load_volume_and_histogram)
        main_layout.addWidget(self.btn_load_volume)

        # 3D Histogram + bin table + preview
        middle_layout = QHBoxLayout()
        
        # Left: Histogram
        self.histogram = Histogram3DView(self)
        # Connect bin boundary changes to update table
        self.histogram.set_bin_boundary_callback(self._on_histogram_bin_moved)
        middle_layout.addWidget(self.histogram, stretch=3)
        
        # Middle: Bin table with controls
        bin_group = QGroupBox("Intensity Bins")
        bin_layout = QVBoxLayout(bin_group)
        
        # Add/Delete buttons
        bin_buttons = QHBoxLayout()
        self.btn_add_bin = QPushButton("Add Bin")
        self.btn_add_bin.clicked.connect(self._on_add_bin)
        self.btn_delete_bin = QPushButton("Delete Selected")
        self.btn_delete_bin.clicked.connect(self._on_delete_bin)
        bin_buttons.addWidget(self.btn_add_bin)
        bin_buttons.addWidget(self.btn_delete_bin)
        bin_layout.addLayout(bin_buttons)
        
        self.bins_table = QTableWidget(0, 5, self)
        self.bins_table.setHorizontalHeaderLabels(["Enabled", "Low", "High", "Name", "Color"])
        self.bins_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.bins_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.bins_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.bins_table.cellChanged.connect(self._on_bin_table_changed)
        self.bins_table.cellDoubleClicked.connect(self._on_bin_table_double_clicked)
        # Also connect itemChanged for checkbox changes
        self.bins_table.itemChanged.connect(self._on_bin_item_changed)
        bin_layout.addWidget(self.bins_table)
        
        # Store spinbox references for Low/High columns (now integers)
        self._bin_low_spinboxes: List[QSpinBox] = []
        self._bin_high_spinboxes: List[QSpinBox] = []
        
        # Continuous bins checkbox
        self.continuous_bins_cb = QCheckBox("Continuous bins")
        self.continuous_bins_cb.setToolTip(
            "When enabled, adjusting bin boundaries automatically adjusts adjacent bins to keep them continuous"
        )
        bin_layout.addWidget(self.continuous_bins_cb)
        
        middle_layout.addWidget(bin_group, stretch=2)
        
        # Right: Preview - make it bigger
        self.slice_preview = SlicePreviewWidget(self)
        middle_layout.addWidget(self.slice_preview, stretch=2)  # Give it more space

        main_layout.addLayout(middle_layout)

        # Output directory
        out_row = QHBoxLayout()
        self.output_label = QLabel("Output directory: (none)")
        self.btn_select_output = QPushButton("Select Output Folder")
        self.btn_select_output.clicked.connect(self._on_select_output_dir)
        out_row.addWidget(self.output_label, stretch=1)
        out_row.addWidget(self.btn_select_output)
        main_layout.addLayout(out_row)

        # Filename prefix
        prefix_row = QHBoxLayout()
        prefix_row.addWidget(QLabel("Filename prefix:"))
        self.edit_prefix = QLineEdit("ct_layer")
        prefix_row.addWidget(self.edit_prefix)
        main_layout.addLayout(prefix_row)

        # Export options group
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        # File organization
        file_org_row = QHBoxLayout()
        file_org_row.addWidget(QLabel("File organization:"))
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItems(["Multiple files (one per bin)", "Single file (all bins combined)"])
        file_org_row.addWidget(self.export_mode_combo)
        file_org_row.addStretch()
        export_layout.addLayout(file_org_row)
        
        # Format (must be selected first to determine available options)
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        # Only PLY is currently implemented and working
        self.format_combo.addItems(["PLY"])
        self.format_combo.setToolTip(
            "PLY: Supports per-vertex colors and opacity\n"
            "Note: Other formats (OBJ, STL, GLTF) may be added in the future"
        )
        format_row.addWidget(self.format_combo)
        format_row.addStretch()
        export_layout.addLayout(format_row)
        
        # Format info label (updates based on selected format)
        self.format_info_label = QLabel("PLY supports colors and opacity")
        self.format_info_label.setStyleSheet("color: gray; font-style: italic;")
        export_layout.addWidget(self.format_info_label)
        
        # Connect format change to update available options
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        
        # Options checkboxes
        options_row = QHBoxLayout()
        self.export_colors_cb = QCheckBox("Export with colors")
        self.export_colors_cb.setChecked(True)
        self.export_opacity_cb = QCheckBox("Export with opacity")
        self.export_opacity_cb.setChecked(False)
        options_row.addWidget(self.export_colors_cb)
        options_row.addWidget(self.export_opacity_cb)
        options_row.addStretch()
        export_layout.addLayout(options_row)
        
        # Opacity value (only enabled if opacity checkbox is checked)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity (0.0-1.0):"))
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setValue(1.0)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setEnabled(False)
        self.export_opacity_cb.toggled.connect(self._on_opacity_checkbox_toggled)
        opacity_row.addWidget(self.opacity_spin)
        opacity_row.addStretch()
        export_layout.addLayout(opacity_row)
        
        main_layout.addWidget(export_group)
        
        # Export button
        self.btn_export_meshes = QPushButton("Export meshes")
        self.btn_export_meshes.clicked.connect(self.on_export_meshes_clicked)
        self.btn_export_meshes.setEnabled(False)
        main_layout.addWidget(self.btn_export_meshes)

    # ------------------------------------------------------------------ #
    # Format capabilities and UI updates
    # ------------------------------------------------------------------ #
    def _get_format_capabilities(self, format_name: str) -> dict[str, bool]:
        """
        Get capabilities for a given format.
        
        Format capabilities:
        - PLY: Supports per-vertex colors (RGB) and per-vertex opacity (alpha channel) ✓
        - OBJ: Supports colors via MTL files (not per-vertex), opacity via materials (not implemented)
        - STL: Does NOT support colors or opacity (geometry only)
        - GLTF/GLB: Supports colors and opacity (not yet implemented)
        - FBX: Supports colors and opacity (not yet implemented)
        """
        capabilities = {
            "PLY": {"colors": True, "opacity": True, "implemented": True},
            "OBJ": {"colors": False, "opacity": False, "implemented": False},  # Not implemented yet
            "STL": {"colors": False, "opacity": False, "implemented": False},  # Not implemented yet
            "GLTF": {"colors": True, "opacity": True, "implemented": False},  # Not implemented yet
            "GLB": {"colors": True, "opacity": True, "implemented": False},  # Not implemented yet
            "FBX": {"colors": True, "opacity": True, "implemented": False},  # Not implemented yet
        }
        return capabilities.get(format_name.upper(), {"colors": False, "opacity": False, "implemented": False})
    
    def _on_format_changed(self, format_name: str) -> None:
        """Update UI based on selected format capabilities."""
        caps = self._get_format_capabilities(format_name)
        
        # Update format info label
        info_parts = []
        if caps["colors"]:
            info_parts.append("per-vertex colors")
        if caps["opacity"]:
            info_parts.append("per-vertex opacity")
        
        if info_parts:
            info_text = f"{format_name} supports {', '.join(info_parts)}"
        else:
            info_text = f"{format_name} supports geometry only (no colors or opacity)"
        
        if not caps.get("implemented", False):
            info_text += " (not yet implemented)"
        
        self.format_info_label.setText(info_text)
        
        # Enable/disable checkboxes based on format capabilities
        self.export_colors_cb.setEnabled(caps["colors"] and caps.get("implemented", False))
        if not caps["colors"] or not caps.get("implemented", False):
            self.export_colors_cb.setChecked(False)
            if not caps.get("implemented", False):
                self.export_colors_cb.setToolTip(f"{format_name} format is not yet implemented")
            else:
                self.export_colors_cb.setToolTip(f"{format_name} does not support per-vertex colors")
        else:
            self.export_colors_cb.setToolTip(f"{format_name} supports per-vertex RGB colors")
        
        self.export_opacity_cb.setEnabled(caps["opacity"] and caps.get("implemented", False))
        if not caps["opacity"] or not caps.get("implemented", False):
            self.export_opacity_cb.setChecked(False)
            self.opacity_spin.setEnabled(False)
            if not caps.get("implemented", False):
                self.export_opacity_cb.setToolTip(f"{format_name} format is not yet implemented")
            else:
                self.export_opacity_cb.setToolTip(f"{format_name} does not support per-vertex opacity")
        else:
            self.export_opacity_cb.setToolTip(f"{format_name} supports per-vertex alpha channel")
            # Opacity checkbox state determines spinbox state
            self._on_opacity_checkbox_toggled(self.export_opacity_cb.isChecked())
    
    def _on_opacity_checkbox_toggled(self, checked: bool) -> None:
        """Enable/disable opacity spinbox based on checkbox state."""
        # Only enable if checkbox is checked AND format supports opacity AND is implemented
        format_name = self.format_combo.currentText()
        caps = self._get_format_capabilities(format_name)
        self.opacity_spin.setEnabled(
            checked and caps["opacity"] and caps.get("implemented", False)
        )
    
    # ------------------------------------------------------------------ #
    # Directory selection
    # ------------------------------------------------------------------ #
    def _on_select_processed_dir(self) -> None:
        # Start with default directory if available
        start_dir = str(self._default_processed_dir) if self._default_processed_dir else ""
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Select processed slices directory (or use default)",
            start_dir
        )
        if not folder:
            return
        self._processed_dir = Path(folder)
        self.processed_label.setText(f"Processed slices directory: {folder}")
    
    def set_default_processed_dir(self, path: Optional[Path]) -> None:
        """
        Set the default processed directory from preprocessing output.
        This will be used as the default input for meshing.
        """
        self._default_processed_dir = path
        if path is not None and path.exists():
            # Auto-set if not already set
            if self._processed_dir is None:
                self._processed_dir = path
                self.processed_label.setText(
                    f"Processed slices directory: {path} (default from preprocessing)"
                )
            else:
                # Update label to show default is available
                self.processed_label.setText(
                    f"Processed slices directory: {self._processed_dir}\n"
                    f"(Default available: {path})"
                )

    def _on_select_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not folder:
            return
        self._output_dir = Path(folder)
        self.output_label.setText(f"Output directory: {folder}")

    # ------------------------------------------------------------------ #
    # Helpers to build configs
    # ------------------------------------------------------------------ #
    def _collect_spacing(self) -> Tuple[float, float, float]:
        return (
            float(self.spin_spacing_z.value()),
            float(self.spin_spacing_y.value()),
            float(self.spin_spacing_x.value()),
        )

    def _collect_bins_from_table(self) -> List[IntensityBin]:
        bins: List[IntensityBin] = []
        rows = self.bins_table.rowCount()
        for row in range(rows):
            enabled_item = self.bins_table.item(row, 0)
            # Low and High are now spinboxes, not text items
            low_spin = self.bins_table.cellWidget(row, 1)
            high_spin = self.bins_table.cellWidget(row, 2)
            name_item = self.bins_table.item(row, 3)
            color_item = self.bins_table.item(row, 4)

            enabled = True
            if enabled_item is not None:
                enabled = enabled_item.checkState() == Qt.Checked

            try:
                # Round to integers since we work with 1-255
                low = int(round(low_spin.value())) if low_spin is not None else 0
                high = int(round(high_spin.value())) if high_spin is not None else 0
            except Exception:
                continue

            name = name_item.text() if name_item is not None else ""
            
            # Get color from item background or data
            color = None
            if color_item is not None:
                bg_brush = color_item.background()
                # QTableWidgetItem.background() returns a QBrush, not QColor
                # Get the color from the brush
                if bg_brush.style() != Qt.BrushStyle.NoBrush:
                    bg_color = bg_brush.color()
                    if bg_color.isValid() and bg_color != QColor(255, 255, 255):  # Not default white
                        r = bg_color.red() / 255.0
                        g = bg_color.green() / 255.0
                        b = bg_color.blue() / 255.0
                        color = (r, g, b)
            
            bins.append(
                IntensityBin(
                    index=row,
                    enabled=enabled,
                    low=low,
                    high=high,
                    name=name or f"bin_{row}",
                    color=color,
                )
            )
        return bins

    def _populate_default_bins(self, vmin: float, vmax: float) -> None:
        """Create evenly spaced bins according to spin_num_bins and current range."""
        n_bins = int(self.spin_num_bins.value())
        # Round to integers for bin edges
        vmin_int = int(round(vmin))
        vmax_int = int(round(vmax))
        edges = np.linspace(vmin_int, vmax_int, n_bins + 1)
        self.bins_table.setRowCount(n_bins)
        
        # Clear spinbox lists
        self._bin_low_spinboxes.clear()
        self._bin_high_spinboxes.clear()

        # Generate colors for bins
        import colorsys
        for i in range(n_bins):
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            enabled_item.setCheckState(Qt.Checked)

            # Create spinboxes for Low and High (integers)
            low_spin = QSpinBox()
            low_spin.setRange(int(round(vmin)), int(round(vmax)))
            low_spin.setValue(int(round(edges[i])))
            low_spin.setSingleStep(1)
            low_spin.valueChanged.connect(lambda val, row=i: self._on_bin_spin_changed(row, 'low', val))
            self._bin_low_spinboxes.append(low_spin)
            
            high_spin = QSpinBox()
            high_spin.setRange(int(round(vmin)), int(round(vmax)))
            high_spin.setValue(int(round(edges[i + 1])))
            high_spin.setSingleStep(1)
            high_spin.valueChanged.connect(lambda val, row=i: self._on_bin_spin_changed(row, 'high', val))
            self._bin_high_spinboxes.append(high_spin)
            
            name_item = QTableWidgetItem(f"bin_{i + 1}")
            
            # Color item with unique color
            hue = (i / max(n_bins, 1)) * 0.8
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            color = QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            color_item = QTableWidgetItem("")
            color_item.setBackground(color)
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)  # Not editable text
            color_item.setToolTip("Double-click to change color")

            self.bins_table.setItem(i, 0, enabled_item)
            self.bins_table.setCellWidget(i, 1, low_spin)
            self.bins_table.setCellWidget(i, 2, high_spin)
            self.bins_table.setItem(i, 3, name_item)
            self.bins_table.setItem(i, 4, color_item)
        
        self._update_bin_visualization()
    
    def _on_bin_spin_changed(self, row: int, which: str, value: int) -> None:
        """Called when a bin Low/High spinbox value changes."""
        # Handle continuous bins mode
        if self.continuous_bins_cb.isChecked():
            self._adjust_adjacent_bins(row, which, value)
        self._update_bin_visualization()
    
    def _adjust_adjacent_bins(self, row: int, which: str, value: int) -> None:
        """Adjust adjacent bins to maintain continuity when continuous mode is enabled."""
        total_rows = self.bins_table.rowCount()
        
        if which == 'low':
            # When low boundary changes, adjust previous bin's high
            if row > 0:
                prev_high_spin = self.bins_table.cellWidget(row - 1, 2)
                if prev_high_spin is not None:
                    prev_high_spin.blockSignals(True)
                    prev_high_spin.setValue(value)
                    prev_high_spin.blockSignals(False)
        elif which == 'high':
            # When high boundary changes, adjust next bin's low
            if row < total_rows - 1:
                next_low_spin = self.bins_table.cellWidget(row + 1, 1)
                if next_low_spin is not None:
                    next_low_spin.blockSignals(True)
                    next_low_spin.setValue(value)
                    next_low_spin.blockSignals(False)
    
    def _on_histogram_bin_moved(self, bin_table_row: int, boundary_type: str, new_value: float) -> None:
        """Called when a bin boundary is dragged on the histogram."""
        # Round to integer
        new_value_int = int(round(new_value))
        
        # Update the corresponding spinbox in the table
        # bin_table_row is the actual table row index (from bin.index)
        if 0 <= bin_table_row < self.bins_table.rowCount():
            col = 1 if boundary_type == 'low' else 2
            spinbox = self.bins_table.cellWidget(bin_table_row, col)
            if spinbox is not None:
                # Temporarily disconnect to avoid recursive updates
                spinbox.blockSignals(True)
                spinbox.setValue(new_value_int)
                spinbox.blockSignals(False)
                
                # Handle continuous bins mode
                if self.continuous_bins_cb.isChecked():
                    self._adjust_adjacent_bins(bin_table_row, boundary_type, new_value_int)
                
                # Update preview only (skip histogram to avoid update loop)
                self._update_bin_visualization_no_histogram()

    # ------------------------------------------------------------------ #
    # Volume loading + histogram (threaded)
    # ------------------------------------------------------------------ #
    def load_volume_and_histogram(self) -> None:
        if self._processed_dir is None:
            self.status.show_error("Please select a processed slices directory first.")
            return

        folder = self._processed_dir

        # Precompute list of slice files so we know how many there are
        paths = images.list_slice_files(folder)
        total_slices = len(paths)
        if total_slices == 0:
            self.status.show_error("No image slices found in the selected directory.")
            return

        def task(progress_cb) -> Tuple[np.ndarray, np.ndarray, float, float]:
            # Load all RGB slices with progress
            slices = []
            for i, p in enumerate(paths, start=1):
                arr = images.load_image_rgb(p)
                slices.append(arr)
                progress_cb(i, total_slices)

            volume = np.stack(slices, axis=0)  # (Z, Y, X, 3)

            # Grayscale intensities
            grayscale = volume.mean(axis=-1)
            intensities = grayscale.ravel()

            # Exclude zeros from histogram / range to avoid the huge spike
            non_zero = intensities[intensities > 0]

            if non_zero.size > 0:
                hist_values = non_zero
            else:
                # Fallback: if everything is zero (strange, but safe)
                hist_values = intensities

            vmin = float(hist_values.min())
            vmax = float(hist_values.max())
            return volume, hist_values, vmin, vmax

        worker = FunctionWorker(task, with_progress=True)

        def on_success(result: object) -> None:
            volume, hist_values, vmin, vmax = result  # type: ignore[misc]
            self._volume = volume
            self._current_range = (vmin, vmax)

            self.spin_int_min.setValue(int(round(vmin)))
            self.spin_int_max.setValue(int(round(vmax)))

            # Update 3D histogram
            self.histogram.set_histogram_3d(volume, n_bins=256, value_range=(vmin, vmax))
            self._populate_default_bins(vmin, vmax)
            
            # Update preview
            self.slice_preview.set_volume(volume)
            self.slice_preview.set_bins(self._collect_bins_from_table())
            
            self.btn_export_meshes.setEnabled(True)

        self.status.run_threaded_with_progress(
            worker,
            "Loading volume...",
            on_success=on_success,
        )

    # ------------------------------------------------------------------ #
    # Export meshes
    # ------------------------------------------------------------------ #
    def on_export_meshes_clicked(self) -> None:
        """Export meshes with bin colors and optional opacity."""
        if self._volume is None:
            self.status.show_error("Please load a volume first.")
            return
        if self._output_dir is None:
            self.status.show_error("Please select an output directory.")
            return
        
        bins = [b for b in self._collect_bins_from_table() if b.enabled]
        if not bins:
            self.status.show_error("No enabled intensity bins to export.")
            return
        
        # Get export options from UI
        export_mode_text = self.export_mode_combo.currentText()
        export_mode = "separate" if "Multiple" in export_mode_text else "combined"
        
        # Get opacity if enabled
        opacity = None
        if self.export_opacity_cb.isChecked():
            opacity = float(self.opacity_spin.value())
            if opacity >= 1.0:
                opacity = None  # Don't add alpha channel if fully opaque
        
        filename_prefix = self.edit_prefix.text().strip() or "ct_layer"
        
        spacing = self._collect_spacing()
        vmin = int(self.spin_int_min.value())
        vmax = int(self.spin_int_max.value())
        
        cfg = MeshingConfig(
            spacing=spacing,
            min_intensity=vmin,
            max_intensity=vmax,
        )
        cfg.opacity = opacity  # Add opacity to config
        
        volume = self._volume
        output_dir = self._output_dir
        total_bins = len(bins)
        
        # Get format and capabilities
        export_format = self.format_combo.currentText().upper()
        caps = self._get_format_capabilities(export_format)
        
        # Validate format is implemented
        if not caps.get("implemented", False):
            self.status.show_error(
                f"Format '{export_format}' is not yet implemented. "
                "Currently only PLY format is supported."
            )
            return
        
        # Create a custom worker with phase-aware progress for export
        class ExportWorker(WorkerBase):
            phase_progress = Signal(str, int, int, int)  # phase, current, phase_total, overall_total
            
            def run(self) -> None:  # type: ignore[override]
                def phase_cb(phase: str, current: int, phase_total: int, overall_total: int) -> None:
                    self.phase_progress.emit(phase, current, phase_total, overall_total)
                    self.progress.emit(current, overall_total)
                
                # Apply colors option: if not exporting colors or format doesn't support it, remove colors
                export_bins = bins
                if not self.export_colors_cb.isChecked() or not caps["colors"]:
                    # Create bins without colors
                    from ct23d.core.models import IntensityBin
                    export_bins = [
                        IntensityBin(
                            index=b.index,
                            enabled=b.enabled,
                            low=b.low,
                            high=b.high,
                            name=b.name,
                            color=None,  # Remove color
                        )
                        for b in bins
                    ]
                
                # Export uses the same meshing functions, so it will get phase progress
                export_core.export_bins_to_meshes(
                    volume=volume,
                    bins=export_bins,
                    config=cfg,
                    output_dir=output_dir,
                    filename_prefix=filename_prefix,
                    export_mode=export_mode,
                    format_name=export_format,
                    opacity=opacity if caps["opacity"] else None,  # Only pass opacity if format supports it
                    phase_progress_callback=phase_cb,
                )
                self.finished.emit(total_bins)
        
        worker = ExportWorker()
        
        def on_success(result: object) -> None:
            n = int(result)
            if export_mode == "combined":
                self.status.show_info(f"Export complete: 1 combined file with {n} bins.")
            else:
                self.status.show_info(f"Export complete: {n} separate files generated.")
        
        self.status.run_threaded_with_progress(
            worker,
            "Exporting meshes...",
            on_success=on_success,
        )
    
    # ------------------------------------------------------------------ #
    # Bin management
    # ------------------------------------------------------------------ #
    def _on_add_bin(self) -> None:
        """Add a new bin to the table."""
        vmin = int(self.spin_int_min.value())
        vmax = int(self.spin_int_max.value())
        
        # Add new row
        row = self.bins_table.rowCount()
        self.bins_table.insertRow(row)
        
        # Create default bin in the middle of the range
        mid = (vmin + vmax) / 2
        bin_width = (vmax - vmin) / 10  # Default 10% of range
        
        enabled_item = QTableWidgetItem()
        enabled_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        enabled_item.setCheckState(Qt.Checked)
        
        # Create spinboxes for Low and High (integers)
        low_spin = QSpinBox()
        low_spin.setRange(int(round(vmin)), int(round(vmax)))
        low_spin.setValue(int(round(mid - bin_width/2)))
        low_spin.setSingleStep(1)
        low_spin.valueChanged.connect(lambda val, r=row: self._on_bin_spin_changed(r, 'low', val))
        if row >= len(self._bin_low_spinboxes):
            self._bin_low_spinboxes.append(low_spin)
        else:
            self._bin_low_spinboxes.insert(row, low_spin)
        
        high_spin = QSpinBox()
        high_spin.setRange(int(round(vmin)), int(round(vmax)))
        high_spin.setValue(int(round(mid + bin_width/2)))
        high_spin.setSingleStep(1)
        high_spin.valueChanged.connect(lambda val, r=row: self._on_bin_spin_changed(r, 'high', val))
        if row >= len(self._bin_high_spinboxes):
            self._bin_high_spinboxes.append(high_spin)
        else:
            self._bin_high_spinboxes.insert(row, high_spin)
        
        name_item = QTableWidgetItem(f"bin_{row + 1}")
        
        # Generate unique color
        import colorsys
        hue = (row / max(row + 1, 1)) * 0.8
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        color_item = QTableWidgetItem("")
        color_item.setBackground(color)
        color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
        color_item.setToolTip("Double-click to change color")
        
        self.bins_table.setItem(row, 0, enabled_item)
        self.bins_table.setCellWidget(row, 1, low_spin)
        self.bins_table.setCellWidget(row, 2, high_spin)
        self.bins_table.setItem(row, 3, name_item)
        self.bins_table.setItem(row, 4, color_item)
        
        self._update_bin_visualization()
    
    def _on_delete_bin(self) -> None:
        """Delete the selected bin(s)."""
        selected_rows = set()
        for item in self.bins_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            # If no selection, delete the last row
            if self.bins_table.rowCount() > 0:
                selected_rows = {self.bins_table.rowCount() - 1}
        
        # Delete rows in reverse order to maintain indices
        for row in sorted(selected_rows, reverse=True):
            self.bins_table.removeRow(row)
            # Remove from spinbox lists
            if row < len(self._bin_low_spinboxes):
                self._bin_low_spinboxes.pop(row)
            if row < len(self._bin_high_spinboxes):
                self._bin_high_spinboxes.pop(row)
        
        self._update_bin_visualization()
    
    def _on_bin_table_changed(self, row: int, column: int) -> None:
        """Called when a bin table cell is changed."""
        # Update visualization when bin values change
        # Note: Low/High columns (1, 2) are now spinboxes, handled by _on_bin_spin_changed
        if column == 3:  # Name changed
            self._update_bin_visualization()
        elif column == 0:  # Enabled changed
            self._update_bin_visualization()
        elif column == 4:  # Color changed
            self._update_bin_visualization()
    
    def _on_bin_item_changed(self, item: QTableWidgetItem) -> None:
        """Called when a bin table item is changed (including checkboxes)."""
        # Update visualization when enabled state changes
        if item.column() == 0:  # Enabled column
            self._update_bin_visualization()
    
    def _on_bin_table_double_clicked(self, row: int, column: int) -> None:
        """Called when a bin table cell is double-clicked."""
        if column == 4:  # Color column
            color_item = self.bins_table.item(row, 4)
            if color_item is None:
                return
            
            # Get current color from brush
            bg_brush = color_item.background()
            if bg_brush.style() != Qt.BrushStyle.NoBrush:
                current_color = bg_brush.color()
            else:
                current_color = QColor(128, 128, 128)  # Default gray
            
            if not current_color.isValid():
                current_color = QColor(128, 128, 128)  # Default gray
            
            # Open color picker
            new_color = QColorDialog.getColor(current_color, self, "Choose bin color")
            if new_color.isValid():
                color_item.setBackground(new_color)
                # Update the bin color in the model
                self._update_bin_visualization()
    
    def _update_bin_visualization(self) -> None:
        """Update histogram and preview with current bins."""
        bins = self._collect_bins_from_table()
        
        # Update histogram with bin boundaries
        self.histogram.update_bins(bins)
        
        # Update preview with bin colors - always update if volume is loaded
        if self._volume is not None:
            self.slice_preview.set_bins(bins)
            # Force preview to update
            self.slice_preview._update_preview()
    
    def _update_bin_visualization_no_histogram(self) -> None:
        """Update preview only (skip histogram to avoid update loop when dragging)."""
        bins = self._collect_bins_from_table()
        
        # Update preview with bin colors - always update if volume is loaded
        if self._volume is not None:
            self.slice_preview.set_bins(bins)
            # Force preview to update
            self.slice_preview._update_preview()
    


class ExportDialog(QDialog):
    """Dialog for export options: format, file organization, opacity."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Meshes")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        # Format (currently only PLY, but can be extended)
        format_label = QLabel("Format:")
        format_info = QLabel("PLY (Polygon File Format)")
        format_info.setStyleSheet("color: gray;")
        format_row = QHBoxLayout()
        format_row.addWidget(format_label)
        format_row.addWidget(format_info)
        format_row.addStretch()
        form.addRow(format_row)
        
        # File organization
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItems(["Multiple files (one per bin)", "Single file (all bins combined)"])
        self.export_mode_combo.setCurrentIndex(0)
        self.export_mode_combo.setToolTip(
            "Multiple files: One PLY file per bin (e.g., ct_export_bin_00_1.0_43.33.ply)\n"
            "Single file: All bins combined into one PLY file (e.g., ct_export_combined.ply)"
        )
        form.addRow("File organization:", self.export_mode_combo)
        
        # Opacity
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setValue(1.0)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setToolTip("Opacity value (0.0 = transparent, 1.0 = opaque)")
        form.addRow("Opacity (0.0-1.0):", self.opacity_spin)
        
        # Filename prefix
        self.filename_edit = QLineEdit("ct_export")
        form.addRow("Filename prefix:", self.filename_edit)
        
        # Info label explaining difference from Generate
        info_label = QLabel(
            "Note: 'Generate meshes' uses default settings.\n"
            "'Export meshes' allows you to customize format, file organization, and opacity."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-style: italic; padding: 10px;")
        form.addRow("", info_label)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_export_mode(self) -> str:
        """Get selected export mode."""
        text = self.export_mode_combo.currentText()
        if "Multiple" in text:
            return "separate"
        else:
            return "combined"
    
    def get_opacity(self) -> Optional[float]:
        """Get opacity value, or None if fully opaque."""
        opacity = float(self.opacity_spin.value())
        if opacity >= 1.0:
            return None  # Don't add alpha channel if fully opaque
        return opacity
    
    def get_filename_prefix(self) -> str:
        """Get filename prefix."""
        return self.filename_edit.text().strip()
