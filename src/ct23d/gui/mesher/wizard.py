from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer
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
        self._num_slices: int = 0  # Track number of slices for Z height calculation
        self._histogram_values: Optional[np.ndarray] = None  # Store histogram values for auto-bin
        self._is_dicom: bool = False  # Track if volume is DICOM (uint16) or normal (uint8)

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Top: processed directory selector (button on left, label on right)
        top_row = QHBoxLayout()
        top_row.setSpacing(8)  # Small spacing between button and label
        self.btn_select_processed = QPushButton("Select Folder (or use default)")
        self.btn_select_processed.setMaximumWidth(180)  # Smaller button
        self.btn_select_processed.clicked.connect(self._on_select_processed_dir)
        top_row.addWidget(self.btn_select_processed)
        self.processed_label = QLabel("Processed slices directory: (none)")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        top_row.addWidget(self.processed_label)
        top_row.addStretch()  # Push everything to the left
        main_layout.addLayout(top_row)

        # 3D Histogram + bin table + preview
        middle_layout = QHBoxLayout()
        
        # Left: Histogram (tabbed view)
        histogram_container = QVBoxLayout()
        histogram_container.setContentsMargins(0, 50, 0, 0)  # Add top margin to lower it, giving space for Patient Info
        
        from ct23d.gui.mesher.histogram_tabbed_view import HistogramTabbedView
        self.histogram = HistogramTabbedView(self)
        # Connect bin boundary changes to update table
        self.histogram.set_bin_boundary_callback(self._on_histogram_bin_moved)
        # Connect range line changes to update auto-bin controls
        self.histogram.set_range_line_callback(self._on_range_line_moved)
        histogram_container.addWidget(self.histogram)
        
        histogram_widget = QWidget()
        histogram_widget.setLayout(histogram_container)
        middle_layout.addWidget(histogram_widget, stretch=3)
        
        # Middle: Bin table with controls
        # Wrap in a container with spacer to align top with histogram/preview content (below their headers)
        bin_container = QWidget()
        bin_container_layout = QVBoxLayout(bin_container)
        bin_container_layout.setContentsMargins(0, 50, 0, 0)  # Add top margin to align with histogram/preview
        bin_container_layout.setSpacing(0)
        
        bin_group = QGroupBox("Intensity Bins")
        bin_layout = QVBoxLayout(bin_group)
        
        # Auto-bin section
        auto_bin_group = QGroupBox("Auto-bin")
        auto_bin_layout = QVBoxLayout(auto_bin_group)
        
        # Min/Max intensity range with Visualize checkbox
        min_max_row = QHBoxLayout()
        min_max_row.addWidget(QLabel("Intensity range:"))
        self.auto_bin_min_spin = QSpinBox()
        self.auto_bin_min_spin.setRange(1, 255)
        self.auto_bin_min_spin.setValue(1)
        self.auto_bin_min_spin.setToolTip("Minimum intensity for auto-bin range (draggable on histogram when visualized)")
        self.auto_bin_min_spin.valueChanged.connect(self._on_auto_bin_range_changed)
        min_max_row.addWidget(self.auto_bin_min_spin)
        min_max_row.addWidget(QLabel("to"))
        self.auto_bin_max_spin = QSpinBox()
        self.auto_bin_max_spin.setRange(1, 255)
        self.auto_bin_max_spin.setValue(255)
        self.auto_bin_max_spin.setToolTip("Maximum intensity for auto-bin range (draggable on histogram when visualized)")
        self.auto_bin_max_spin.valueChanged.connect(self._on_auto_bin_range_changed)
        min_max_row.addWidget(self.auto_bin_max_spin)
        self.visualize_range_cb = QCheckBox("Visualize")
        self.visualize_range_cb.setToolTip("Show range lines (min/max) on histogram graphs")
        self.visualize_range_cb.setChecked(False)
        self.visualize_range_cb.toggled.connect(self._on_visualize_range_toggled)
        min_max_row.addWidget(self.visualize_range_cb)
        min_max_row.addStretch()
        auto_bin_layout.addLayout(min_max_row)
        
        # Number of bins
        num_bins_row = QHBoxLayout()
        num_bins_row.addWidget(QLabel("Number of bins:"))
        self.auto_bin_count_spin = QSpinBox()
        self.auto_bin_count_spin.setRange(2, 20)
        self.auto_bin_count_spin.setValue(6)
        self.auto_bin_count_spin.setToolTip("Number of bins to generate")
        num_bins_row.addWidget(self.auto_bin_count_spin)
        num_bins_row.addStretch()
        auto_bin_layout.addLayout(num_bins_row)
        
        # Uniformity parameter
        uniformity_row = QHBoxLayout()
        uniformity_row.addWidget(QLabel("Uniformity:"))
        self.auto_bin_uniformity_spin = QDoubleSpinBox()
        self.auto_bin_uniformity_spin.setRange(0.0, 1.0)
        self.auto_bin_uniformity_spin.setDecimals(2)
        self.auto_bin_uniformity_spin.setSingleStep(0.1)
        self.auto_bin_uniformity_spin.setValue(0.0)
        self.auto_bin_uniformity_spin.setToolTip(
            "0.0 = auto-bins has full control based on intensity distribution\n"
            "1.0 = uniform bins between min/max values"
        )
        uniformity_row.addWidget(self.auto_bin_uniformity_spin)
        uniformity_row.addStretch()
        auto_bin_layout.addLayout(uniformity_row)
        
        # Auto-bin button
        auto_bin_button_row = QHBoxLayout()
        self.btn_apply_auto_bins = QPushButton("Apply Auto-bins")
        self.btn_apply_auto_bins.clicked.connect(self._on_apply_auto_bins)
        self.btn_apply_auto_bins.setEnabled(False)
        auto_bin_button_row.addWidget(self.btn_apply_auto_bins)
        auto_bin_button_row.addStretch()
        auto_bin_layout.addLayout(auto_bin_button_row)
        
        bin_layout.addWidget(auto_bin_group)
        
        # Add button
        bin_buttons = QHBoxLayout()
        self.btn_add_bin = QPushButton("Add Bin")
        self.btn_add_bin.clicked.connect(self._on_add_bin)
        bin_buttons.addWidget(self.btn_add_bin)
        bin_buttons.addStretch()
        bin_layout.addLayout(bin_buttons)
        
        self.bins_table = QTableWidget(0, 6, self)
        self.bins_table.setHorizontalHeaderLabels(["Enabled", "Low", "High", "Name", "Color", "Delete"])
        self.bins_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.bins_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.bins_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
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
        self.continuous_bins_cb.setChecked(True)  # Enabled by default
        self.continuous_bins_cb.setToolTip(
            "When enabled, adjusting bin boundaries automatically adjusts adjacent bins to keep them continuous"
        )
        bin_layout.addWidget(self.continuous_bins_cb)
        
        # Add bin group to container - make it stretch to fill full height
        bin_container_layout.addWidget(bin_group, stretch=1)  # Stretch to fill available height
        # Don't add stretch here - we want bins box to fill all available space down to bottom
        
        middle_layout.addWidget(bin_container, stretch=2)
        
        # Right: Preview - make it bigger, add top margin to lower it
        preview_container = QWidget()
        preview_container_layout = QVBoxLayout(preview_container)
        preview_container_layout.setContentsMargins(0, 50, 0, 0)  # Add top margin to lower it, giving space for Patient Info
        self.slice_preview = SlicePreviewWidget(self)
        preview_container_layout.addWidget(self.slice_preview)
        middle_layout.addWidget(preview_container, stretch=2)  # Give it more space

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

        # Options row: Mesh Processing Options (left) and Export Options (right)
        options_row = QHBoxLayout()
        
        # Mesh processing options group (left side)
        processing_group = QGroupBox("Mesh Processing Options")
        processing_layout = QVBoxLayout(processing_group)
        
        # Spacing (voxel size in mm)
        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("Voxel spacing (mm):"))
        spacing_row.addWidget(QLabel("Z:"))
        self.spin_spacing_z = QDoubleSpinBox()
        self.spin_spacing_z.setRange(0.0, 100.0)
        self.spin_spacing_z.setDecimals(2)
        self.spin_spacing_z.setSingleStep(0.1)
        self.spin_spacing_z.setValue(1.0)
        self.spin_spacing_z.setToolTip("Slice thickness in millimeters (Z direction)")
        self.spin_spacing_z.valueChanged.connect(self._update_z_height)  # Update Z height when changed
        spacing_row.addWidget(self.spin_spacing_z)
        
        # Z height calculation label (number of slices × voxel spacing)
        self.z_height_label = QLabel("Z height: —")
        self.z_height_label.setStyleSheet("font-weight: bold;")
        spacing_row.addWidget(QLabel("|"))
        spacing_row.addWidget(self.z_height_label)
        
        spacing_row.addWidget(QLabel("Y:"))
        self.spin_spacing_y = QDoubleSpinBox()
        self.spin_spacing_y.setRange(0.01, 100.0)
        self.spin_spacing_y.setDecimals(2)
        self.spin_spacing_y.setSingleStep(0.1)
        self.spin_spacing_y.setValue(1.0)
        self.spin_spacing_y.setToolTip("Pixel size in Y direction (millimeters)")
        spacing_row.addWidget(self.spin_spacing_y)
        
        spacing_row.addWidget(QLabel("X:"))
        self.spin_spacing_x = QDoubleSpinBox()
        self.spin_spacing_x.setRange(0.01, 100.0)
        self.spin_spacing_x.setDecimals(2)
        self.spin_spacing_x.setSingleStep(0.1)
        self.spin_spacing_x.setValue(1.0)
        self.spin_spacing_x.setToolTip("Pixel size in X direction (millimeters)")
        spacing_row.addWidget(self.spin_spacing_x)
        spacing_row.addStretch()
        processing_layout.addLayout(spacing_row)
        
        # Component filtering option
        component_row = QHBoxLayout()
        self.enable_component_filtering_cb = QCheckBox("Enable component filtering")
        self.enable_component_filtering_cb.setChecked(True)
        self.enable_component_filtering_cb.setToolTip(
            "Remove small isolated regions as noise. "
            "Uncheck to preserve all details (may include noise)."
        )
        component_row.addWidget(self.enable_component_filtering_cb)
        component_explanation = QLabel("(Removes small isolated regions as noise)")
        component_explanation.setStyleSheet("color: gray; font-style: italic;")
        component_explanation.setToolTip(
            "Component filtering removes small connected regions that are likely noise. "
            "Uncheck this to preserve all details, including potentially noisy small features."
        )
        component_row.addWidget(component_explanation)
        component_row.addStretch()
        processing_layout.addLayout(component_row)
        
        # Min component size (for noise filtering)
        component_size_row = QHBoxLayout()
        component_size_row.addWidget(QLabel("Min component size:"))
        self.spin_min_component_size = QSpinBox()
        self.spin_min_component_size.setRange(0, 1000000)
        self.spin_min_component_size.setSingleStep(100)
        self.spin_min_component_size.setValue(5000)
        self.spin_min_component_size.setToolTip(
            "Minimum connected component size (in voxels) to keep. "
            "Smaller components are removed as noise. "
            "Lower values preserve more detail but may keep noise. "
            "Higher values remove more noise but may lose small features. "
            "Default: 5000. For minimal loss, use 0-1000."
        )
        component_size_row.addWidget(self.spin_min_component_size)
        component_size_row.addStretch()
        processing_layout.addLayout(component_size_row)
        
        # Connect checkbox to enable/disable spinbox
        self.enable_component_filtering_cb.toggled.connect(
            lambda checked: self.spin_min_component_size.setEnabled(checked)
        )
        self.spin_min_component_size.setEnabled(True)  # Enabled by default since checkbox is checked

        # Smoothing option
        smoothing_row = QHBoxLayout()
        self.enable_smoothing_cb = QCheckBox("Enable Gaussian smoothing")
        self.enable_smoothing_cb.setChecked(True)
        self.enable_smoothing_cb.setToolTip(
            "Blur edges for smoother surfaces. "
            "WARNING: Can remove small objects and thin connections! "
            "Uncheck if you're experiencing surface loss."
        )
        smoothing_row.addWidget(self.enable_smoothing_cb)
        smoothing_explanation = QLabel("(Blurs edges for smoother surfaces - may remove small objects)")
        smoothing_explanation.setStyleSheet("color: gray; font-style: italic;")
        smoothing_explanation.setToolTip(
            "Gaussian smoothing can remove small isolated objects and break thin connections. "
            "If you're seeing surface loss, try disabling this option."
        )
        smoothing_row.addWidget(smoothing_explanation)
        smoothing_row.addStretch()
        processing_layout.addLayout(smoothing_row)
        
        # Smoothing sigma (for noise reduction)
        smoothing_sigma_row = QHBoxLayout()
        smoothing_sigma_row.addWidget(QLabel("Smoothing sigma:"))
        self.spin_smoothing_sigma = QDoubleSpinBox()
        self.spin_smoothing_sigma.setRange(0.0, 10.0)
        self.spin_smoothing_sigma.setDecimals(2)
        self.spin_smoothing_sigma.setSingleStep(0.1)
        self.spin_smoothing_sigma.setValue(1.0)
        self.spin_smoothing_sigma.setToolTip(
            "Gaussian smoothing strength (in voxels). "
            "Lower values preserve more detail but may create jagged surfaces. "
            "Higher values create smoother surfaces but may lose small details. "
            "Default: 1.0. For minimal loss, use 0.0-0.5."
        )
        smoothing_sigma_row.addWidget(self.spin_smoothing_sigma)
        smoothing_sigma_row.addStretch()
        processing_layout.addLayout(smoothing_sigma_row)
        
        # Connect checkbox to enable/disable spinbox
        self.enable_smoothing_cb.toggled.connect(
            lambda checked: self.spin_smoothing_sigma.setEnabled(checked)
        )
        self.spin_smoothing_sigma.setEnabled(True)  # Enabled by default since checkbox is checked
        
        options_row.addWidget(processing_group, stretch=1)

        # Export options group (right side)
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        # File organization
        file_org_row = QHBoxLayout()
        file_org_row.addWidget(QLabel("File organization:"))
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItems(["Multiple files (one per bin)", "Single file (all bins combined)"])
        # File size calculation is now manual only (button click)
        file_org_row.addWidget(self.export_mode_combo)
        file_org_row.addStretch()
        export_layout.addLayout(file_org_row)
        
        # Format (must be selected first to determine available options)
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        # PLY and STL are implemented
        self.format_combo.addItems(["PLY", "STL"])
        self.format_combo.setToolTip(
            "PLY: Supports per-vertex colors and opacity\n"
            "STL: Geometry only (no colors/opacity), widely compatible"
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
        # File size calculation is now manual only (button click)
        
        # Options checkboxes
        export_options_row = QHBoxLayout()
        self.export_colors_cb = QCheckBox("Export with colors")
        self.export_colors_cb.setChecked(True)
        # File size calculation is now manual only (button click)
        self.export_opacity_cb = QCheckBox("Export with opacity")
        self.export_opacity_cb.setChecked(False)
        # File size calculation is now manual only (button click)
        export_options_row.addWidget(self.export_colors_cb)
        export_options_row.addWidget(self.export_opacity_cb)
        export_options_row.addStretch()
        export_layout.addLayout(export_options_row)
        
        # Format-specific options (STL binary/ASCII)
        self.stl_format_row = QHBoxLayout()
        self.stl_format_row.addWidget(QLabel("STL format:"))
        self.stl_binary_cb = QCheckBox("Binary (faster, smaller)")
        self.stl_binary_cb.setChecked(True)
        self.stl_binary_cb.setToolTip(
            "Binary STL: Faster to write/read, smaller file size (default)\n"
            "ASCII STL: Human-readable, larger file size"
        )
        # File size calculation is now manual only (button click)
        self.stl_format_row.addWidget(self.stl_binary_cb)
        self.stl_format_row.addStretch()
        export_layout.addLayout(self.stl_format_row)
        # Initially hide STL options (only show for STL format)
        self.stl_format_row.itemAt(0).widget().setVisible(False)
        self.stl_binary_cb.setVisible(False)
        
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
        
        # File size estimation
        size_row = QHBoxLayout()
        self.calc_size_btn = QPushButton("Calculate approximate file size")
        self.calc_size_btn.setToolTip(
            "Estimate the total file size based on current settings.\n"
            "For multiple files: shows the sum of all files.\n"
            "For single file: shows the size of that file."
        )
        self.calc_size_btn.clicked.connect(self._on_calculate_file_size_clicked)
        size_row.addWidget(self.calc_size_btn)
        self.size_label = QLabel("")
        self.size_label.setStyleSheet("color: white; font-weight: bold;")
        size_row.addWidget(self.size_label)
        size_row.addStretch()
        export_layout.addLayout(size_row)
        
        options_row.addWidget(export_group, stretch=1)
        
        main_layout.addLayout(options_row)
        
        # Export button
        self.btn_export_meshes = QPushButton("Export meshes")
        self.btn_export_meshes.clicked.connect(self.on_export_meshes_clicked)
        self.btn_export_meshes.setEnabled(False)
        self.btn_export_meshes.setMaximumWidth(300)  # Smaller width
        self.btn_export_meshes.setStyleSheet("font-weight: bold; font-size: 14pt;")  # Bigger, bold font
        export_btn_layout = QHBoxLayout()
        export_btn_layout.addStretch()
        export_btn_layout.addWidget(self.btn_export_meshes)
        export_btn_layout.addStretch()
        main_layout.addLayout(export_btn_layout)

    # ------------------------------------------------------------------ #
    # File size estimation
    # ------------------------------------------------------------------ #
    def _on_calculate_file_size_clicked(self) -> None:
        """Handle button click to calculate file size in background thread."""
        if self._volume is None:
            self.size_label.setText("(Load volume first)")
            self.size_label.setStyleSheet("color: red; font-weight: normal;")
            return
        
        # Get current settings for the worker
        bins = [b for b in self._collect_bins_from_table() if b.enabled]
        if not bins:
            self.size_label.setText("(No enabled bins)")
            self.size_label.setStyleSheet("color: red; font-weight: normal;")
            return
        
        export_mode_text = self.export_mode_combo.currentText()
        export_mode = "separate" if "Multiple" in export_mode_text else "combined"
        format_name = self.format_combo.currentText().upper()
        export_colors = self.export_colors_cb.isChecked() and format_name == "PLY"
        export_opacity = self.export_opacity_cb.isChecked() and format_name == "PLY"
        stl_binary = self.stl_binary_cb.isChecked() if format_name == "STL" else True
        
        # Create worker function
        volume = self._volume  # Capture volume for worker thread
        def calculate_task(progress_cb) -> str:
            return self._calculate_file_size_impl(
                volume, bins, export_mode, format_name, export_colors, export_opacity, stl_binary, progress_cb
            )
        
        # Run in background thread
        from ct23d.gui.workers import FunctionWorker
        worker = FunctionWorker(calculate_task, with_progress=True)
        
        def on_success(result: str) -> None:
            self.size_label.setText(result)
            self.size_label.setStyleSheet("color: white; font-weight: bold;")
        
        def on_error(msg: str) -> None:
            self.size_label.setText(f"(Error: {msg})")
            self.size_label.setStyleSheet("color: red; font-weight: normal;")
        
        self.status.run_threaded_with_progress(
            worker,
            title="Calculating file size...",
            on_success=on_success,
        )
        worker.error.connect(on_error)
    
    def _calculate_file_size_impl(
        self,
        volume: np.ndarray,
        bins: list,
        export_mode: str,
        format_name: str,
        export_colors: bool,
        export_opacity: bool,
        stl_binary: bool,
        progress_cb,
    ) -> str:
        """Calculate file size (runs in background thread)."""
        try:
            # Estimate mesh complexity from volume
            volume_gray = volume.mean(axis=-1) if volume.ndim == 4 else volume
            total_voxels = volume_gray.size
            non_zero_voxels = np.count_nonzero(volume_gray)
            
            # Estimate vertices and faces per bin
            # Marching cubes typically produces ~2 faces per surface voxel
            # Each face has 3 vertices, but vertices are shared (~2 faces share 1 vertex)
            # So roughly: vertices ≈ surface_voxels, faces ≈ 2 * surface_voxels
            
            total_estimated_size = 0
            
            if export_mode == "combined":
                # Combined: estimate total mesh size
                # Estimate surface voxels as a fraction of non-zero voxels
                # This is a rough estimate - actual depends on bin ranges
                estimated_surface_voxels = int(non_zero_voxels * 0.1)  # ~10% are on surface
                estimated_vertices = estimated_surface_voxels
                estimated_faces = estimated_surface_voxels * 2
                
                # Calculate file size
                if format_name == "STL":
                    if stl_binary:
                        # Binary STL: 80 byte header + 4 bytes (face count) + 50 bytes per face
                        total_estimated_size = 80 + 4 + (estimated_faces * 50)
                    else:
                        # ASCII STL: similar to PLY but simpler
                        header_size = 100
                        vertex_line_size = 30  # "vertex x y z\n"
                        face_line_size = 20  # "facet normal ... endfacet\n"
                        total_estimated_size = header_size + (estimated_vertices * vertex_line_size) + (estimated_faces * face_line_size)
                else:  # PLY
                    header_size = 300
                    if export_colors and export_opacity:
                        vertex_line_size = 50
                    elif export_colors or export_opacity:
                        vertex_line_size = 45
                    else:
                        vertex_line_size = 30
                    face_line_size = 18
                    total_estimated_size = header_size + (estimated_vertices * vertex_line_size) + (estimated_faces * face_line_size)
            else:
                # Separate: estimate size per bin and sum
                total_bins = len(bins)
                for i, bin_ in enumerate(bins):
                    # Report progress
                    progress_cb(i + 1, total_bins)
                    
                    # Estimate voxels in this bin range
                    bin_mask = (volume_gray >= bin_.low) & (volume_gray < bin_.high)
                    bin_voxels = np.count_nonzero(bin_mask)
                    # Estimate surface voxels
                    estimated_surface_voxels = int(bin_voxels * 0.1)  # ~10% are on surface
                    estimated_vertices = estimated_surface_voxels
                    estimated_faces = estimated_surface_voxels * 2
                    
                    # Calculate file size for this bin
                    if format_name == "STL":
                        if stl_binary:
                            file_size = 80 + 4 + (estimated_faces * 50)
                        else:
                            header_size = 100
                            vertex_line_size = 30
                            face_line_size = 20
                            file_size = header_size + (estimated_vertices * vertex_line_size) + (estimated_faces * face_line_size)
                    else:  # PLY
                        header_size = 300
                        if export_colors and export_opacity:
                            vertex_line_size = 50
                        elif export_colors or export_opacity:
                            vertex_line_size = 45
                        else:
                            vertex_line_size = 30
                        face_line_size = 18
                        file_size = header_size + (estimated_vertices * vertex_line_size) + (estimated_faces * face_line_size)
                    
                    total_estimated_size += file_size
            
            # Convert to MB and return
            size_mb = total_estimated_size / (1024 * 1024)
            if size_mb < 0.1:
                size_str = f"{size_mb * 1024:.1f} KB"
            else:
                size_str = f"{size_mb:.2f} MB"
            
            return size_str
            
        except Exception as e:
            raise RuntimeError(f"Error calculating file size: {str(e)}")
    
    # ------------------------------------------------------------------ #
    # Format capabilities and UI updates
    # ------------------------------------------------------------------ #
    def _get_format_capabilities(self, format_name: str) -> dict[str, bool]:
        """
        Get capabilities for a given format.
        
        Format capabilities:
        - PLY: Supports per-vertex colors (RGB) and per-vertex opacity (alpha channel) ✓
        - STL: Does NOT support colors or opacity (geometry only) ✓
        - OBJ: Supports colors via MTL files (not per-vertex), opacity via materials (not implemented)
        - GLTF/GLB: Supports colors and opacity (not yet implemented)
        - FBX: Supports colors and opacity (not yet implemented)
        """
        capabilities = {
            "PLY": {"colors": True, "opacity": True, "implemented": True},
            "STL": {"colors": False, "opacity": False, "implemented": True},
            "OBJ": {"colors": False, "opacity": False, "implemented": False},  # Not implemented yet
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
        
        # Show/hide STL-specific options
        is_stl = format_name.upper() == "STL" and caps.get("implemented", False)
        self.stl_format_row.itemAt(0).widget().setVisible(is_stl)
        self.stl_binary_cb.setVisible(is_stl)
        
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
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Auto-load volume and show preview (but don't auto-populate bins)
        self.load_volume_only()
        
        # Notify main window to update patient info if callback is set
        if hasattr(self, '_patient_info_callback') and self._patient_info_callback:
            QTimer.singleShot(100, self._patient_info_callback)
    
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
                self.processed_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                # Auto-load volume and show preview (but don't auto-populate bins)
                self.load_volume_only()
                
                # Notify main window to update patient info if callback is set
                if hasattr(self, '_patient_info_callback') and self._patient_info_callback:
                    QTimer.singleShot(100, self._patient_info_callback)
            else:
                # Update label to show default is available
                self.processed_label.setText(
                    f"Processed slices directory: {self._processed_dir}\n"
                    f"(Default available: {path})"
                )
                self.processed_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

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

    def _populate_optimal_bins(self, optimal_bins: List[IntensityBin], vmin: float, vmax: float) -> None:
        """Populate bins table with optimal bins detected from intensity distribution."""
        n_bins = len(optimal_bins)
        if n_bins == 0:
            # Fallback to default if no bins detected
            self._populate_default_bins(vmin, vmax)
            return
        
        # Round to integers for bin edges
        vmin_int = int(round(vmin))
        vmax_int = int(round(vmax))
        # Ensure minimum is at least 1 (0 is background/air)
        if vmin_int < 1:
            vmin_int = 1
        self.bins_table.setRowCount(n_bins)
        
        # Clear spinbox lists
        self._bin_low_spinboxes.clear()
        self._bin_high_spinboxes.clear()

        # Generate colors for bins
        import colorsys
        for i, bin in enumerate(optimal_bins):
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            enabled_item.setCheckState(Qt.Checked)

            # Create spinboxes for Low and High (integers)
            # Use detected range for spinbox limits
            min_val = max(1, vmin_int)
            max_val = vmax_int
            low_spin = QSpinBox()
            low_spin.setRange(min_val, max_val)
            low_spin.setValue(bin.low)
            low_spin.setSingleStep(1)
            low_spin.valueChanged.connect(lambda val, row=i: self._on_bin_spin_changed(row, 'low', val))
            self._bin_low_spinboxes.append(low_spin)
            
            high_spin = QSpinBox()
            high_spin.setRange(min_val, max_val)
            high_spin.setValue(bin.high)
            high_spin.setSingleStep(1)
            high_spin.valueChanged.connect(lambda val, row=i: self._on_bin_spin_changed(row, 'high', val))
            self._bin_high_spinboxes.append(high_spin)
            
            name_item = QTableWidgetItem(bin.name if bin.name else f"bin_{i + 1}")
            
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
            
            # Delete button
            delete_btn = QPushButton("Delete")
            delete_btn.setToolTip("Delete this bin")
            delete_btn.clicked.connect(lambda checked, row=i: self._on_delete_bin_row(row))

            self.bins_table.setItem(i, 0, enabled_item)
            self.bins_table.setCellWidget(i, 1, low_spin)
            self.bins_table.setCellWidget(i, 2, high_spin)
            self.bins_table.setItem(i, 3, name_item)
            self.bins_table.setItem(i, 4, color_item)
            self.bins_table.setCellWidget(i, 5, delete_btn)
        
        self._update_bin_visualization()
    
    def _populate_default_bins(self, vmin: float, vmax: float) -> None:
        """Create evenly spaced bins with default number (6 bins)."""
        n_bins = 6  # Default number of bins
        # Round to integers for bin edges
        vmin_int = int(round(vmin))
        vmax_int = int(round(vmax))
        # Ensure minimum is at least 1 (0 is background/air)
        if vmin_int < 1:
            vmin_int = 1
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
            # Ensure minimum is at least 1
            min_val = max(1, int(round(vmin)))
            max_val = int(round(vmax))
            low_spin = QSpinBox()
            low_spin.setRange(min_val, max_val)
            low_spin.setValue(max(min_val, int(round(edges[i]))))
            low_spin.setSingleStep(1)
            low_spin.valueChanged.connect(lambda val, row=i: self._on_bin_spin_changed(row, 'low', val))
            self._bin_low_spinboxes.append(low_spin)
            
            high_spin = QSpinBox()
            high_spin.setRange(min_val, max_val)
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
            
            # Delete button
            delete_btn = QPushButton("Delete")
            delete_btn.setToolTip("Delete this bin")
            delete_btn.clicked.connect(lambda checked, row=i: self._on_delete_bin_row(row))

            self.bins_table.setItem(i, 0, enabled_item)
            self.bins_table.setCellWidget(i, 1, low_spin)
            self.bins_table.setCellWidget(i, 2, high_spin)
            self.bins_table.setItem(i, 3, name_item)
            self.bins_table.setItem(i, 4, color_item)
            self.bins_table.setCellWidget(i, 5, delete_btn)
        
        self._update_bin_visualization()
    
    def _on_bin_spin_changed(self, row: int, which: str, value: int) -> None:
        # File size calculation is now manual only (button click)
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
    
    def _on_visualize_range_toggled(self, checked: bool) -> None:
        """Called when the Visualize checkbox is toggled."""
        self._update_range_lines_on_histogram()
        self._update_preview_intensity_range()
    
    def _update_range_lines_on_histogram(self) -> None:
        """Update the range lines (min/max) on histogram views."""
        if hasattr(self, 'histogram') and self._current_range is not None:
            # Only show range lines if visualize checkbox is checked
            if hasattr(self, 'visualize_range_cb') and self.visualize_range_cb.isChecked():
                bin_min = self.auto_bin_min_spin.value()
                bin_max = self.auto_bin_max_spin.value()
                self.histogram.update_range_lines(bin_min, bin_max)
            else:
                # Hide range lines
                self.histogram.update_range_lines(None, None)
    
    def _update_preview_intensity_range(self) -> None:
        """Update the intensity range highlighting in slice preview."""
        if hasattr(self, 'visualize_range_cb') and self.visualize_range_cb.isChecked():
            min_val = self.auto_bin_min_spin.value()
            max_val = self.auto_bin_max_spin.value()
            self.slice_preview.set_intensity_range(min_val, max_val)
        else:
            self.slice_preview.set_intensity_range(None, None)
    
    def _on_auto_bin_range_changed(self) -> None:
        """Called when auto-bin range spinboxes change."""
        # Update range lines on histogram
        self._update_range_lines_on_histogram()
        # Update preview highlighting
        self._update_preview_intensity_range()
        # Ensure min < max
        min_val = self.auto_bin_min_spin.value()
        max_val = self.auto_bin_max_spin.value()
        if min_val >= max_val:
            # Adjust to ensure min < max
            if self.auto_bin_min_spin.hasFocus():
                # Min was changed, adjust max
                self.auto_bin_max_spin.blockSignals(True)
                self.auto_bin_max_spin.setValue(min_val + 1)
                self.auto_bin_max_spin.blockSignals(False)
            else:
                # Max was changed, adjust min
                self.auto_bin_min_spin.blockSignals(True)
                self.auto_bin_min_spin.setValue(max(1, max_val - 1))
                self.auto_bin_min_spin.blockSignals(False)
    
    def _on_range_line_moved(self, line_type: str, new_value: float) -> None:
        """Called when a range line (min/max) is dragged on the histogram."""
        new_int = int(round(new_value))
        if line_type == 'min':
            # Ensure min < max
            current_max = self.auto_bin_max_spin.value()
            if new_int >= current_max:
                new_int = max(1, current_max - 1)
            self.auto_bin_min_spin.blockSignals(True)
            self.auto_bin_min_spin.setValue(new_int)
            self.auto_bin_min_spin.blockSignals(False)
        elif line_type == 'max':
            # Ensure max > min
            current_min = self.auto_bin_min_spin.value()
            if new_int <= current_min:
                new_int = current_min + 1
            self.auto_bin_max_spin.blockSignals(True)
            self.auto_bin_max_spin.setValue(new_int)
            self.auto_bin_max_spin.blockSignals(False)
        # Update preview highlighting (doesn't recreate lines)
        self._update_preview_intensity_range()
        # DO NOT call _update_range_lines_on_histogram() here - it recreates the lines
        # The lines are updated directly in the histogram view's _on_range_line_moved
    
    def _on_histogram_bin_moved(self, bin_table_row: int, boundary_type: str, new_value: float) -> None:
        """Called when a bin boundary is dragged on the histogram."""
        # Ensure minimum is at least 1
        new_value = max(1.0, new_value)
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
                    # After adjusting adjacent bins, update histogram to show the changes
                    self._update_bin_visualization()
                else:
                    # Update preview only (skip histogram to avoid update loop)
                    self._update_bin_visualization_no_histogram()

    # ------------------------------------------------------------------ #
    # Volume loading + histogram (threaded)
    # ------------------------------------------------------------------ #
    def load_volume_only(self) -> None:
        """Load volume and show slice previews, but don't compute histogram or auto-populate bins."""
        if self._processed_dir is None:
            self.status.show_error("Please select a processed slices directory first.")
            return

        folder = self._processed_dir

        # Precompute list of slice files so we know how many there are
        paths = images.list_slice_files(folder)
        total_slices = len(paths)
        self._num_slices = total_slices  # Store for Z height calculation
        if total_slices == 0:
            self.status.show_error("No image slices found in the selected directory.")
            self._update_z_height()  # Update to show no slices
            return

        # Create a custom worker that handles loading only
        class VolumeLoadWorker(WorkerBase):
            phase_progress = Signal(str, int, int, int)  # phase, current, phase_total, overall_total
            
            def run(self) -> None:  # type: ignore[override]
                try:
                    from PySide6.QtWidgets import QApplication
                    
                    # Phase 1: Loading image slices
                    slices = []
                    for i, p in enumerate(paths, start=1):
                        if self.isInterruptionRequested():
                            raise InterruptedError("Loading was cancelled")
                        arr = images.load_image_rgb(p)
                        slices.append(arr)
                        # Emit phase progress - total includes slices + building + histogram computation
                        # Histogram computation will process all slices, so total = slices + 1 (building) + slices (histogram)
                        total_phases = total_slices + 1 + total_slices
                        self.phase_progress.emit("Loading image slices", i, total_slices, total_phases)
                        # Also emit standard progress for compatibility
                        self.progress.emit(i, total_phases)
                        # Process events less frequently
                        if i % 50 == 0:
                            QApplication.processEvents()
                    
                    # Phase 2: Building volume
                    total_phases = total_slices + 1 + total_slices
                    self.phase_progress.emit("Building volume", 1, 1, total_phases)
                    self.progress.emit(total_slices + 1, total_phases)
                    volume = np.stack(slices, axis=0)  # (Z, Y, X, 3)
                    
                    self.finished.emit((volume,))
                except InterruptedError:
                    self.error.emit("Loading was cancelled by user")
                except BaseException as exc:  # noqa: BLE001
                    self._handle_exception(exc)
        
        worker = VolumeLoadWorker()
        
        def on_success(result: object, progress_dialog=None, progress_updater=None) -> None:
            from PySide6.QtWidgets import QApplication
            
            volume, = result  # type: ignore[misc]
            self._volume = volume

            # Get reference to the progress dialog from status controller
            # We need to access it to update progress during histogram computation
            # The dialog should still be visible at this point
            
            # Process events before starting
            QApplication.processEvents()
            
            # Compute intensity range for auto-bin controls
            try:
                if volume.ndim == 4 and volume.shape[-1] == 3:
                    grayscale = volume.max(axis=-1)
                else:
                    grayscale = volume
                intensities = grayscale.ravel()
                non_zero = intensities[intensities > 0]
                if non_zero.size > 0:
                    vmin = float(non_zero.min())
                    vmax = float(non_zero.max())
                    self._histogram_values = non_zero
                else:
                    vmin = float(intensities.min())
                    vmax = float(intensities.max())
                    self._histogram_values = intensities
                
                self._current_range = (vmin, vmax)
                
                # Detect if DICOM (uint16) or normal image (uint8)
                # DICOM typically has max intensity > 255, normal images max <= 255
                self._is_dicom = vmax > 255
                
                # Update auto-bin controls with detected range (adapt to DICOM vs normal)
                vmin_int = max(1, int(round(vmin)))
                vmax_int = int(round(vmax))
                max_range = 65535 if self._is_dicom else 255  # uint16 max or uint8 max
                
                self.auto_bin_min_spin.setRange(1, max_range)
                self.auto_bin_max_spin.setRange(1, max_range)
                self.auto_bin_min_spin.setValue(vmin_int)
                self.auto_bin_max_spin.setValue(vmax_int)
                
                # Update range lines on histogram if already computed
                if hasattr(self, 'histogram'):
                    self._update_range_lines_on_histogram()
            except Exception as e:
                print(f"Error computing intensity range: {e}")
                # Use defaults
                self._current_range = (1.0, 255.0)
                self._histogram_values = None
            
            # Update preview (no bins yet, but show images) - quick operation
            try:
                self.slice_preview.set_volume(volume)
                # Don't set bins - let preview show raw images
                # The preview widget should display images even without bins
                QApplication.processEvents()
                # Force a refresh
                self.slice_preview.update()
                QApplication.processEvents()
            except Exception as e:
                print(f"Error setting up preview: {e}")
                import traceback
                traceback.print_exc()
            
            # Enable buttons
            self.btn_apply_auto_bins.setEnabled(True)
            
            # Update Z height calculation
            self._update_z_height()
            
            # Now compute histogram graphs - update the existing progress dialog
            num_slices = volume.shape[0]
            base_progress = total_slices + 1  # Already completed: slices + building volume
            total_with_histograms = total_slices + 1 + num_slices  # slices + building + histogram computation
            
            # Progress callback to update the main progress dialog
            def on_histogram_progress(current: int, total: int) -> None:
                """Update progress dialog based on actual histogram computation progress."""
                if progress_dialog is not None and progress_updater is not None:
                    # Update phase progress for histogram computation
                    overall_current = base_progress + current
                    progress_updater("Computing histograms", current, total, total_with_histograms)
                    # Also update overall progress
                    if progress_dialog.maximum() != total_with_histograms:
                        progress_dialog.setMaximum(total_with_histograms)
                    progress_dialog.setValue(overall_current)
                    QApplication.processEvents()
            
            try:
                # Update progress dialog to show histogram computation phase
                if progress_dialog is not None:
                    progress_dialog.setMaximum(total_with_histograms)
                    progress_dialog.setValue(base_progress)
                    if progress_updater is not None:
                        progress_updater("Computing histograms", 0, num_slices, total_with_histograms)
                    QApplication.processEvents()
                
                # Update both histogram graphs (aggregated and slice-by-slice heatmap)
                # Pass progress callback to track actual progress during computation
                self.histogram.set_histogram_3d(
                    volume, 
                    n_bins=256, 
                    value_range=(vmin, vmax),
                    progress_callback=on_histogram_progress if progress_dialog is not None else None
                )
                
                # Complete histogram phase
                if progress_dialog is not None and progress_updater is not None:
                    progress_updater("Computing histograms", num_slices, num_slices, total_with_histograms)
                    progress_dialog.setValue(total_with_histograms)
                    QApplication.processEvents()
                
                # Update range lines on histogram if visualize is enabled
                self._update_range_lines_on_histogram()
                QApplication.processEvents()
            except Exception as e:
                print(f"Error computing histogram graphs: {e}")
                import traceback
                traceback.print_exc()

        # Run volume loading with progress
        self.status.run_threaded_with_progress(
            worker,
            "Loading volume...",
            on_success=on_success,
        )
    
    def _on_apply_auto_bins(self) -> None:
        """Apply auto-bin generation with user-specified parameters."""
        if self._volume is None:
            self.status.show_error("Please load a volume first.")
            return
        
        # Get parameters from UI
        bin_min = self.auto_bin_min_spin.value()
        bin_max = self.auto_bin_max_spin.value()
        n_bins = self.auto_bin_count_spin.value()
        uniformity = self.auto_bin_uniformity_spin.value()
        
        if bin_min >= bin_max:
            self.status.show_error("Minimum intensity must be less than maximum intensity.")
            return
        
        # Get histogram values if available, otherwise compute
        if not hasattr(self, '_histogram_values') or self._histogram_values is None:
            volume = self._volume
            if volume.ndim == 4 and volume.shape[-1] == 3:
                grayscale = volume.max(axis=-1)
            else:
                grayscale = volume
            intensities = grayscale.ravel()
            non_zero = intensities[intensities > 0]
            self._histogram_values = non_zero if non_zero.size > 0 else intensities
        
        hist_values = self._histogram_values
        
        # Generate bins based on uniformity parameter
        from ct23d.core import bins as binsmod
        from ct23d.core.models import IntensityBin
        import colorsys
        
        if uniformity >= 1.0:
            # Pure uniform bins
            edges = np.linspace(bin_min, bin_max, num=n_bins + 1, endpoint=True)
            edges = np.round(edges).astype(int)
            generated_bins = []
            for idx in range(n_bins):
                low = int(edges[idx])
                high = int(edges[idx + 1])
                if high <= low:
                    high = low + 1
                if low < 1:
                    low = 1
                    if high <= low:
                        high = low + 1
                generated_bins.append(
                    IntensityBin(
                        index=idx,
                        low=low,
                        high=high,
                        name=f"bin_{idx:02d}",
                        color=None,
                        enabled=True,
                    )
                )
        elif uniformity <= 0.0:
            # Pure auto-bins (full control based on intensity distribution)
            # Filter histogram values to user-specified range
            filtered_values = hist_values[(hist_values >= bin_min) & (hist_values <= bin_max)]
            if filtered_values.size == 0:
                # Fallback to uniform if no values in range
                filtered_values = hist_values
            optimal_bins = binsmod.generate_optimal_bins(filtered_values, min_bins=n_bins, max_bins=n_bins)
            generated_bins = optimal_bins
        else:
            # Interpolate between uniform and auto-bins
            # Uniform bins
            edges_uniform = np.linspace(bin_min, bin_max, num=n_bins + 1, endpoint=True)
            edges_uniform = np.round(edges_uniform).astype(int)
            uniform_bins = []
            for idx in range(n_bins):
                low = int(edges_uniform[idx])
                high = int(edges_uniform[idx + 1])
                if high <= low:
                    high = low + 1
                if low < 1:
                    low = 1
                    if high <= low:
                        high = low + 1
                uniform_bins.append((low, high))
            
            # Auto-bins
            filtered_values = hist_values[(hist_values >= bin_min) & (hist_values <= bin_max)]
            if filtered_values.size == 0:
                filtered_values = hist_values
            optimal_bins = binsmod.generate_optimal_bins(filtered_values, min_bins=n_bins, max_bins=n_bins)
            
            # Interpolate boundaries
            auto_edges = []
            if optimal_bins:
                for bin in optimal_bins:
                    auto_edges.append(bin.low)
                auto_edges.append(optimal_bins[-1].high)
            else:
                auto_edges = edges_uniform.tolist()
            
            # Blend edges with more aggressive uniformity factor
            # Use a power curve (uniformity^3) to make the effect more aggressive
            # This means even small uniformity values will have stronger effect
            aggressive_uniformity = uniformity ** 3
            blended_edges = []
            for i in range(len(edges_uniform)):
                uniform_val = float(edges_uniform[i])
                auto_val = float(auto_edges[i]) if i < len(auto_edges) else uniform_val
                blended_val = aggressive_uniformity * uniform_val + (1.0 - aggressive_uniformity) * auto_val
                blended_edges.append(int(round(blended_val)))
            
            # Ensure sorted and within range
            blended_edges = sorted(blended_edges)
            max_range = 65535 if self._is_dicom else 255
            blended_edges[0] = max(1, bin_min)
            blended_edges[-1] = min(max_range, bin_max)
            
            # Create bins from blended edges
            generated_bins = []
            for idx in range(n_bins):
                low = blended_edges[idx]
                high = blended_edges[idx + 1] if idx + 1 < len(blended_edges) else bin_max
                if high <= low:
                    high = low + 1
                if low < 1:
                    low = 1
                    if high <= low:
                        high = low + 1
                generated_bins.append(
                    IntensityBin(
                        index=idx,
                        low=low,
                        high=high,
                        name=f"bin_{idx:02d}",
                        color=None,
                        enabled=True,
                    )
                )
        
        # Populate bins table
        vmin = float(bin_min)
        vmax = float(bin_max)
        self._populate_optimal_bins(generated_bins, vmin, vmax)
        
        # Update histogram with bin boundaries
        bins = self._collect_bins_from_table()
        if hasattr(self, 'histogram'):
            self.histogram.update_bins(bins)
        
        self.status.show_info(f"Applied {len(generated_bins)} bins with uniformity {uniformity:.2f}.")
    
    def load_volume_and_histogram(self) -> None:
        if self._processed_dir is None:
            self.status.show_error("Please select a processed slices directory first.")
            return

        folder = self._processed_dir

        # Precompute list of slice files so we know how many there are
        paths = images.list_slice_files(folder)
        total_slices = len(paths)
        self._num_slices = total_slices  # Store for Z height calculation
        if total_slices == 0:
            self.status.show_error("No image slices found in the selected directory.")
            self._update_z_height()  # Update to show no slices
            return

        # Create a custom worker that handles both loading and histogram computation
        class VolumeLoadWorker(WorkerBase):
            phase_progress = Signal(str, int, int, int)  # phase, current, phase_total, overall_total
            
            def run(self) -> None:  # type: ignore[override]
                try:
                    from PySide6.QtWidgets import QApplication
                    
                    # Phase 1: Loading slices
                    slices = []
                    for i, p in enumerate(paths, start=1):
                        if self.isInterruptionRequested():
                            raise InterruptedError("Loading was cancelled")
                        arr = images.load_image_rgb(p)
                        slices.append(arr)
                        # Emit phase progress
                        self.phase_progress.emit("Loading slices", i, total_slices, total_slices + 2)
                        # Also emit standard progress for compatibility
                        self.progress.emit(i, total_slices + 2)
                        # Process events less frequently
                        if i % 50 == 0:
                            QApplication.processEvents()
                    
                    volume = np.stack(slices, axis=0)  # (Z, Y, X, 3)
                    
                    # Phase 2: Computing intensity range
                    self.phase_progress.emit("Computing intensity range", 0, 1, total_slices + 2)
                    QApplication.processEvents()
                    # Use max intensity across channels (same as meshing) to preserve
                    # information from colored overlays in medical imaging
                    if volume.ndim == 4 and volume.shape[-1] == 3:
                        grayscale = volume.max(axis=-1)
                    else:
                        grayscale = volume
                    intensities = grayscale.ravel()
                    non_zero = intensities[intensities > 0]
                    if non_zero.size > 0:
                        hist_values = non_zero
                    else:
                        hist_values = intensities
                    vmin = float(hist_values.min())
                    vmax = float(hist_values.max())
                    self.phase_progress.emit("Computing intensity range", 1, 1, total_slices + 2)
                    QApplication.processEvents()
                    
                    # Phase 3: Computing histograms (aggregated and heatmap)
                    # This is done in the success callback, but we report it here for progress
                    self.phase_progress.emit("Preparing histograms", 0, 1, total_slices + 2)
                    QApplication.processEvents()
                    self.phase_progress.emit("Preparing histograms", 1, 1, total_slices + 2)
                    
                    self.finished.emit((volume, hist_values, vmin, vmax))
                except InterruptedError:
                    self.error.emit("Loading was cancelled by user")
                except BaseException as exc:  # noqa: BLE001
                    self._handle_exception(exc)
        
        worker = VolumeLoadWorker()
        
        def on_success(result: object) -> None:
            from PySide6.QtWidgets import QApplication, QProgressDialog
            
            volume, hist_values, vmin, vmax = result  # type: ignore[misc]
            self._volume = volume
            self._current_range = (vmin, vmax)

            # Process events before starting
            QApplication.processEvents()
            
            # Auto-detect optimal bins from intensity distribution
            try:
                # Use histogram values for optimal bin detection
                from ct23d.core import bins as binsmod
                optimal_bins = binsmod.generate_optimal_bins(hist_values, min_bins=3, max_bins=12)
                
                # Extract detected range from optimal bins
                if optimal_bins:
                    detected_min = min(b.low for b in optimal_bins)
                    detected_max = max(b.high for b in optimal_bins)
                    # Use detected range if it's reasonable, otherwise use full range
                    if detected_min < detected_max:
                        vmin = float(detected_min)
                        vmax = float(detected_max)
                
                # Populate bins with auto-detected values
                self._populate_optimal_bins(optimal_bins, vmin, vmax)
                QApplication.processEvents()
                
                bins = self._collect_bins_from_table()
                
                # Update preview immediately
                self.slice_preview.set_volume(volume)
                self.slice_preview.set_bins(bins)
                # Force a complete update of the preview widget
                QApplication.processEvents()
                # Give preview time to render
                QApplication.processEvents()
            except Exception as e:
                print(f"Error setting up bins and preview: {e}")
                import traceback
                traceback.print_exc()
            
            # Update histogram (already computed in worker, but need to set it)
            try:
                self.histogram.set_histogram_3d(volume, n_bins=256, value_range=(vmin, vmax))
                # Force histogram to render
                QApplication.processEvents()
                # Give histogram time to render
                QApplication.processEvents()
            except Exception as e:
                print(f"Error setting histogram: {e}")
                import traceback
                traceback.print_exc()
            
            # Update histogram with bin boundaries (now that bins exist)
            try:
                bins = self._collect_bins_from_table()
                self.histogram.update_bins(bins)
                # Force final update
                QApplication.processEvents()
                # Give histogram time to update with bin boundaries
                QApplication.processEvents()
            except Exception as e:
                print(f"Error updating bins on histogram: {e}")
                import traceback
                traceback.print_exc()
            
            # File size calculation is now manual only (button click)
            
            self.btn_export_meshes.setEnabled(True)
            
            # Update Z height calculation
            self._update_z_height()
            
            # Final process events to ensure everything is rendered
            QApplication.processEvents()

        # Run volume loading with progress (single dialog)
            self.status.run_threaded_with_progress(
            worker,
            "Loading volume...",
            on_success=on_success,
        )
    
    def _update_z_height(self) -> None:
        """Update the Z height calculation display (number of slices × voxel spacing)."""
        if self._num_slices > 0 and hasattr(self, 'spin_spacing_z'):
            z_spacing = self.spin_spacing_z.value()
            z_height = self._num_slices * z_spacing
            self.z_height_label.setText(f"Z height: {z_height:.2f} mm ({self._num_slices} slices × {z_spacing:.2f} mm)")
        else:
            self.z_height_label.setText("Z height: —")
    
    def get_patient_info(self) -> tuple[str, str]:
        """
        Get patient info text and style from DICOM metadata if available.
        
        Returns
        -------
        tuple[str, str]
            Tuple of (text, style) for the patient info label
        """
        if self._processed_dir is None:
            return ("No DICOM files loaded", "color: gray;")
        
        # Get first file from the processed directory
        try:
            paths = images.list_slice_files(self._processed_dir)
            if not paths:
                return ("No DICOM files loaded", "color: gray;")
            
            # Try to get patient info from first DICOM file
            patient_info = images.get_dicom_patient_info(paths[0])
            
            if patient_info:
                # Format patient info nicely
                lines = []
                if 'PatientName' in patient_info:
                    lines.append(f"<b>Patient Name:</b> {patient_info['PatientName']}")
                if 'PatientID' in patient_info:
                    lines.append(f"<b>Patient ID:</b> {patient_info['PatientID']}")
                if 'PatientBirthDate' in patient_info:
                    lines.append(f"<b>Birth Date:</b> {patient_info['PatientBirthDate']}")
                if 'PatientSex' in patient_info:
                    lines.append(f"<b>Sex:</b> {patient_info['PatientSex']}")
                if 'StudyDate' in patient_info:
                    lines.append(f"<b>Study Date:</b> {patient_info['StudyDate']}")
                if 'StudyTime' in patient_info:
                    lines.append(f"<b>Study Time:</b> {patient_info['StudyTime']}")
                if 'StudyDescription' in patient_info:
                    lines.append(f"<b>Study Description:</b> {patient_info['StudyDescription']}")
                if 'Modality' in patient_info:
                    lines.append(f"<b>Modality:</b> {patient_info['Modality']}")
                
                if lines:
                    return ("<br>".join(lines), "color: white;")
                else:
                    return ("DICOM file loaded (no patient info available)", "color: gray;")
            else:
                return ("No DICOM files loaded", "color: gray;")
        except Exception:
            return ("No DICOM files loaded", "color: gray;")
    
    def _update_patient_info(self) -> None:
        """Update patient info - now just a placeholder for compatibility."""
        # This method is kept for compatibility but does nothing
        # The main window will call get_patient_info() instead
        pass
    
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
        # Get intensity range from current range (set when volume is loaded)
        if self._current_range is not None:
            vmin, vmax = self._current_range
            vmin = int(round(vmin))
            vmax = int(round(vmax))
        else:
            # Fallback if no volume loaded yet
            vmin, vmax = 1, 255
        
        cfg = MeshingConfig(
            spacing=spacing,
            min_intensity=vmin,
            max_intensity=vmax,
            smoothing_sigma=float(self.spin_smoothing_sigma.value()),
            min_component_size=int(self.spin_min_component_size.value()),
            enable_component_filtering=self.enable_component_filtering_cb.isChecked(),
            enable_smoothing=self.enable_smoothing_cb.isChecked(),
        )
        cfg.opacity = opacity  # Add opacity to config
        
        volume = self._volume
        output_dir = self._output_dir
        total_bins = len(bins)
        
        # Get format and capabilities
        export_format = self.format_combo.currentText().upper()
        caps = self._get_format_capabilities(export_format)
        
        # Get STL binary option
        stl_binary = self.stl_binary_cb.isChecked() if export_format == "STL" else True
        
        # Validate format is implemented
        if not caps.get("implemented", False):
            self.status.show_error(
                f"Format '{export_format}' is not yet implemented. "
                "Currently only PLY and STL formats are supported."
            )
            return
        
        # Capture export colors setting before creating worker
        export_colors = self.export_colors_cb.isChecked() and caps["colors"]
        
        # Create a custom worker with phase-aware progress for export
        class ExportWorker(WorkerBase):
            phase_progress = Signal(str, int, int, int)  # phase, current, phase_total, overall_total
            
            def __init__(self):
                super().__init__()
                self._cancelled = False
            
            def run(self) -> None:  # type: ignore[override]
                try:
                    from PySide6.QtWidgets import QApplication
                    
                    def phase_cb(phase: str, current: int, phase_total: int, overall_total: int) -> None:
                        # Check for interruption request FIRST, before any other operations
                        if self.isInterruptionRequested():
                            self._cancelled = True
                            raise InterruptedError("Export was cancelled by user")
                        # Only emit progress if not cancelled
                        if not self._cancelled:
                            self.phase_progress.emit(phase, current, phase_total, overall_total)
                            self.progress.emit(current, overall_total)
                            # Process events to keep UI responsive and prevent freezing
                            QApplication.processEvents()
                        else:
                            # Already cancelled, stop immediately
                            raise InterruptedError("Export was cancelled by user")
                    
                    def progress_cb(current: int, total: int) -> None:
                        # Check for interruption request
                        if self.isInterruptionRequested():
                            self._cancelled = True
                            raise InterruptedError("Export was cancelled by user")
                        # Only emit progress if not cancelled
                        if not self._cancelled:
                            self.progress.emit(current, total)
                        else:
                            # Already cancelled, stop immediately
                            raise InterruptedError("Export was cancelled by user")
                    
                    # Check before starting
                    if self.isInterruptionRequested():
                        raise InterruptedError("Export was cancelled by user")
                    
                    # Apply colors option: if not exporting colors or format doesn't support it, remove colors
                    export_bins = bins
                    if not export_colors:
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
                        stl_binary=stl_binary,  # STL format option
                        progress_callback=progress_cb,
                        phase_progress_callback=phase_cb,
                    )
                    
                    # Check one more time before finishing
                    if self.isInterruptionRequested() or self._cancelled:
                        raise InterruptedError("Export was cancelled by user")
                    
                    self.finished.emit(total_bins)
                except InterruptedError:
                    # User cancelled - emit error signal
                    self.error.emit("Export was cancelled by user")
                except BaseException as exc:  # noqa: BLE001
                    self._handle_exception(exc)
        
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
        # Get intensity range from current range (set when volume is loaded)
        if self._current_range is not None:
            vmin, vmax = self._current_range
            vmin = int(round(vmin))
            vmax = int(round(vmax))
        else:
            # Fallback if no volume loaded yet
            vmin, vmax = 1, 255
        
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
        low_spin.setRange(max(1, int(round(vmin))), int(round(vmax)))  # Minimum 1
        low_spin.setValue(max(1, int(round(mid - bin_width/2))))  # Ensure at least 1
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
        
        # Delete button
        delete_btn = QPushButton("Delete")
        delete_btn.setToolTip("Delete this bin")
        delete_btn.clicked.connect(lambda checked, r=row: self._on_delete_bin_row(r))
        
        self.bins_table.setItem(row, 0, enabled_item)
        self.bins_table.setCellWidget(row, 1, low_spin)
        self.bins_table.setCellWidget(row, 2, high_spin)
        self.bins_table.setItem(row, 3, name_item)
        self.bins_table.setItem(row, 4, color_item)
        self.bins_table.setCellWidget(row, 5, delete_btn)
        
        self._update_bin_visualization()
    
    def _on_delete_bin_row(self, row: int) -> None:
        """Delete a specific bin row."""
        if row < 0 or row >= self.bins_table.rowCount():
            return
        
        # Remove from spinbox lists
        if row < len(self._bin_low_spinboxes):
            self._bin_low_spinboxes.pop(row)
        if row < len(self._bin_high_spinboxes):
            self._bin_high_spinboxes.pop(row)
        
        # Remove the row from the table
        self.bins_table.removeRow(row)
        
        # Reconnect delete buttons for all remaining rows to ensure correct row indices
        # (since row indices shift after deletion)
        for r in range(self.bins_table.rowCount()):
            delete_btn = self.bins_table.cellWidget(r, 5)
            if delete_btn is not None:
                # Disconnect all existing connections
                try:
                    delete_btn.clicked.disconnect()
                except TypeError:
                    # No connections to disconnect
                    pass
                # Reconnect with correct row index (using default argument to capture r)
                delete_btn.clicked.connect(lambda checked, row_idx=r: self._on_delete_bin_row(row_idx))
        
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
        # File size calculation is now manual only (button click)
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
