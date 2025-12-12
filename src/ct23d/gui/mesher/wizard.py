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
)

from ct23d.core import images
from ct23d.core.models import IntensityBin, MeshingConfig
from ct23d.core import meshing as meshing_core
from ct23d.gui.status import StatusController
from ct23d.gui.workers import FunctionWorker
from ct23d.gui.mesher.histogram_view import HistogramView


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
        self.btn_select_processed = QPushButton("Select Folder")
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

        # Intensity range
        params_layout.addWidget(QLabel("Intensity range min / max:"), 1, 2)
        self.spin_int_min = QDoubleSpinBox()
        self.spin_int_max = QDoubleSpinBox()
        for sp in (self.spin_int_min, self.spin_int_max):
            sp.setRange(-10_000.0, 10_000.0)
            sp.setDecimals(1)
            sp.setSingleStep(1.0)
        self.spin_int_min.setValue(1.0)
        self.spin_int_max.setValue(255.0)

        params_layout.addWidget(self.spin_int_min, 1, 3)
        params_layout.addWidget(self.spin_int_max, 1, 4)

        main_layout.addWidget(params_group)

        # Histogram + bin table
        middle_layout = QHBoxLayout()
        self.histogram = HistogramView(self)
        middle_layout.addWidget(self.histogram, stretch=3)

        self.bins_table = QTableWidget(0, 4, self)
        self.bins_table.setHorizontalHeaderLabels(["Enabled", "Low", "High", "Name"])
        self.bins_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.bins_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.bins_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        middle_layout.addWidget(self.bins_table, stretch=2)

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

        # Buttons: load volume + histogram, generate meshes
        bottom_row = QHBoxLayout()
        self.btn_load_volume = QPushButton("Load volume  compute histogram")
        self.btn_load_volume.clicked.connect(self.load_volume_and_histogram)
        bottom_row.addWidget(self.btn_load_volume)

        self.btn_generate_meshes = QPushButton("Generate meshes")
        self.btn_generate_meshes.clicked.connect(self.on_generate_meshes_clicked)
        self.btn_generate_meshes.setEnabled(False)
        bottom_row.addWidget(self.btn_generate_meshes)

        main_layout.addLayout(bottom_row)

    # ------------------------------------------------------------------ #
    # Directory selection
    # ------------------------------------------------------------------ #
    def _on_select_processed_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select processed slices directory")
        if not folder:
            return
        self._processed_dir = Path(folder)
        self.processed_label.setText(f"Processed slices directory: {folder}")

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
            low_item = self.bins_table.item(row, 1)
            high_item = self.bins_table.item(row, 2)
            name_item = self.bins_table.item(row, 3)

            enabled = True
            if enabled_item is not None:
                enabled = enabled_item.checkState() == Qt.Checked

            try:
                low = float(low_item.text()) if low_item is not None else 0.0
                high = float(high_item.text()) if high_item is not None else 0.0
            except Exception:
                continue

            name = name_item.text() if name_item is not None else ""
            bins.append(
                IntensityBin(
                    enabled=enabled,
                    low=low,
                    high=high,
                    name=name or f"bin_{row}",
                )
            )
        return bins

    def _populate_default_bins(self, vmin: float, vmax: float) -> None:
        """Create evenly spaced bins according to spin_num_bins and current range."""
        n_bins = int(self.spin_num_bins.value())
        edges = np.linspace(vmin, vmax, n_bins + 1)
        self.bins_table.setRowCount(n_bins)

        for i in range(n_bins):
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(enabled_item.flags() | Qt.ItemIsUserCheckable)
            enabled_item.setCheckState(Qt.Checked)

            low_item = QTableWidgetItem(f"{edges[i]:.2f}")
            high_item = QTableWidgetItem(f"{edges[i + 1]:.2f}")
            name_item = QTableWidgetItem(f"bin_{i + 1}")

            self.bins_table.setItem(i, 0, enabled_item)
            self.bins_table.setItem(i, 1, low_item)
            self.bins_table.setItem(i, 2, high_item)
            self.bins_table.setItem(i, 3, name_item)

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

            self.spin_int_min.setValue(vmin)
            self.spin_int_max.setValue(vmax)

            self.histogram.set_histogram(hist_values, n_bins=256, value_range=(vmin, vmax))
            self._populate_default_bins(vmin, vmax)
            self.btn_generate_meshes.setEnabled(True)

        self.status.run_threaded_with_progress(
            worker,
            "Loading volume...",
            on_success=on_success,
        )

    # ------------------------------------------------------------------ #
    # Mesh generation (threaded)
    # ------------------------------------------------------------------ #
    def on_generate_meshes_clicked(self) -> None:
        if self._volume is None:
            self.status.show_error("Please load a volume first.")
            return
        if self._output_dir is None:
            self.status.show_error("Please select an output directory.")
            return

        spacing = self._collect_spacing()
        bins = [b for b in self._collect_bins_from_table() if b.enabled]
        if not bins:
            self.status.show_error("No enabled intensity bins to mesh.")
            return

        prefix = self.edit_prefix.text().strip() or "ct_layer"
        vmin = float(self.spin_int_min.value())
        vmax = float(self.spin_int_max.value())

        cfg = MeshingConfig(
            spacing=spacing,
            bins=bins,
            intensity_min=vmin,
            intensity_max=vmax,
        )

        volume = self._volume
        output_dir = self._output_dir
        total_bins = len(bins)

        def mesh_task(progress_cb) -> int:
            def bin_progress(i: int) -> None:
                progress_cb(i, total_bins)

            meshing_core.generate_meshes_from_volume(
                volume=volume,
                config=cfg,
                output_dir=output_dir,
                filename_prefix=prefix,
                progress_callback=bin_progress,
            )
            return total_bins

        worker = FunctionWorker(mesh_task, with_progress=True)

        def on_success(result: object) -> None:
            n = int(result)
            self.status.show_info(f"Mesh generation complete ({n} bins processed).")

        self.status.run_threaded_with_progress(
            worker,
            "Generating meshes...",
            on_success=on_success,
        )
