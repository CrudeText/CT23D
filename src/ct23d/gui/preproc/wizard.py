from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
)

from ct23d.core.models import PreprocessConfig, ProjectConfig
from ct23d.core import preprocessing
from ct23d.gui.status import StatusController  # type: ignore[import]


class PreprocessWizard(QWidget):
    """
    Simple functional preprocessing tool.

    Allows you to:
      - select an input directory with raw CT slices
      - choose the processed output directory
      - configure overlay / bed removal parameters
      - run the preprocessing pipeline
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        status: Optional[StatusController] = None,
    ) -> None:
        super().__init__(parent)

        self.status = status
        self._preproc_worker = None  # keep reference to avoid GC

        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # ------------------------------------------------------------------
        # Input directory selector
        # ------------------------------------------------------------------
        row1 = QHBoxLayout()
        self.input_label = QLabel("Input directory: (none)")
        choose_input_btn = QPushButton("Select Input Folder")
        choose_input_btn.clicked.connect(self.select_input_dir)
        row1.addWidget(self.input_label)
        row1.addWidget(choose_input_btn)
        layout.addLayout(row1)

        # ------------------------------------------------------------------
        # Processed output directory selector
        # ------------------------------------------------------------------
        row2 = QHBoxLayout()
        self.output_label = QLabel("Processed directory: (auto)")
        choose_output_btn = QPushButton("Select Output Folder")
        choose_output_btn.clicked.connect(self.select_output_dir)
        row2.addWidget(self.output_label)
        row2.addWidget(choose_output_btn)
        layout.addLayout(row2)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        params_layout = QVBoxLayout()

        # grayscale tolerance
        hl1 = QHBoxLayout()
        hl1.addWidget(QLabel("Grayscale tolerance:"))
        self.gray_spin = QSpinBox()
        self.gray_spin.setRange(0, 50)
        self.gray_spin.setValue(1)
        hl1.addWidget(self.gray_spin)
        params_layout.addLayout(hl1)

        # saturation threshold
        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Saturation threshold:"))
        self.sat_spin = QDoubleSpinBox()
        self.sat_spin.setRange(0.0, 1.0)
        self.sat_spin.setSingleStep(0.01)
        self.sat_spin.setValue(0.08)
        hl2.addWidget(self.sat_spin)
        params_layout.addLayout(hl2)

        # remove bed checkbox
        self.remove_bed_cb = QCheckBox("Remove bed/headrest")
        self.remove_bed_cb.setChecked(True)
        params_layout.addWidget(self.remove_bed_cb)

        layout.addLayout(params_layout)

        # ------------------------------------------------------------------
        # Run button
        # ------------------------------------------------------------------
        run_btn = QPushButton("Run preprocessing")
        run_btn.clicked.connect(self.run_preprocessing)
        layout.addWidget(run_btn)

        layout.addStretch(1)

    # ----------------------------------------------------------------------
    # Directory selection slots
    # ----------------------------------------------------------------------
    def select_input_dir(self) -> None:
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setWindowTitle("Select input CT slice directory")

        if dlg.exec():
            dir_path = Path(dlg.selectedFiles()[0])
            self.input_dir = dir_path
            self.input_label.setText(f"Input directory: {dir_path}")

            # Auto-suggest processed_slices under the input dir
            auto_out = dir_path / "processed_slices"
            self.output_dir = auto_out
            self.output_label.setText(f"Processed directory: {auto_out}")

    def select_output_dir(self) -> None:
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setWindowTitle("Select processed output directory")

        if dlg.exec():
            dir_path = Path(dlg.selectedFiles()[0])
            self.output_dir = dir_path
            self.output_label.setText(f"Processed directory: {dir_path}")

    # ----------------------------------------------------------------------
    # Run preprocessing
    # ----------------------------------------------------------------------
    def run_preprocessing(self) -> None:
        """
        Trigger the preprocessing pipeline.

        If a StatusController is available, this runs in a background QThread
        (PreprocessWorker) with a determinate progress bar (slices processed / total).

        If no StatusController is provided (e.g. in tests), it falls back
        to a simple synchronous call.
        """
        if self.input_dir is None:
            QMessageBox.warning(
                self,
                "Missing input directory",
                "Please select an input directory containing CT slices.",
            )
            return

        cfg = PreprocessConfig(
            input_dir=self.input_dir,
            processed_dir=self.output_dir,
            use_cache=True,
            grayscale_tolerance=self.gray_spin.value(),
            saturation_threshold=self.sat_spin.value(),
            remove_bed=self.remove_bed_cb.isChecked(),
        )

        project = ProjectConfig(
            name="GUI Preprocessing",
            preprocess=cfg,
        )

        # ------------------------------------------------------------------
        # Fallback path: no StatusController -> synchronous
        # ------------------------------------------------------------------
        if self.status is None:
            try:
                out_dir = preprocessing.preprocess_slices(project.preprocess)
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(
                    self,
                    "Error during preprocessing",
                    f"An error occurred while preprocessing slices:\n\n{e}",
                )
                return

            QMessageBox.information(
                self,
                "Preprocessing complete",
                f"Processed slices saved to:\n{out_dir}",
            )
            return

        # ------------------------------------------------------------------
        # Threaded path: use PreprocessWorker + run_threaded_with_progress
        # ------------------------------------------------------------------
        from ct23d.gui.workers import PreprocessWorker  # local import to avoid cycles

        worker = PreprocessWorker(project.preprocess)
        self._preproc_worker = worker  # keep reference

        def on_finished(out_dir: Path) -> None:
            QMessageBox.information(
                self,
                "Preprocessing complete",
                f"Processed slices saved to:\n{out_dir}",
            )
            self.status.show_message("Preprocessing complete")

        def on_error(exc: Exception) -> None:
            QMessageBox.critical(
                self,
                "Error during preprocessing",
                f"An error occurred while preprocessing slices:\n\n{exc}",
            )
            self.status.show_message("Preprocessing failed")

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)

        self.status.run_threaded_with_progress(
            worker,
            title="Preprocessing",
            parent=self,
            cancellable=True,
        )
