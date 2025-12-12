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
    QLineEdit,
)


class ExportPanel(QWidget):
    """
    Small UI section to choose:
      - output directory for meshes
      - filename prefix
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Output directory
        row1 = QHBoxLayout()
        self.output_label = QLabel("Output directory: (none)")
        self.select_output_btn = QPushButton("Select Output Folder")
        self.select_output_btn.clicked.connect(self.select_output_dir)
        row1.addWidget(self.output_label)
        row1.addWidget(self.select_output_btn)
        layout.addLayout(row1)

        # Prefix
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Filename prefix:"))
        self.prefix_edit = QLineEdit("ct_layer")
        row2.addWidget(self.prefix_edit)
        layout.addLayout(row2)

        self._output_dir: Optional[Path] = None

    # ------------------------------------------------------------------ #

    def select_output_dir(self) -> None:
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setWindowTitle("Select mesh output directory")

        if dlg.exec():
            p = Path(dlg.selectedFiles()[0])
            self._output_dir = p
            self.output_label.setText(f"Output directory: {p}")

    # ------------------------------------------------------------------ #

    @property
    def output_dir(self) -> Optional[Path]:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, path: Optional[Path]) -> None:
        self._output_dir = path
        if path is None:
            self.output_label.setText("Output directory: (none)")
        else:
            self.output_label.setText(f"Output directory: {path}")

    @property
    def prefix(self) -> str:
        return self.prefix_edit.text().strip() or "ct_layer"
