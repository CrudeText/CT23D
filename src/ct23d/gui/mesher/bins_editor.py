from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)

from ct23d.core.models import IntensityBin


class BinsEditor(QWidget):
    """
    Simple table-based editor for intensity bins.

    Columns:
      [0] Enabled (checkbox)
      [1] Low
      [2] High
      [3] Name
    """

    HEADERS = ["Enabled", "Low", "High", "Name"]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(self)
        self.table.setColumnCount(len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(self.table)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_bins(self, bins: List[IntensityBin]) -> None:
        self.table.setRowCount(len(bins))
        for row, b in enumerate(bins):
            # Enabled checkbox
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            enabled_item.setCheckState(Qt.Checked if b.enabled else Qt.Unchecked)
            self.table.setItem(row, 0, enabled_item)

            # Low
            low_item = QTableWidgetItem(str(b.low))
            low_item.setData(Qt.EditRole, b.low)
            self.table.setItem(row, 1, low_item)

            # High
            high_item = QTableWidgetItem(str(b.high))
            high_item.setData(Qt.EditRole, b.high)
            self.table.setItem(row, 2, high_item)

            # Name
            name_item = QTableWidgetItem(b.name or "")
            self.table.setItem(row, 3, name_item)

    def get_bins(self) -> List[IntensityBin]:
        bins: List[IntensityBin] = []
        for row in range(self.table.rowCount()):
            enabled_item = self.table.item(row, 0)
            low_item = self.table.item(row, 1)
            high_item = self.table.item(row, 2)
            name_item = self.table.item(row, 3)

            enabled = (
                enabled_item is not None and enabled_item.checkState() == Qt.Checked
            )
            try:
                low = int(low_item.data(Qt.EditRole)) if low_item is not None else 0
                high = int(high_item.data(Qt.EditRole)) if high_item is not None else 0
            except (TypeError, ValueError):
                continue

            name = name_item.text() if name_item is not None else None
            bins.append(
                IntensityBin(
                    index=row,
                    low=low,
                    high=high,
                    name=name or None,
                    color=None,
                    enabled=enabled,
                )
            )

        return bins
