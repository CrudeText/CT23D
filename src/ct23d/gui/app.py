from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def get_qapp() -> QApplication:
    """
    Get or create the global QApplication instance.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def run() -> None:
    """
    Launch the CT23D GUI application.
    """
    app = get_qapp()

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
