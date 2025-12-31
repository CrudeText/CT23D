"""
Launch the CT23D GUI application.

Run this from the project root, e.g.:

    python scripts/run_ct23d_gui.py
"""

import os

# Set QT_API to pyside6 before any imports to avoid warnings from qtpy (used by pyvistaqt)
# This ensures pyvistaqt uses PySide6 directly instead of trying PyQt5 first
os.environ['QT_API'] = 'pyside6'

from ct23d.gui.app import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
