"""Punto de entrada de la aplicación OMR."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from ui_main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - punto de entrada
    main()