"""Punto de entrada de la aplicación OMR."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from ui_main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)

    # Estilo base moderno
    app.setStyle("Fusion")

    # Fuente global un poco más pequeña
    font = app.font()
    font.setFamily("Segoe UI")          # o "Roboto", "Open Sans", etc. si los tienes
    font.setPointSize(9)               # antes suele ser 10–11
    app.setFont(font)

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - punto de entrada
    main()