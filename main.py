"""Punto de entrada de la aplicación OMR."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QDialog

from login_dialog import LoginDialog
from ui_main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)

    # Estilo base moderno
    app.setStyle("Fusion")

    # Fuente global un poco más pequeña
    font = app.font()
    font.setFamily("Segoe UI")          # o "Roboto", "Open Sans", etc. si los tienes
    font.setPointSize(8)               # antes suele ser 10–11
    app.setFont(font)

    login = LoginDialog(api_base=MainWindow.API_BASE)
    if login.exec() != QDialog.DialogCode.Accepted or not login.session:
        sys.exit(0)

    window = MainWindow(auth_session=login.session)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - punto de entrada
    main()
