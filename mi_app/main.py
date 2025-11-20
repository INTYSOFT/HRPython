"""Punto de entrada de la aplicación.

Crea los componentes de infraestructura, servicios y estado, y arranca la
interfaz gráfica principal.
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from mi_app.core.services import UserService
from mi_app.core.state import AppState
from mi_app.infrastructure.api_client import APIClient
from mi_app.infrastructure.repositories import UserRepository
from mi_app.ui.main_window import MainWindow


def main() -> None:
    """Arranca la aplicación PyQt6 con las dependencias configuradas."""

    app = QApplication(sys.argv)

    api_client = APIClient()
    repository = UserRepository(api_client)
    user_service = UserService(repository)
    state = AppState()

    window = MainWindow(state=state, user_service=user_service)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - punto de entrada interactivo
    main()
