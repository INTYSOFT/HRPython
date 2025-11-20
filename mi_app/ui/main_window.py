"""Ventana principal de la aplicación."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mi_app.core.services import UserService
from mi_app.core.state import AppState
from mi_app.models.user import User


@dataclass(slots=True)
class _TableColumns:
    nombre: int = 0
    email: int = 1


class MainWindow(QMainWindow):
    """Ventana principal con listado de usuarios."""

    def __init__(self, *, state: AppState, user_service: UserService) -> None:
        super().__init__()
        self.state = state
        self.user_service = user_service
        self._columns = _TableColumns()

        self.setWindowTitle("Gestión de usuarios")
        self.resize(640, 400)

        self.search_box = QLineEdit(placeholderText="Buscar por nombre o email")
        self.search_box.textChanged.connect(self._on_search_changed)

        self.refresh_button = QPushButton("Recargar")
        self.refresh_button.clicked.connect(self._reload_data)

        self.table = QTableWidget(columnCount=2)
        self.table.setHorizontalHeaderLabels(["Nombre", "Email"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Usuarios"))
        top_bar.addStretch(1)
        top_bar.addWidget(self.search_box)
        top_bar.addWidget(self.refresh_button)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self._reload_data()

    # ------------------------------------------------------------------
    # Eventos y acciones
    # ------------------------------------------------------------------
    def _reload_data(self) -> None:
        """Recarga los usuarios desde el servicio y refresca la tabla."""

        try:
            usuarios = self.user_service.obtener_activos()
        except Exception as exc:  # pragma: no cover - UI
            QMessageBox.critical(self, "Error", f"No se pudieron cargar usuarios: {exc}")
            return

        self.state.actualizar_usuarios(usuarios)
        self._populate_table(usuarios)

    def _on_search_changed(self, text: str) -> None:
        usuarios = self.user_service.buscar_por_nombre(text)
        self._populate_table(usuarios)

    def _on_selection_changed(self) -> None:
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.state.usuarios):
            self.state.seleccionar_usuario(None)
            return

        self.state.seleccionar_usuario(self.state.usuarios[current_row])

    # ------------------------------------------------------------------
    # Renderizado
    # ------------------------------------------------------------------
    def _populate_table(self, usuarios: list[User]) -> None:
        self.table.setRowCount(len(usuarios))

        for row, usuario in enumerate(usuarios):
            nombre_item = QTableWidgetItem(usuario.nombre)
            email_item = QTableWidgetItem(usuario.email)

            nombre_item.setFlags(nombre_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            email_item.setFlags(email_item.flags() ^ Qt.ItemFlag.ItemIsEditable)

            self.table.setItem(row, self._columns.nombre, nombre_item)
            self.table.setItem(row, self._columns.email, email_item)

        self.table.resizeColumnsToContents()
        if usuarios:
            self.table.selectRow(0)
        else:
            self.state.seleccionar_usuario(None)


__all__ = ["MainWindow"]
