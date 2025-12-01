"""Diálogo de inicio de sesión con autenticación JWT."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


@dataclass(slots=True)
class AuthSession:
    """Mantiene la información de autenticación activa."""

    token: str
    expiration: Optional[datetime]
    api_base: str

    def is_valid(self) -> bool:
        """Indica si el token sigue vigente (si se proporcionó expiración)."""

        if self.expiration is None:
            return True
        if self.expiration.tzinfo is None:
            return datetime.utcnow() < self.expiration
        return datetime.now(timezone.utc) < self.expiration.astimezone(timezone.utc)


class LoginDialog(QDialog):
    """Pantalla modal de login contra el endpoint /api/Auth/login."""

    def __init__(self, api_base: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Autenticación requerida")
        self.setModal(True)
        self._default_api_base = api_base.rstrip("/")
        self.session: Optional[AuthSession] = None
        self._brand_logo_path = Path(__file__).parent / "mi_app" / "assets" / "logo.png"

        self._lbl_status = QLabel("")
        self._lbl_status.setObjectName("statusLabel")
        self._lbl_status.setStyleSheet("color: #b91c1c;")

        self._input_user = QLineEdit()
        self._input_user.setPlaceholderText("usuario")

        self._input_password = QLineEdit()
        self._input_password.setPlaceholderText("contraseña")
        self._input_password.setEchoMode(QLineEdit.EchoMode.Password)

        self._btn_login = QPushButton("Ingresar")
        self._btn_login.clicked.connect(self._on_submit)

        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.clicked.connect(self.reject)

        self._build_ui()
        self._input_user.setFocus()

    def _build_ui(self) -> None:
        hero = QHBoxLayout()
        hero.setContentsMargins(0, 0, 0, 0)
        hero.setSpacing(10)
        logo_label = QLabel()
        pixmap = QPixmap(str(self._brand_logo_path)) if self._brand_logo_path.exists() else QPixmap()
        if not pixmap.isNull():
            logo_label.setPixmap(pixmap.scaledToHeight(42, Qt.TransformationMode.SmoothTransformation))
        logo_label.setFixedSize(52, 52)
        logo_label.setStyleSheet("background: #fee2e2; border-radius: 12px;")

        title_box = QVBoxLayout()
        title = QLabel("Ingreso seguro")
        title.setStyleSheet("font-size: 15pt; font-weight: 700; color: #7f1d1d;")
        subtitle = QLabel("Usa tus credenciales para continuar")
        subtitle.setStyleSheet("color: #9f1239; font-weight: 500;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        hero.addWidget(logo_label)
        hero.addLayout(title_box)
        hero.addStretch(1)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.addRow("Usuario", self._input_user)
        form.addRow("Contraseña", self._input_password)

        buttons = QDialogButtonBox()
        buttons.addButton(self._btn_login, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(self._btn_cancel, QDialogButtonBox.ButtonRole.RejectRole)

        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.addLayout(hero)
        layout.addLayout(form)
        layout.addWidget(buttons)

        status_layout = QHBoxLayout()
        status_layout.addWidget(self._lbl_status)
        status_layout.addStretch(1)
        layout.addLayout(status_layout)

        self.setLayout(layout)
        self.setMinimumWidth(420)
        self._apply_styles()

    def _on_submit(self) -> None:
        username = self._input_user.text().strip()
        password = self._input_password.text()
        api_base = self._default_api_base

        if not username or not password:
            self._show_status("Usuario y contraseña son obligatorios.")
            return

        payload = json.dumps({"username": username, "password": password}).encode("utf-8")
        login_url = f"{api_base}/api/Auth/login"
        request = Request(
            login_url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=15) as response:
                raw = response.read()
        except HTTPError as exc:
            if exc.code == 401:
                self._show_status("Credenciales inválidas.")
            else:
                self._show_status(f"Error HTTP {exc.code} durante el login.")
            return
        except URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                self._show_status("El intento de login expiró por timeout.")
            else:
                self._show_status(f"No se pudo conectar al servicio: {exc.reason}.")
            return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._show_status("Respuesta del login no es JSON válido.")
            return

        token = payload.get("token") or payload.get("Token")
        expiration_text = payload.get("expiration") or payload.get("Expiration")
        if not token:
            self._show_status("El servicio no devolvió un token.")
            return

        expiration = self._parse_expiration(expiration_text)
        self.session = AuthSession(token=token, expiration=expiration, api_base=api_base)
        self.accept()

    def _parse_expiration(self, value: Optional[str]) -> Optional[datetime]:
        if not value or not isinstance(value, str):
            return None
        candidate = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None

    def _show_status(self, message: str) -> None:
        self._lbl_status.setText(message)
        self._lbl_status.setToolTip(message)
        self._lbl_status.setVisible(bool(message))
        if message:
            QMessageBox.warning(self, "Inicio de sesión", message)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #fff1f2;
                color: #7f1d1d;
                font-family: 'Segoe UI', 'Open Sans', sans-serif;
                font-size: 9pt;
            }
            QLineEdit {
                border: 1px solid #fda4af;
                border-radius: 8px;
                padding: 8px 10px;
                background: #fff;
            }
            QLineEdit:focus {
                border: 2px solid #e11d48;
                outline: none;
            }
            QLabel {
                color: #7f1d1d;
            }
            QPushButton {
                background: #e11d48;
                color: #fff;
                border: none;
                border-radius: 10px;
                padding: 9px 16px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #be123c;
            }
            QPushButton:disabled {
                background: #fecdd3;
                color: #9f1239;
            }
            #statusLabel {
                color: #b91c1c;
                font-weight: 600;
            }
            """
        )


__all__ = ["AuthSession", "LoginDialog"]
