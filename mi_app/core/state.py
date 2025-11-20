"""Estado compartido de la aplicación."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from mi_app.models.user import User


@dataclass
class AppState:
    """Mantiene los usuarios visibles y la selección actual."""

    usuarios: List[User] = field(default_factory=list)
    usuario_seleccionado: User | None = None

    def actualizar_usuarios(self, usuarios: list[User]) -> None:
        self.usuarios = usuarios
        self.usuario_seleccionado = usuarios[0] if usuarios else None

    def seleccionar_usuario(self, usuario: User | None) -> None:
        self.usuario_seleccionado = usuario


__all__ = ["AppState"]
