"""Servicios de aplicación que coordinan el acceso a datos."""

from __future__ import annotations

from typing import Iterable

from mi_app.infrastructure.repositories import UserRepository
from mi_app.models.user import User


class UserService:
    """Orquesta el flujo de datos relacionado con usuarios."""

    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def obtener_activos(self) -> list[User]:
        """Devuelve los usuarios activos ordenados por nombre."""

        usuarios = self._repository.obtener_usuarios()
        activos = (usuario for usuario in usuarios if usuario.activo)
        ordenados = sorted(activos, key=lambda usuario: usuario.nombre.lower())
        return list(ordenados)

    def buscar_por_nombre(self, consulta: str) -> list[User]:
        """Filtra usuarios activos por nombre o correo."""

        consulta_normalizada = consulta.strip().lower()
        if not consulta_normalizada:
            return self.obtener_activos()

        return [
            usuario
            for usuario in self.obtener_activos()
            if consulta_normalizada in usuario.nombre.lower()
            or consulta_normalizada in usuario.email.lower()
        ]


__all__ = ["UserService"]
