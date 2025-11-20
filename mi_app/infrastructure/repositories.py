"""Implementaciones de repositorios para acceso a datos."""

from __future__ import annotations

from mi_app.infrastructure.api_client import APIClient
from mi_app.models.user import User


class UserRepository:
    """Repositorio de usuarios basado en un cliente API."""

    def __init__(self, api_client: APIClient) -> None:
        self._api_client = api_client

    def obtener_usuarios(self) -> list[User]:
        """Devuelve la lista completa de usuarios."""

        usuarios_crudos = self._api_client.obtener_usuarios()
        return [
            User(
                id=datos["id"],
                nombre=datos["nombre"],
                email=datos["email"],
                activo=bool(datos.get("activo", True)),
            )
            for datos in usuarios_crudos
        ]


__all__ = ["UserRepository"]
