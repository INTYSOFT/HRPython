"""Cliente API simulado.

En un escenario real, este módulo encapsularía las peticiones HTTP.
Para la demostración, los datos están embebidos y se devuelven de forma
sincrónica, pero la interfaz permite intercambiar fácilmente la
implementación por una real.
"""

from __future__ import annotations

from typing import Iterable


class APIClient:
    """Provee acceso a datos de usuarios."""

    def obtener_usuarios(self) -> Iterable[dict]:
        """Recupera los usuarios del backend."""

        return (
            {
                "id": 1,
                "nombre": "Ana García",
                "email": "ana.garcia@example.com",
                "activo": True,
            },
            {
                "id": 2,
                "nombre": "Bruno Díaz",
                "email": "bruno.diaz@example.com",
                "activo": True,
            },
            {
                "id": 3,
                "nombre": "Carla Pérez",
                "email": "carla.perez@example.com",
                "activo": False,
            },
            {
                "id": 4,
                "nombre": "Diego Flores",
                "email": "diego.flores@example.com",
                "activo": True,
            },
        )


__all__ = ["APIClient"]
