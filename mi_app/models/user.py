"""Definiciones de modelos de dominio."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class User:
    """Entidad de usuario mínima para la demostración de la aplicación."""

    id: int
    nombre: str
    email: str
    activo: bool = True


__all__ = ["User"]
