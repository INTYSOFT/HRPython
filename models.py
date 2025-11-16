"""Modelos de datos utilizados por la aplicación OMR.

Se utilizan dataclasses para mantener el código simple y facilitar su
conversión a estructuras tabulares cuando se exportan los resultados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Respuesta:
    """Representa la respuesta de una pregunta individual.

    Attributes
    ----------
    pregunta:
        Número de pregunta (comienza en 1).
    alternativa:
        Letra marcada (A-E) o textos especiales como "SIN RESPUESTA".
    estado:
        Etiqueta opcional para marcar respuestas dudosas o múltiples.
    intensidad:
        Valor numérico usado internamente para auditoría. No se muestra en la UI,
        pero puede resultar útil al depurar el algoritmo OMR.
    """

    pregunta: int
    alternativa: str
    estado: str = "OK"
    intensidad: float | None = None

    def to_dict(self, pagina: int, dni: str) -> dict:
        """Devuelve la respuesta en formato listo para pandas."""

        return {
            "pagina": pagina,
            "dni": dni,
            "pregunta": self.pregunta,
            "respuesta": self.alternativa,
            "estado": self.estado,
            "intensidad": self.intensidad,
        }


@dataclass
class AlumnoHoja:
    """Información procesada de una sola hoja de respuestas."""

    pagina: int
    dni: str
    respuestas: List[Respuesta] = field(default_factory=list)
    imagen_path: Path | None = None

    def to_records(self) -> List[dict]:
        """Expande todas las respuestas en una lista de diccionarios."""

        return [resp.to_dict(self.pagina, self.dni) for resp in self.respuestas]


__all__ = ["Respuesta", "AlumnoHoja"]

