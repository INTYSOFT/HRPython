"""Configuración de plantilla y helpers para el sistema OMR.

Esta plantilla modela una hoja base de 1240×874 px y expone fórmulas para
posicionar las celdas de DNI y respuestas descritas en los requerimientos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class TemplateGeometry:
    """Parámetros geométricos de la hoja OMR.

    Los valores se corresponden con la plantilla base indicada en los
    requerimientos y pueden ajustarse en tiempo de ejecución si se calibra una
    nueva hoja.
    """

    # Tamaño ideal de la hoja en píxeles
    width: float = 1240.0
    height: float = 874.0

    # Marcas de alineación inferiores (coordenadas ideales)
    mark_left: Point = (100.0, 820.0)
    mark_center: Point = (620.0, 820.0)
    mark_right: Point = (1140.0, 820.0)

    # Configuración del bloque de DNI (8×10 burbujas)
    dni_origin: Point = (58.7, 176.5)
    dni_dx: float = 6.0
    dni_dy: float = 6.0
    dni_cell_size: Point = (8.0, 8.0)

    # Configuración de la matriz de respuestas
    answer_origin: Point = (456.0, 94.0)
    answer_dx: float = 8.0
    answer_dy: float = 6.5
    column_offset: float = 150.0
    answers_columns: int = 4
    questions_per_column: int = 25
    options_per_question: int = 5
    answer_cell_size: Point = (8.0, 8.0)

    def alignment_reference_points(self) -> List[Point]:
        return [self.mark_left, self.mark_center, self.mark_right]

    def dni_cell_centers(self) -> Iterable[Point]:
        """Genera los centros ideales para las 8×10 burbujas de DNI."""

        ox, oy = self.dni_origin
        for col in range(8):
            for row in range(10):
                yield (ox + col * self.dni_dx, oy + row * self.dni_dy)

    def answer_cell_centers(self) -> Iterable[Tuple[int, int, int, Point]]:
        """Genera centros ideales de todas las respuestas.

        Devuelve tuplas ``(question_index, option_index, column_index, point)``.
        ``question_index`` empieza en 0.
        """

        base_x, base_y = self.answer_origin
        for col in range(self.answers_columns):
            x_base = base_x + col * self.column_offset
            for q in range(self.questions_per_column):
                y = base_y + q * self.answer_dy
                question_idx = col * self.questions_per_column + q
                for option in range(self.options_per_question):
                    x = x_base + option * self.answer_dx
                    yield (question_idx, option, col, (x, y))


__all__ = ["TemplateGeometry", "Point"]
