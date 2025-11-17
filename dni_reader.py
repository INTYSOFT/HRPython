"""Lector específico del bloque de DNI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from alignment import AlignmentResult
from bubble_reader import measure_bubble
from template import TemplateGeometry


@dataclass
class DNIResult:
    digits: str
    conflicts: List[int]
    blanks: List[int]


def read_dni(
    gray: "np.ndarray",
    template: TemplateGeometry,
    alignment: AlignmentResult,
    threshold: float = 0.25,
    conflict_ratio: float = 0.85,
) -> DNIResult:
    import numpy as np  # import local para no romper al cargar sin cv2

    digit_values: Sequence[int] = tuple(range(10))
    digits: List[str] = []
    conflicts: List[int] = []
    blanks: List[int] = []

    ox, oy = template.dni_origin
    for col in range(8):
        metrics = []
        for row in range(10):
            ref_center = (ox + col * template.dni_dx, oy + row * template.dni_dy)
            real = alignment.transform_point(ref_center)
            metrics.append(
                measure_bubble(gray, real, template.dni_cell_size, threshold)
            )

        ink_levels = [m.ink_level for m in metrics]
        best_idx = int(np.argmax(ink_levels)) if ink_levels else 0
        best_val = ink_levels[best_idx] if ink_levels else 0.0
        second_val = sorted(ink_levels, reverse=True)[1] if len(ink_levels) > 1 else 0.0

        if best_val < threshold:
            digits.append("")
            blanks.append(col)
            continue

        if best_val > 0 and second_val >= conflict_ratio * best_val:
            digits.append("")
            conflicts.append(col)
            continue

        mapped_digit = digit_values[best_idx]
        digits.append(str(mapped_digit))

    return DNIResult(digits="".join(digits), conflicts=conflicts, blanks=blanks)


__all__ = ["DNIResult", "read_dni"]
