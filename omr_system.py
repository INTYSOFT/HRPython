"""Sistema OMR completo basado en plantilla y marcas inferiores.

Implementa el flujo de trabajo descrito en los requerimientos: carga de imagen,
preprocesado, detección de marcas de alineación, cálculo de la transformación y
lectura de DNI y respuestas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alignment import AlignmentSettings, compute_alignment, detect_alignment_marks
from answers_reader import AnswersSummary, read_answers
from dni_reader import DNIResult, read_dni
from scanner_input import PreprocessSettings, load_image, preprocess_image
from template import TemplateGeometry


@dataclass
class PageResult:
    dni: DNIResult
    answers: AnswersSummary
    alignment_marks: List[tuple[float, float]]


class OMRSystem:
    def __init__(
        self,
        template: TemplateGeometry | None = None,
        preprocess_settings: PreprocessSettings | None = None,
        alignment_settings: AlignmentSettings | None = None,
    ) -> None:
        self.template = template or TemplateGeometry()
        self.preprocess_settings = preprocess_settings or PreprocessSettings()
        self.alignment_settings = alignment_settings or AlignmentSettings()

    def process_image(self, path: str | Path) -> PageResult:
        image = load_image(path)
        gray, processed = preprocess_image(image, self.preprocess_settings)

        marks = detect_alignment_marks(processed, self.alignment_settings)
        alignment = compute_alignment(self.template, marks)
        if alignment is None:
            raise RuntimeError("No se pudieron calcular las marcas de alineación")

        dni_result = read_dni(gray, self.template, alignment)
        answers_result = read_answers(gray, self.template, alignment)

        return PageResult(dni=dni_result, answers=answers_result, alignment_marks=alignment.detected_marks)


__all__ = ["OMRSystem", "PageResult"]
