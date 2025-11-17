"""Lector de respuestas A–E para 100 preguntas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from alignment import AlignmentResult
from bubble_reader import measure_bubble
from template import TemplateGeometry


@dataclass
class AnswerResult:
    question: int
    selected: str | None
    status: str
    intensity: float


@dataclass
class AnswersSummary:
    answers: List[AnswerResult]
    blanks: int
    conflicts: int


def read_answers(
    gray: "np.ndarray",
    template: TemplateGeometry,
    alignment: AlignmentResult,
    labels: Sequence[str] = ("A", "B", "C", "D", "E"),
    threshold: float = 0.25,
    conflict_ratio: float = 0.80,
) -> AnswersSummary:
    results: List[AnswerResult] = []
    blanks = 0
    conflicts = 0

    for q_idx, opt_idx, col_idx, ref_center in template.answer_cell_centers():
        real = alignment.transform_point(ref_center)
        metric = measure_bubble(gray, real, template.answer_cell_size, threshold)
        results.append(
            AnswerResult(
                question=q_idx + 1,
                selected=labels[opt_idx] if metric.marked else None,
                status="RAW",  # se ajustará después
                intensity=metric.ink_level,
            )
        )

    # Normalizar agrupando por pregunta
    final: List[AnswerResult] = []
    questions = template.answers_columns * template.questions_per_column
    for q in range(questions):
        slice_start = q * len(labels)
        slice_end = slice_start + len(labels)
        options = results[slice_start:slice_end]
        scores = [o.intensity for o in options]
        best_idx = int(np.argmax(scores)) if scores else 0
        best = scores[best_idx] if scores else 0.0
        second = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0.0

        if best < threshold:
            blanks += 1
            final.append(
                AnswerResult(question=q + 1, selected=None, status="SIN RESPUESTA", intensity=best)
            )
            continue

        if second >= conflict_ratio * best:
            conflicts += 1
            final.append(
                AnswerResult(question=q + 1, selected=None, status="RESPUESTA MULTIPLE", intensity=best)
            )
            continue

        final.append(
            AnswerResult(question=q + 1, selected=labels[best_idx], status="OK", intensity=best)
        )

    return AnswersSummary(answers=final, blanks=blanks, conflicts=conflicts)


__all__ = ["AnswerResult", "AnswersSummary", "read_answers"]
