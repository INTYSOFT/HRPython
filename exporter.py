"""Exportadores simples a JSON y CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from answers_reader import AnswersSummary
from dni_reader import DNIResult


def export_to_json(path: str | Path, dni: DNIResult, answers: AnswersSummary) -> None:
    data = {
        "dni": dni.digits,
        "conflictos_dni": dni.conflicts,
        "dni_en_blanco": dni.blanks,
        "respuestas": [
            {
                "pregunta": ans.question,
                "respuesta": ans.selected,
                "estado": ans.status,
                "intensidad": ans.intensity,
            }
            for ans in answers.answers
        ],
        "preguntas_en_blanco": answers.blanks,
        "preguntas_conflictivas": answers.conflicts,
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def export_to_csv(path: str | Path, dni: DNIResult, answers: AnswersSummary) -> None:
    fieldnames = ["dni", "pregunta", "respuesta", "estado", "intensidad"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ans in answers.answers:
            writer.writerow(
                {
                    "dni": dni.digits,
                    "pregunta": ans.question,
                    "respuesta": ans.selected or "-",
                    "estado": ans.status,
                    "intensidad": ans.intensity,
                }
            )


__all__ = ["export_to_json", "export_to_csv"]
