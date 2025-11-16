"""Módulo encargado del procesamiento OMR.

Aunque el algoritmo es configurable, aquí se proporcionan heurísticas
razonables para hojas con columnas de DNI y columnas de respuestas con
marcas de referencia rectangulares en la parte inferior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import fitz  # PyMuPDF
import numpy as np

from models import AlumnoHoja, Respuesta


@dataclass
class OMRConfig:
    """Parámetros del algoritmo OMR."""

    dpi: int = 200
    dni_columns: int = 9
    questions: int = 50
    answer_labels: Sequence[str] = ("A", "B", "C", "D", "E")
    dni_vertical_band: tuple[float, float] = (0.08, 0.32)
    answer_vertical_band: tuple[float, float] = (0.34, 0.9)
    sync_band_height_ratio: float = 0.18
    min_rect_area_ratio: float = 0.0005
    cell_activation_threshold: float = 0.45
    ambiguity_margin: float = 0.15


def procesar_pdf(pdf_path: str | Path, cache_dir: str | Path | None = None, config: OMRConfig | None = None) -> List[AlumnoHoja]:
    """Procesa todas las páginas de un PDF y devuelve objetos ``AlumnoHoja``.

    Parameters
    ----------
    pdf_path:
        Ruta al archivo PDF.
    cache_dir:
        Directorio donde se guardarán las imágenes renderizadas. Si es ``None``
        se crea ``processed_pages`` junto al PDF.
    config:
        Configuración opcional. Si no se provee se usa ``OMRConfig`` por defecto.
    """

    config = config or OMRConfig()
    pdf_path = Path(pdf_path)
    if cache_dir is None:
        cache_dir = pdf_path.parent / "processed_pages"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    resultados: List[AlumnoHoja] = []

    for index, page in enumerate(doc, start=1):
        image_bgr, img_path = _render_page(page, cache_dir, index, config.dpi)
        anchors = _detectar_rectangulos_sync(image_bgr, config)
        if len(anchors) < config.dni_columns:
            # Si no se detectaron suficientes anclas, se crea un objeto vacío
            # para permitir al usuario revisar manualmente la página.
            resultados.append(
                AlumnoHoja(
                    pagina=index,
                    dni="NO DETECTADO",
                    respuestas=[],
                    imagen_path=img_path,
                )
            )
            continue

        dni = _leer_dni(image_bgr, anchors[: config.dni_columns], config)
        respuestas = _leer_respuestas(
            image_bgr,
            anchors[config.dni_columns :],
            config,
        )
        resultados.append(
            AlumnoHoja(pagina=index, dni=dni, respuestas=respuestas, imagen_path=img_path)
        )

    return resultados


def _render_page(page: fitz.Page, cache_dir: Path, index: int, dpi: int) -> tuple[np.ndarray, Path]:
    """Renderiza una página en imagen BGR (formato OpenCV) y la guarda."""

    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    image = arr.reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        image = image[:, :, :3]
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_path = cache_dir / f"pagina_{index:03d}.png"
    cv2.imwrite(str(img_path), image_bgr)
    return image_bgr, img_path


def _detectar_rectangulos_sync(image: np.ndarray, config: OMRConfig) -> List[tuple[int, int, int, int]]:
    """Detecta rectángulos negros en la banda inferior.

    Retorna una lista de bounding boxes ``(x, y, w, h)`` en coordenadas de la
    imagen completa, ordenada de izquierda a derecha.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    band_start = int(h * (1 - config.sync_band_height_ratio))
    bottom_band = gray[band_start:, :]
    blur = cv2.GaussianBlur(bottom_band, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: List[tuple[int, int, int, int]] = []
    min_area = config.min_rect_area_ratio * w * h
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        aspect = rw / rh if rh else 0
        if area < min_area:
            continue
        if aspect < 0.5 or aspect > 5:
            continue
        rects.append((x, y + band_start, rw, rh))

    rects.sort(key=lambda r: r[0])
    return rects


def _leer_dni(image: np.ndarray, columnas: Sequence[tuple[int, int, int, int]], config: OMRConfig) -> str:
    """Lee los dígitos del DNI aprovechando las columnas detectadas."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    y0 = int(h * config.dni_vertical_band[0])
    y1 = int(h * config.dni_vertical_band[1])
    banda = gray[y0:y1, :]
    digits = []
    column_width = _estimacion_ancho_columnas(columnas)

    for col in columnas:
        x_center = col[0] + col[2] // 2
        x0 = max(x_center - column_width // 2, 0)
        x1 = min(x_center + column_width // 2, w)
        sub = banda[:, x0:x1]
        digit = _clasificar_digito(sub)
        digits.append(str(digit))

    return "".join(digits)


def _leer_respuestas(
    image: np.ndarray,
    columnas: Sequence[tuple[int, int, int, int]],
    config: OMRConfig,
) -> List[Respuesta]:
    """Procesa el bloque de preguntas y devuelve ``Respuesta`` por pregunta."""

    if not columnas:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    y0 = int(h * config.answer_vertical_band[0])
    y1 = int(h * config.answer_vertical_band[1])
    banda = gray[y0:y1, :]
    answers_per_column = int(np.ceil(config.questions / len(columnas)))
    column_width = _estimacion_ancho_columnas(columnas)
    resultados: List[Respuesta] = []

    for question_index in range(config.questions):
        column_index = min(question_index // answers_per_column, len(columnas) - 1)
        col = columnas[column_index]
        x_center = col[0] + col[2] // 2
        x0 = max(x_center - column_width // 2, 0)
        x1 = min(x_center + column_width // 2, w)

        row_height = (y1 - y0) / answers_per_column
        local_top = int((question_index % answers_per_column) * row_height)
        local_bottom = int(local_top + row_height)

        sub = banda[local_top:local_bottom, x0:x1]
        alternativa, estado, intensidad = _clasificar_alternativa(sub, config.answer_labels, config)
        resultados.append(
            Respuesta(
                pregunta=question_index + 1,
                alternativa=alternativa,
                estado=estado,
                intensidad=intensidad,
            )
        )

    return resultados


def _estimacion_ancho_columnas(columnas: Sequence[tuple[int, int, int, int]]) -> int:
    """Calcula un ancho representativo para las columnas a partir de las anclas."""

    if len(columnas) < 2:
        return max(columnas[0][2], 20)
    diffs = [b[0] - a[0] for a, b in zip(columnas, columnas[1:])]
    median = int(np.median(diffs)) if diffs else columnas[0][2]
    return max(int(median * 0.6), 20)


def _clasificar_digito(column_img: np.ndarray) -> int:
    """Divide la columna en 10 celdas y escoge la de mayor tinta."""

    inverted = cv2.bitwise_not(column_img)
    normalized = inverted / 255.0
    height = normalized.shape[0]
    cell_height = height // 10 or 1
    scores = []
    for i in range(10):
        start = i * cell_height
        end = (i + 1) * cell_height if i < 9 else height
        cell = normalized[start:end, :]
        scores.append(cell.mean())
    return int(np.argmax(scores))


def _clasificar_alternativa(
    answer_img: np.ndarray,
    labels: Sequence[str],
    config: OMRConfig,
) -> tuple[str, str, float]:
    """Determina la opción con más tinta y aplica reglas de negocio."""

    rows = len(labels)
    inverted = cv2.bitwise_not(answer_img)
    normalized = inverted / 255.0
    height = normalized.shape[0]
    cell_height = height // rows or 1
    scores = []
    for idx in range(rows):
        start = idx * cell_height
        end = (idx + 1) * cell_height if idx < rows - 1 else height
        cell = normalized[start:end, :]
        scores.append(cell.mean())

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    if best_score < config.cell_activation_threshold:
        return ("-", "SIN RESPUESTA", best_score)

    # Evaluar ambigüedad
    sorted_scores = sorted(scores, reverse=True)
    if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < config.ambiguity_margin:
        return (labels[best_idx], "MULTIPLES", best_score)

    return (labels[best_idx], "OK", best_score)


__all__ = ["procesar_pdf", "OMRConfig"]

