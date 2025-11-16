"""Módulo encargado del procesamiento OMR.

Aunque el algoritmo es configurable, aquí se proporcionan heurísticas
razonables para hojas con columnas de DNI y columnas de respuestas con
marcas de referencia rectangulares en la parte inferior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import cv2  # type: ignore

import cv2
import fitz  # PyMuPDF
import numpy as np

from models import AlumnoHoja, Respuesta


@dataclass
class OMRConfig:
    """Parámetros del algoritmo OMR."""

    dpi: int = 200
    dni_columns: int = 8
    questions: int = 100
    answer_labels: Sequence[str] = ("A", "B", "C", "D", "E")
    # Banda vertical "de respaldo" para el DNI cuando no hay mejores pistas.
    # Por defecto se toma únicamente el 8 % - 32 % de la altura total, que es
    # donde viven los dígitos en las hojas de referencia.
    dni_vertical_band: tuple[float, float] = (0.08, 0.32)
    answer_vertical_band: tuple[float, float] = (0.34, 0.9)
    sync_band_height_ratio: float = 0.18
    min_rect_area_ratio: float = 0.0005
    cell_activation_threshold: float = 0.45
    ambiguity_margin: float = 0.15
    profile_threshold_dni: float = .32
    profile_threshold_respuestas: float = 0.28
    profile_margin_ratio: float = 0.04
    x_band_padding_ratio: float = 0.12
    dni_region_dir: Path | None = Path("d:/depurrar/dni")
    respuestas_region_dir: Path | None = Path("d:/depurrar/respuesta")


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

        dni = _leer_dni(image_bgr, anchors[: config.dni_columns], config, index)
        respuestas = _leer_respuestas(
            image_bgr,
            anchors[config.dni_columns :],
            config,
            index,
        )
        resultados.append(
            AlumnoHoja(pagina=index, dni=dni, respuestas=respuestas, imagen_path=img_path)
        )

    return resultados


def _normalized_inverted(region: np.ndarray) -> np.ndarray:
    """Devuelve la región invertida y normalizada entre 0 y 1.

    Cuando la región está vacía (ancho o alto cero) OpenCV devuelve ``None`` al
    aplicar ``bitwise_not``. Para evitar los errores posteriores y mantener el
    flujo del algoritmo, se devuelve una matriz pequeña llena de ceros.
    """

    if region.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    inverted = cv2.bitwise_not(region)
    return inverted.astype(np.float32) / 255.0


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


def _leer_dni(
    image: np.ndarray,
    columnas: Sequence[tuple[int, int, int, int]],
    config: OMRConfig,
    pagina: int,
) -> str:
    """Lee los dígitos del DNI aprovechando las columnas detectadas."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    default_range = (
        int(h * config.dni_vertical_band[0]),
        int(h * config.dni_vertical_band[1]),
    )
    y0, y1 = _banda_vertical_desde_referencias(
        gray,
        columnas,
        config.profile_threshold_dni,
        config.profile_margin_ratio,
        default_range,
        config.x_band_padding_ratio,
    )
    banda = gray[y0:y1, :]
    _guardar_region_debug(banda, config.dni_region_dir, f"pagina_{pagina:03d}_dni.png")
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
    pagina: int,
) -> List[Respuesta]:
    """Procesa el bloque de preguntas y devuelve ``Respuesta`` por pregunta."""

    if not columnas:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    default_range = (
        int(h * config.answer_vertical_band[0]),
        int(h * config.answer_vertical_band[1]),
    )
    y0, y1 = _banda_vertical_desde_referencias(
        gray,
        columnas,
        config.profile_threshold_respuestas,
        config.profile_margin_ratio,
        default_range,
        config.x_band_padding_ratio,
    )
    if y1 - y0 < 1:
        return []
    banda = gray[y0:y1, :]
    _guardar_region_debug(
        banda,
        config.respuestas_region_dir,
        f"pagina_{pagina:03d}_respuestas.png",
    )
    answers_per_column = int(np.ceil(config.questions / len(columnas)))
    column_width = _estimacion_ancho_columnas(columnas)
    resultados: List[Respuesta] = []
    band_height = banda.shape[0]
    row_boundaries = np.linspace(0, band_height, answers_per_column + 1)

    for question_index in range(config.questions):
        column_index = min(question_index // answers_per_column, len(columnas) - 1)
        col = columnas[column_index]
        x_center = col[0] + col[2] // 2
        x0 = max(x_center - column_width // 2, 0)
        x1 = min(x_center + column_width // 2, w)
        if x1 <= x0:
            x0 = max(min(x_center, w - 1), 0)
            x1 = min(x0 + 1, w)

        row_idx = question_index % answers_per_column
        local_top = int(np.floor(row_boundaries[row_idx]))
        local_bottom = int(np.ceil(row_boundaries[row_idx + 1]))
        if local_bottom <= local_top:
            local_bottom = min(local_top + 1, band_height)

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

    normalized = _normalized_inverted(column_img)
    height = normalized.shape[0]
    boundaries = np.linspace(0, height, 11, dtype=int)
    scores = []
    for i in range(10):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            end = min(start + 1, height)
        cell = normalized[start:end, :]
        scores.append(float(cell.mean()) if cell.size else 0.0)
    return int(np.argmax(scores))


def _clasificar_alternativa(
    answer_img: np.ndarray,
    labels: Sequence[str],
    config: OMRConfig,
) -> tuple[str, str, float]:
    """Determina la opción con más tinta y aplica reglas de negocio."""

    rows = len(labels)
    normalized = _normalized_inverted(answer_img)
    height = normalized.shape[0]
    boundaries = np.linspace(0, height, rows + 1, dtype=int)
    scores = []
    for idx in range(rows):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if end <= start:
            end = min(start + 1, height)
        cell = normalized[start:end, :]
        scores.append(float(cell.mean()) if cell.size else 0.0)

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    if best_score < config.cell_activation_threshold:
        return ("-", "SIN RESPUESTA", best_score)

    # Evaluar ambigüedad
    sorted_scores = sorted(scores, reverse=True)
    if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < config.ambiguity_margin:
        return (labels[best_idx], "MULTIPLES", best_score)

    return (labels[best_idx], "OK", best_score)


def _banda_vertical_desde_referencias(
    gray: np.ndarray,
    columnas: Sequence[tuple[int, int, int, int]],
    threshold: float,
    margin_ratio: float,
    default_range: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int]:
    """Calcula automáticamente la banda vertical donde viven las burbujas."""

    if not columnas:
        return default_range

    height, width = gray.shape
    x_min = min(col[0] for col in columnas)
    x_max = max(col[0] + col[2] for col in columnas)
    padding = int((x_max - x_min) * padding_ratio)
    x0 = max(x_min - padding, 0)
    x1 = min(x_max + padding, width)
    if x1 <= x0:
        return default_range

    roi = gray[:, x0:x1]
    inverted = cv2.bitwise_not(roi)
    profile = inverted.mean(axis=1)
    span = profile.max() - profile.min()
    if span <= 1e-6:
        return default_range
    normalized = (profile - profile.min()) / span
    mask = normalized > threshold
    start, end = _mayor_segmento_activo(mask)
    if end - start <= 0:
        return default_range

    margin = int((end - start) * margin_ratio)
    return (max(start - margin, 0), min(end + margin, height))


def _mayor_segmento_activo(mask: np.ndarray) -> tuple[int, int]:
    """Devuelve el tramo más largo con ``True`` dentro de ``mask``."""

    best_start = 0
    best_len = 0
    current_start = None

    for idx, value in enumerate(mask.tolist()):
        if value:
            if current_start is None:
                current_start = idx
            continue
        if current_start is not None:
            length = idx - current_start
            if length > best_len:
                best_len = length
                best_start = current_start
            current_start = None

    if current_start is not None:
        length = len(mask) - current_start
        if length > best_len:
            best_len = length
            best_start = current_start

    return best_start, best_start + best_len


def _guardar_region_debug(region: np.ndarray, directory: Path | None, filename: str) -> None:
    """Guarda la imagen de una región de interés si se proporcionó un directorio."""

    if directory is None:
        return

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / filename), region)


__all__ = ["procesar_pdf", "OMRConfig"]

