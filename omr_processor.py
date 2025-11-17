from __future__ import annotations

"""
Módulo encargado del procesamiento OMR y utilidades de depuración.

Adaptado a la hoja de ACADEMIA Lumbreras:

- Usa las barras rectangulares inferiores como anclas de sincronización.
- Las 8 primeras barras (de izquierda a derecha) corresponden al bloque de DNI.
- El resto de barras se agrupa de 5 en 5 para obtener 4 columnas de preguntas.
- Las alternativas A–E de cada pregunta se leen en HORIZONTAL.

Reglas de negocio importantes:
- Si se detectan marcaciones múltiples (más de una alternativa fuerte),
  se considera la pregunta como EN BLANCO:
    -> alternativa "-" y estado "SIN RESPUESTA".
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2  # type: ignore
import fitz  # PyMuPDF
import numpy as np

from models import AlumnoHoja, Respuesta


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------


@dataclass
class OMRConfig:
    """Parámetros del algoritmo OMR para la hoja de Academia Lumbreras."""

    # Resolución de renderizado del PDF
    dpi: int = 200

    # Número de columnas de dígitos en el bloque de DNI (8 dígitos)
    dni_columns: int = 8

    # Número total de preguntas en la hoja
    questions: int = 100

    # Etiquetas de alternativas
    answer_labels: Sequence[str] = ("A", "B", "C", "D", "E")

    # Mapeo de filas (de arriba a abajo) a dígitos reales.
    # Si en tu hoja el 0 está abajo y el 9 arriba, cambia esto a:
    # dni_digit_values = (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
    dni_digit_values: Sequence[int] = tuple(range(10))

    # Banda vertical aproximada del bloque de DNI (relativa a la altura total)
    # Ajusta estos dos valores si ves en las imágenes debug que la banda no
    # coincide con el DNI real. El rango por defecto evita la zona superior
    # donde suele escribirse el número manual y se extiende más abajo para
    # cubrir las burbujas 0–9 completas.
    dni_vertical_band: tuple[float, float] = (0.12, 0.64)

    # Banda vertical aproximada de las respuestas (relativa a la altura total)
    answer_vertical_band: tuple[float, float] = (0.12, 0.94)

    # Porcentaje de altura usada como banda inferior de sincronización
    sync_band_height_ratio: float = 0.18

    # Área mínima de rectángulos de sincronización respecto al área total
    min_rect_area_ratio: float = 0.00018

    # --------- Umbrales de respuestas ---------

    # Umbral mínimo para considerar que una burbuja está marcada
    # (score combinado en [0, 1]).
    cell_activation_threshold: float = 0.25

    # Relación máxima second_best / best para aceptar la marca.
    # Si second >= multi_mark_ratio * best => se considera EN BLANCO.
    multi_mark_ratio: float = 0.80

    # Margen de recorte lateral para evitar tomar el número de la pregunta
    # (parte izquierda suele tener texto/números, burbujas suelen estar más a la derecha).
    answer_left_margin_ratio: float = 0.15

    # Estos se mantienen por compatibilidad (no se usan directamente ahora)
    profile_threshold_dni: float = 0.32
    profile_threshold_respuestas: float = 0.28
    profile_margin_ratio: float = 0.04
    x_band_padding_ratio: float = 0.12

    # Directorios opcionales para guardar recortes de depuración
    dni_region_dir: Path | None = None
    respuestas_region_dir: Path | None = None


# ---------------------------------------------------------------------------
# Punto principal de procesamiento (para la UI)
# ---------------------------------------------------------------------------


def procesar_pdf(
    pdf_path: str | Path,
    cache_dir: str | Path | None = None,
    config: OMRConfig | None = None,
) -> List[AlumnoHoja]:
    """Procesa todas las páginas de un PDF y devuelve una lista de AlumnoHoja."""

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
            # No hay suficientes anclas ni siquiera para el DNI.
            resultados.append(
                AlumnoHoja(
                    pagina=index,
                    dni="NO DETECTADO",
                    respuestas=[],
                    imagen_path=img_path,
                )
            )
            continue

        # Las primeras N barras son las del bloque de DNI.
        dni_anchors = anchors[: config.dni_columns]

        # El resto pertenecen al bloque de respuestas.
        answer_anchors_raw = anchors[config.dni_columns :]

        # En la hoja hay 5 barras por columna de preguntas;
        # las compactamos en columnas lógicas.
        answer_columns = _compactar_columnas_respuestas(
            answer_anchors_raw,
            group_size=len(config.answer_labels),
        )

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        dni = _leer_dni(gray, dni_anchors, config, index)
        respuestas = _leer_respuestas(gray, answer_columns, config, index)

        resultados.append(
            AlumnoHoja(
                pagina=index,
                dni=dni,
                respuestas=respuestas,
                imagen_path=img_path,
            )
        )

    return resultados


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------


def _normalized_inverted(region: np.ndarray) -> np.ndarray:
    """Devuelve la región invertida y normalizada entre 0 y 1."""
    if region.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    inverted = cv2.bitwise_not(region)
    return inverted.astype(np.float32) / 255.0


def _ajustar_banda_vertical(
    gray_band: np.ndarray,
    x_ranges: Sequence[tuple[int, int]],
    min_height: int,
    activation_ratio: float = 0.12,
    smooth_kernel: int = 9,
    margin: int = 4,
) -> tuple[int, int]:
    """Ajusta dinámicamente la banda vertical basándose en el perfil de tinta.

    Se calcula un perfil vertical promedio (invirtiendo la imagen) usando sólo las
    columnas indicadas en ``x_ranges``. A partir de ese perfil se busca la zona con
    tinta significativa y se devuelve un rango [top, bottom) refinado.
    """

    h_band, w_band = gray_band.shape[:2]
    if h_band <= 0 or w_band <= 0:
        return (0, h_band)

    slices = []
    for x0, x1 in x_ranges:
        x0 = max(0, min(x0, w_band))
        x1 = max(x0 + 1, min(x1, w_band))
        if x1 <= x0:
            continue
        slices.append(gray_band[:, x0:x1])

    if not slices:
        return (0, h_band)

    stacked = np.concatenate(slices, axis=1)
    inv = cv2.bitwise_not(stacked).astype(np.float32) / 255.0
    profile = inv.mean(axis=1)

    if profile.size == 0:
        return (0, h_band)

    kernel = max(3, smooth_kernel | 1)  # impar
    profile_smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, kernel), 0).flatten()

    max_val = float(profile_smooth.max())
    if max_val <= 0:
        return (0, h_band)

    threshold = max_val * activation_ratio
    active_idx = np.where(profile_smooth >= threshold)[0]
    if active_idx.size == 0:
        return (0, h_band)

    top = int(active_idx.min())
    bottom = int(active_idx.max())

    top = max(top - margin, 0)
    bottom = min(bottom + margin, h_band - 1)

    if bottom - top + 1 < min_height:
        center = (top + bottom) // 2
        half = max(min_height // 2, 1)
        top = max(center - half, 0)
        bottom = min(center + half, h_band - 1)

    return (top, bottom + 1)


def _render_page(
    page: fitz.Page,
    cache_dir: Path,
    index: int,
    dpi: int,
) -> tuple[np.ndarray, Path]:
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


# ---------------------------------------------------------------------------
# Detección de rectángulos de sincronización
# ---------------------------------------------------------------------------


def _detectar_rectangulos_sync(
    image: np.ndarray,
    config: OMRConfig,
) -> List[tuple[int, int, int, int]]:
    """Detecta las barras rectangulares de sincronización en la banda inferior."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    band_start = int(h * (1 - config.sync_band_height_ratio))
    bottom_band = gray[band_start:, :]

    blur = cv2.GaussianBlur(bottom_band, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    rects: List[tuple[int, int, int, int]] = []
    min_area = max(int(config.min_rect_area_ratio * w * h), 60)
    band_h = bottom_band.shape[0]

    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        if area < min_area:
            continue

        aspect = rw / rh if rh else 0.0
        if aspect < 0.25 or aspect > 1.4:
            continue

        # Sólo barras realmente cerca del borde inferior
        bottom = y + rh
        if bottom < int(0.7 * band_h):   # 70 % de la banda inferior
            continue

        rects.append((x, y + band_start, rw, rh))

    rects.sort(key=lambda r: r[0])
    return rects


def _compactar_columnas_respuestas(
    raw_anchors: Sequence[tuple[int, int, int, int]],
    group_size: int,
) -> List[tuple[int, int, int, int]]:
    """Agrupa las barras de respuestas en columnas lógicas (bloques de 5)."""

    if not raw_anchors:
        return []

    anchors = list(raw_anchors)
    if len(anchors) < group_size:
        # No hay suficientes barras para una columna, devolver tal cual.
        return list(anchors)

    columns: List[tuple[int, int, int, int]] = []
    if len(anchors) % group_size == 0:
        for i in range(0, len(anchors), group_size):
            group = anchors[i : i + group_size]
            xs = [a[0] for a in group]
            ys = [a[1] for a in group]
            x_ends = [a[0] + a[2] for a in group]
            y_ends = [a[1] + a[3] for a in group]

            x = min(xs)
            y = min(ys)
            w = max(x_ends) - x
            h = max(y_ends) - y

            columns.append((x, y, w, h))
    else:
        columns = anchors

    return columns


# ---------------------------------------------------------------------------
# Lectura de DNI
# ---------------------------------------------------------------------------


def _leer_dni(
    gray: np.ndarray,
    columnas: Sequence[tuple[int, int, int, int]],
    config: OMRConfig,
    pagina: int,
) -> str:
    """Lee los 8 dígitos del DNI usando las columnas de barras inferiores."""

    h, w = gray.shape

    # Banda general del bloque de DNI (aproximada, luego afinada por perfil)
    band_top = int(h * config.dni_vertical_band[0])
    band_bottom = int(h * config.dni_vertical_band[1])
    band_top = max(0, min(band_top, h - 1))
    band_bottom = max(band_top + 1, min(band_bottom, h))

    banda = gray[band_top:band_bottom, :]
    band_height = banda.shape[0]

    digits: List[str] = []
    column_width = _estimacion_ancho_columnas(columnas)

    x_ranges: List[tuple[int, int]] = []
    for col in columnas:
        x_center = col[0] + col[2] // 2
        x0 = max(x_center - column_width // 2, 0)
        x1 = min(x_center + column_width // 2, w)
        x_ranges.append((x0, x1))

    # Afinar la banda vertical usando el perfil de tinta real de las columnas
    adj_top, adj_bottom = _ajustar_banda_vertical(
        banda,
        x_ranges,
        min_height=max(int(band_height * 0.8), 140),
        activation_ratio=0.12,
    )
    banda = banda[adj_top:adj_bottom, :]

    for x0, x1 in x_ranges:
        sub = banda[:, x0:x1]
        digit = _clasificar_digito(sub, config)
        digits.append(str(digit))

    return "".join(digits)


def _clasificar_digito(
    column_img: np.ndarray,
    config: OMRConfig,
) -> int:
    """Divide la columna en 10 celdas horizontales y escoge la de mayor tinta."""

    normalized = _normalized_inverted(column_img)
    height = normalized.shape[0]
    if height <= 0:
        return 0

    boundaries = np.linspace(0, height, 11, dtype=int)
    scores: List[float] = []

    for i in range(10):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            end = min(start + 1, height)
        cell = normalized[start:end, :]
        scores.append(float(cell.mean()) if cell.size else 0.0)

    best_idx = int(np.argmax(scores))
    # Mapear índice de fila (0–9) al dígito real según la configuración.
    if 0 <= best_idx < len(config.dni_digit_values):
        return int(config.dni_digit_values[best_idx])
    return int(best_idx)


# ---------------------------------------------------------------------------
# Lectura de respuestas
# ---------------------------------------------------------------------------


def _leer_respuestas(
    gray: np.ndarray,
    columnas: Sequence[tuple[int, int, int, int]],
    config: OMRConfig,
    pagina: int,
) -> List[Respuesta]:

    if not columnas:
        return []

    h, w = gray.shape

    # Banda ajustada de respuestas
    y0 = int(h * config.answer_vertical_band[0])
    y1 = int(h * config.answer_vertical_band[1])
    if y1 - y0 < 1:
        return []

    banda = gray[y0:y1, :]
    band_height = banda.shape[0]

    answers_per_column = int(np.ceil(config.questions / len(columnas)))
    column_width = _estimacion_ancho_columnas(columnas)
    resultados: List[Respuesta] = []

    x_ranges: List[tuple[int, int]] = []
    for col in columnas:
        x_center = col[0] + col[2] // 2
        x0 = max(x_center - column_width // 2, 0)
        x1 = min(x_center + column_width // 2, w)
        x_ranges.append((x0, x1))

    # Afinar la banda vertical en función de la tinta real de las columnas
    adj_top, adj_bottom = _ajustar_banda_vertical(
        banda,
        x_ranges,
        min_height=max(int(band_height * 0.85), 400 // max(len(columnas), 1)),
        activation_ratio=0.10,
    )
    banda = banda[adj_top:adj_bottom, :]
    band_height = banda.shape[0]

    # Limites “ideales” de cada fila (0..band_height)
    row_boundaries = np.linspace(0, band_height, answers_per_column + 1)

    # Calculamos ventanas verticales por columna basándonos en el perfil real
    # de tinta. Esto permite que las líneas azules (grid de depuración) pasen
    # exactamente junto a las burbujas marcadas, incluso con ligeras
    # desalineaciones o inclinaciones de la hoja.
    row_windows_por_columna: List[List[tuple[int, int]]] = []
    for x0, x1 in x_ranges:
        row_windows_por_columna.append(
            _calcular_filas_columna(
                banda,
                (x0, x1),
                row_boundaries,
                config,
            )
        )

    for question_index in range(config.questions):
        # qué columna lógica (0..3)
        column_index = min(
            question_index // answers_per_column,
            len(columnas) - 1,
        )
        x0, x1 = x_ranges[column_index]

        # fila dentro de la columna (0..answers_per_column-1) y ventana afinada
        row_idx = question_index % answers_per_column
        local_top, local_bottom = row_windows_por_columna[column_index][row_idx]

        sub = banda[local_top:local_bottom, x0:x1]

        alternativa, estado, intensidad = _clasificar_alternativa(
            sub,
            config.answer_labels,
            config,
        )

        resultados.append(
            Respuesta(
                pregunta=question_index + 1,
                alternativa=alternativa,
                estado=estado,
                intensidad=intensidad,
            )
        )

    return resultados



def _estimacion_ancho_columnas(
    columnas: Sequence[tuple[int, int, int, int]],
) -> int:
    """Ancho representativo a partir del tamaño real de las columnas."""
    if not columnas:
        return 40
    widths = [c[2] for c in columnas]
    median_width = int(np.median(widths))
    return max(int(median_width * 1.0), 40)


def _calcular_filas_columna(
    banda: np.ndarray,
    x_range: tuple[int, int],
    base_boundaries: np.ndarray,
    config: OMRConfig,
) -> List[tuple[int, int]]:
    """Devuelve [top, bottom) por fila afinando los límites con el perfil real."""

    h_band, _ = banda.shape
    x0, x1 = x_range
    x0 = max(0, min(x0, banda.shape[1]))
    x1 = max(x0 + 1, min(x1, banda.shape[1]))

    col_gray = banda[:, x0:x1]
    left_margin = int(col_gray.shape[1] * config.answer_left_margin_ratio)
    left_margin = min(left_margin, col_gray.shape[1] - 1) if col_gray.shape[1] > 0 else 0
    right_img = col_gray[:, left_margin:] if col_gray.size else col_gray

    if right_img.size:
        profile = _normalized_inverted(right_img).mean(axis=1)
        profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).flatten()
    else:
        profile = np.zeros((h_band,), dtype=np.float32)

    # Afinamos cada límite buscando los valles (zonas blancas) del perfil.
    base_boundaries = np.asarray(base_boundaries, dtype=float)
    row_height_est = float(np.median(np.diff(base_boundaries))) if len(base_boundaries) > 1 else 0.0
    search_radius_default = max(int(row_height_est * 0.35), 5)

    adjusted_boundaries: List[int] = [0]
    for idx in range(1, len(base_boundaries) - 1):
        predicted_boundary = float(base_boundaries[idx])
        search_radius = search_radius_default
        if idx < len(base_boundaries) - 1:
            next_height = float(base_boundaries[idx + 1] - base_boundaries[idx - 1]) / 2.0
            search_radius = max(int(next_height * 0.35), search_radius)

        start = int(max(predicted_boundary - search_radius, adjusted_boundaries[-1]))
        end = int(min(predicted_boundary + search_radius, h_band - 1))
        if end <= start:
            end = min(start + 1, h_band - 1)

        segment = profile[start : end + 1]
        offset = int(np.argmin(segment)) if segment.size else 0
        boundary = start + offset

        boundary = max(boundary, adjusted_boundaries[-1] + 1)
        adjusted_boundaries.append(boundary)

    adjusted_boundaries.append(h_band)

    row_windows: List[tuple[int, int]] = []
    for i in range(len(adjusted_boundaries) - 1):
        top = adjusted_boundaries[i]
        bottom = adjusted_boundaries[i + 1]
        if bottom <= top:
            bottom = min(top + 1, h_band)
        row_windows.append((top, bottom))

    return row_windows


def _clasificar_alternativa(
    answer_img: np.ndarray,
    labels: Sequence[str],
    config: OMRConfig,
) -> tuple[str, str, float]:
    """Determina la opción con más tinta y aplica reglas de negocio.

    Reglas:
    - Nunca se devuelve "MULTIPLES":
      * Si hay ambigüedad (más de una alternativa fuerte), se considera EN BLANCO.
    - Sólo se devuelve:
      * ("<letra>", "OK", score) o
      * ("-", "SIN RESPUESTA", score)

    Técnica:
    - Se recorta un margen izquierdo para evitar usar el número de la pregunta.
    - Se combina gris invertido + binarización local (Otsu).
    """

    if answer_img.size == 0:
        return ("-", "SIN RESPUESTA", 0.0)

    gray = answer_img
    h_sub, w_sub = gray.shape
    if w_sub <= 0 or h_sub <= 0:
        return ("-", "SIN RESPUESTA", 0.0)

    # Recorte lateral para evitar el número de pregunta
    left_margin = int(w_sub * config.answer_left_margin_ratio)
    if left_margin >= w_sub:
        left_margin = 0
    right_img = gray[:, left_margin:]
    h_use, w_use = right_img.shape

    if w_use <= 0:
        # Fallback sin recorte
        right_img = gray
        h_use, w_use = right_img.shape

    # Invertido normalizado
    inv = cv2.bitwise_not(right_img).astype(np.float32) / 255.0

    # Binarización local (Otsu) invertida
    blur = cv2.GaussianBlur(right_img, (5, 5), 0)
    _, bin_inv = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    cols = len(labels)
    boundaries = np.linspace(0, w_use, cols + 1, dtype=int)

    scores: List[float] = []
    for idx in range(cols):
        x0 = boundaries[idx]
        x1 = boundaries[idx + 1]
        if x1 <= x0:
            x1 = min(x0 + 1, w_use)

        cell_gray = inv[:, x0:x1]
        cell_bin = bin_inv[:, x0:x1]

        score_gray = float(cell_gray.mean()) if cell_gray.size else 0.0
        score_bin = float(cell_bin.mean()) / 255.0 if cell_bin.size else 0.0

        # Peso más la binaria (tinta real), la gris sirve como refinamiento
        score = 0.3 * score_gray + 0.7 * score_bin
        scores.append(score)

    scores_arr = np.array(scores, dtype=float)
    best_idx = int(scores_arr.argmax())
    best_score = float(scores_arr[best_idx])

    # Caso 1: muy poca tinta en general → SIN RESPUESTA
    if best_score < config.cell_activation_threshold:
        return ("-", "SIN RESPUESTA", best_score)

    # Caso 2: múltiples marcas (ambigüedad) → EN BLANCO
    sorted_scores = sorted(scores_arr, reverse=True)
    second_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0

    if best_score > 0.0 and second_score >= config.multi_mark_ratio * best_score:
        # Regla que pediste: se considera en blanco
        return ("-", "SIN RESPUESTA", best_score)

    # Caso normal: una única alternativa claramente dominante
    return (labels[best_idx], "OK", best_score)


# ---------------------------------------------------------------------------
# Utilidades de depuración (imágenes intermedias)
# ---------------------------------------------------------------------------


def _debug_draw_anchors(
    image_bgr: np.ndarray,
    anchors: List[tuple[int, int, int, int]],
    config: OMRConfig,
    out_dir: Path,
) -> None:
    """Dibuja todas las anclas detectadas (barras inferiores)."""

    dbg = image_bgr.copy()
    for i, (x, y, w, h) in enumerate(anchors):
        if i < config.dni_columns:
            color = (255, 0, 0)  # azul: DNI
        else:
            color = (0, 255, 0)  # verde: respuestas
        cv2.rectangle(dbg, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            dbg,
            str(i),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "01_anchors.png"), dbg)


def _debug_dni_band(
    image_bgr: np.ndarray,
    gray: np.ndarray,
    dni_anchors: List[tuple[int, int, int, int]],
    config: OMRConfig,
    out_dir: Path,
) -> None:
    """Guarda banda de DNI y columnas con las 10 filas marcadas."""

    h, w = gray.shape
    band_top = int(h * config.dni_vertical_band[0])
    band_bottom = int(h * config.dni_vertical_band[1])
    band_top = max(0, min(band_top, h - 1))
    band_bottom = max(band_top + 1, min(band_bottom, h))
    band_height = max(band_bottom - band_top, 1)

    # Imagen de la banda de DNI
    banda_gray = gray[band_top:band_bottom, :]
    column_width = _estimacion_ancho_columnas(dni_anchors)

    x_ranges: List[tuple[int, int]] = []
    for col in dni_anchors:
        x_center = col[0] + col[2] // 2
        x0 = max(x_center - column_width // 2, 0)
        x1 = min(x_center + column_width // 2, w)
        x_ranges.append((x0, x1))

    adj_top, adj_bottom = _ajustar_banda_vertical(
        banda_gray,
        x_ranges,
        min_height=max(int(band_height * 0.8), 140),
        activation_ratio=0.12,
    )

    dni_band_color = image_bgr[band_top + adj_top : band_top + adj_bottom, :].copy()
    cv2.imwrite(str(out_dir / "02_dni_band.png"), dni_band_color)

    banda_gray = banda_gray[adj_top:adj_bottom, :]

    for col_idx, (x0, x1) in enumerate(x_ranges):
        col_img = banda_gray[:, x0:x1]

        # columna cruda
        cv2.imwrite(str(out_dir / f"03_dni_col_{col_idx}.png"), col_img)

        # columna con divisiones de filas
        h_col_img = col_img.shape[0]
        boundaries = np.linspace(0, h_col_img, 11, dtype=int)
        col_vis = cv2.cvtColor(col_img, cv2.COLOR_GRAY2BGR)
        for b in boundaries:
            cv2.line(col_vis, (0, b), (col_vis.shape[1] - 1, b), (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / f"04_dni_col_{col_idx}_grid.png"), col_vis)


def _debug_answers_band_and_grid(
    image_bgr: np.ndarray,
    gray: np.ndarray,
    answer_columns: List[tuple[int, int, int, int]],
    config: OMRConfig,
    out_dir: Path,
) -> None:
    """Guarda banda de respuestas y grid de columnas/filas."""

    h, w = gray.shape
    y0 = int(h * config.answer_vertical_band[0])
    y1 = int(h * config.answer_vertical_band[1])
    if y1 - y0 < 1:
        return

    banda_gray = gray[y0:y1, :]
    band_color = image_bgr[y0:y1, :].copy()
    band_height = band_color.shape[0]

    column_width = _estimacion_ancho_columnas(answer_columns)
    x_ranges: List[tuple[int, int]] = []
    for col in answer_columns:
        x_center = col[0] + col[2] // 2
        x0 = max(x_center - column_width // 2, 0)
        x1 = min(x_center + column_width // 2, band_color.shape[1])
        x_ranges.append((x0, x1))

    adj_top, adj_bottom = _ajustar_banda_vertical(
        banda_gray,
        x_ranges,
        min_height=max(int(band_height * 0.85), 400 // max(len(answer_columns), 1)),
        activation_ratio=0.10,
    )

    banda_gray = banda_gray[adj_top:adj_bottom, :]
    band_color = band_color[adj_top:adj_bottom, :]
    band_height = band_color.shape[0]
    answers_per_column = int(np.ceil(config.questions / len(answer_columns)))
    row_boundaries = np.linspace(0, band_height, answers_per_column + 1, dtype=int)
    row_windows_por_columna: List[List[tuple[int, int]]] = []
    for x0, x1 in x_ranges:
        row_windows_por_columna.append(
            _calcular_filas_columna(
                banda_gray,
                (x0, x1),
                row_boundaries,
                config,
            )
        )

    for col_idx, (x0, x1) in enumerate(x_ranges):
        cv2.rectangle(band_color, (x0, 0), (x1, band_height), (0, 255, 0), 1)
        cv2.putText(
            band_color,
            f"C{col_idx}",
            (x0 + 2, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        row_windows = row_windows_por_columna[col_idx]
        for top, _ in row_windows:
            cv2.line(band_color, (x0, top), (x1, top), (255, 0, 0), 1)
        if row_windows:
            cv2.line(
                band_color,
                (x0, row_windows[-1][1]),
                (x1, row_windows[-1][1]),
                (255, 0, 0),
                1,
            )

    cv2.imwrite(str(out_dir / "05_answers_grid.png"), band_color)


def _debug_question_row(
    gray: np.ndarray,
    answer_columns: List[tuple[int, int, int, int]],
    config: OMRConfig,
    pagina: int,
    question_index: int,
    out_dir: Path,
) -> None:
    """Debug de una sola pregunta: guarda recorte y scores A–E."""

    h, w = gray.shape
    y0 = int(h * config.answer_vertical_band[0])
    y1 = int(h * config.answer_vertical_band[1])
    if y1 - y0 < 1:
        return

    banda = gray[y0:y1, :]
    band_height = banda.shape[0]
    answers_per_column = int(np.ceil(config.questions / len(answer_columns)))
    column_width = _estimacion_ancho_columnas(answer_columns)

    x_ranges: List[tuple[int, int]] = []
    for col in answer_columns:
        x_center = col[0] + col[2] // 2
        x0 = max(x_center - column_width // 2, 0)
        x1 = min(x_center + column_width // 2, w)
        x_ranges.append((x0, x1))

    adj_top, adj_bottom = _ajustar_banda_vertical(
        banda,
        x_ranges,
        min_height=max(int(band_height * 0.85), 400 // max(len(answer_columns), 1)),
        activation_ratio=0.10,
    )

    banda = banda[adj_top:adj_bottom, :]
    band_height = banda.shape[0]
    row_boundaries = np.linspace(0, band_height, answers_per_column + 1)
    row_windows_por_columna: List[List[tuple[int, int]]] = []
    for xr in x_ranges:
        row_windows_por_columna.append(
            _calcular_filas_columna(
                banda,
                xr,
                row_boundaries,
                config,
            )
        )

    col_index = min(
        question_index // answers_per_column,
        len(answer_columns) - 1,
    )
    x0, x1 = x_ranges[col_index]

    row_idx = question_index % answers_per_column
    local_top, local_bottom = row_windows_por_columna[col_index][row_idx]

    sub = banda[local_top:local_bottom, x0:x1]
    cv2.imwrite(str(out_dir / f"Q{question_index+1:03d}_row.png"), sub)

    # Calcular scores A–E con la misma lógica que _clasificar_alternativa
    # para ver numéricamente qué está pasando.
    if sub.size == 0:
        return

    gray_sub = sub
    h_sub, w_sub = gray_sub.shape
    left_margin = int(w_sub * config.answer_left_margin_ratio)
    left_margin = min(left_margin, w_sub - 1) if w_sub > 0 else 0
    right_img = gray_sub[:, left_margin:]
    if right_img.size == 0:
        right_img = gray_sub

    inv = cv2.bitwise_not(right_img).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(right_img, (5, 5), 0)
    _, bin_inv = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    cols = len(config.answer_labels)
    h_use, w_use = right_img.shape
    boundaries = np.linspace(0, w_use, cols + 1, dtype=int)

    scores: List[float] = []
    for idx in range(cols):
        x0c = boundaries[idx]
        x1c = boundaries[idx + 1]
        if x1c <= x0c:
            x1c = min(x0c + 1, w_use)

        cell_gray = inv[:, x0c:x1c]
        cell_bin = bin_inv[:, x0c:x1c]
        score_gray = float(cell_gray.mean()) if cell_gray.size else 0.0
        score_bin = float(cell_bin.mean()) / 255.0 if cell_bin.size else 0.0
        score = 0.3 * score_gray + 0.7 * score_bin
        scores.append(score)

    scores_arr = np.array(scores, dtype=float)
    best_idx = int(scores_arr.argmax())
    best_score = float(scores_arr[best_idx])
    sorted_scores = sorted(scores_arr, reverse=True)
    second_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0

    txt_path = out_dir / f"Q{question_index+1:03d}_scores.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Pregunta {question_index+1} (columna {col_index})\n")
        f.write(f"best_idx={best_idx}, best_score={best_score:.4f}\n")
        f.write(f"second_score={second_score:.4f}\n")
        for label, s in zip(config.answer_labels, scores_arr):
            f.write(f"{label}: {s:.4f}\n")


# ---------------------------------------------------------------------------
# Modo debug por consola
# ---------------------------------------------------------------------------


def _debug_page(
    pdf_path: Path,
    page_number: int,
    config: OMRConfig | None = None,
    questions_to_debug: Sequence[int] | None = None,
) -> None:
    """
    Genera imágenes de depuración para una página concreta.

    - 01_anchors.png          -> barras de sincronización detectadas
    - 02_dni_band.png         -> banda vertical usada para el DNI
    - 03/04_dni_col_*.png     -> columnas de DNI y rejilla de 10 filas
    - 05_answers_grid.png     -> banda de respuestas con columnas y filas
    - QXXX_row.png            -> recorte de una pregunta concreta
    - QXXX_scores.txt         -> scores numéricos A–E para esa pregunta
    """

    config = config or OMRConfig()
    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > doc.page_count:
        raise ValueError(f"Página fuera de rango: {page_number}")

    page = doc[page_number - 1]
    out_dir = pdf_path.parent / f"_debug_pagina_{page_number:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr, _ = _render_page(page, out_dir, page_number, config.dpi)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    anchors = _detectar_rectangulos_sync(image_bgr, config)
    _debug_draw_anchors(image_bgr, anchors, config, out_dir)

    if len(anchors) < config.dni_columns:
        print("No hay suficientes anclas para DNI en modo debug.")
        return

    dni_anchors = anchors[: config.dni_columns]
    answer_anchors_raw = anchors[config.dni_columns :]
    answer_columns = _compactar_columnas_respuestas(
        answer_anchors_raw,
        group_size=len(config.answer_labels),
    )

    _debug_dni_band(image_bgr, gray, dni_anchors, config, out_dir)
    _debug_answers_band_and_grid(image_bgr, gray, answer_columns, config, out_dir)

    if questions_to_debug:
        for q in questions_to_debug:
            if 1 <= q <= config.questions:
                _debug_question_row(
                    gray,
                    answer_columns,
                    config,
                    page_number,
                    q - 1,  # index base 0
                    out_dir,
                )


# ---------------------------------------------------------------------------
# Entry point CLI (solo para debug)
# ---------------------------------------------------------------------------


def _print_usage() -> None:
    print(
        "Uso (modo debug):\n"
        "  python omr_processor.py RUTA_PDF NUM_PAGINA [lista_preguntas]\n\n"
        "Ejemplo:\n"
        "  python omr_processor.py ONE.pdf 1 1 2 10 25 50 75 100\n\n"
        "Esto generará una carpeta _debug_pagina_001 con PNGs y TXT de depuración."
    )


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) < 3:
        _print_usage()
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    try:
        page_number = int(sys.argv[2])
    except ValueError:
        _print_usage()
        sys.exit(1)

    questions_to_debug: List[int] = []
    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            try:
                q = int(arg)
                questions_to_debug.append(q)
            except ValueError:
                continue

    cfg = OMRConfig()
    _debug_page(pdf_path, page_number, cfg, questions_to_debug)

__all__ = ["procesar_pdf", "OMRConfig"]
