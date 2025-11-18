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

     # --- Parámetros para detectar filas de burbujas ---
    # Tamaño del kernel vertical para suavizar el perfil (debe ser impar)
    row_profile_smooth_kernel: int = 21
    # Umbral relativo para decidir dónde hay "tinta suficiente"
    row_profile_activation_ratio: float = 0.35
    # Mínimo de píxeles consecutivos por grupo para considerarlo una fila
    row_profile_min_run: int = 3
    
    
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
    answer_vertical_band: tuple[float, float] = (0.10, 0.94)

    # Porcentaje de altura usada como banda inferior de sincronización
    sync_band_height_ratio: float = 0.18

    # Área mínima de rectángulos de sincronización respecto al área total
    min_rect_area_ratio: float = 0.00018

    # --------- Umbrales de respuestas ---------

    # Umbral mínimo para considerar que una burbuja está marcada
    # (score combinado en [0, 1]).
    cell_activation_threshold: float = 0.22

    # Relación máxima second_best / best para aceptar la marca.
    # Si second >= multi_mark_ratio * best => se considera EN BLANCO.
    multi_mark_ratio: float = 0.90

    # Margen de recorte lateral para evitar tomar el número de la pregunta
    # (parte izquierda suele tener texto/números, burbujas suelen estar más a la derecha).
    answer_left_margin_ratio: float = 0.02

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

def _detectar_centros_filas(
    banda_gray: np.ndarray,
    x_range: tuple[int, int],
    expected_rows: int,
    config: OMRConfig,
) -> np.ndarray:
    """
    Detecta los centros verticales de las filas de burbujas usando el perfil de tinta
    en una sola columna de respuestas (x_range).
    """

    h_band, w_band = banda_gray.shape[:2]
    if h_band <= 0 or w_band <= 0:
        return np.array([], dtype=int)

    x0, x1 = x_range
    x0 = max(0, min(x0, w_band - 1))
    x1 = max(x0 + 1, min(x1, w_band))

    col_img = banda_gray[:, x0:x1]

    # Perfil de tinta: invertimos para que más tinta = valor mayor
    inv = cv2.bitwise_not(col_img).astype(np.float32) / 255.0
    profile = inv.mean(axis=1)  # promedio por fila

    if profile.size == 0:
        return np.array([], dtype=int)

    # Suavizado vertical
    kernel = max(3, config.row_profile_smooth_kernel | 1)  # impar
    profile_smooth = cv2.GaussianBlur(
        profile.reshape(-1, 1),
        (1, kernel),
        0,
    ).flatten()

    max_val = float(profile_smooth.max())
    if max_val <= 0:
        return np.array([], dtype=int)

    # Umbral relativo de activación
    threshold = max_val * config.row_profile_activation_ratio
    mask = profile_smooth >= threshold
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.array([], dtype=int)

    # Agrupamos índices consecutivos (cada grupo ≈ una fila)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)

    centers: list[int] = []
    for g in groups:
        if g.size < config.row_profile_min_run:
            continue
        centers.append(int(g.mean()))

    if not centers:
        return np.array([], dtype=int)

    centers_arr = np.array(sorted(centers), dtype=int)

    # Si el número de filas detectadas ≠ expected_rows, remuestreamos linealmente
    if centers_arr.size != expected_rows:
        centers_arr = np.linspace(
            centers_arr[0],
            centers_arr[-1],
            expected_rows,
        ).astype(int)

    return centers_arr


def _calcular_row_boundaries(
    banda_gray: np.ndarray,
    x_ranges: Sequence[tuple[int, int]],
    answers_per_column: int,
    config: OMRConfig,
) -> np.ndarray:
    """
    Calcula los límites de filas (row_boundaries) a partir de los centros detectados.
    Devuelve un array de tamaño answers_per_column + 1 en coordenadas de banda_gray.
    """

    h_band, _ = banda_gray.shape[:2]
    if h_band <= 0 or answers_per_column <= 0:
        return np.linspace(0, h_band, answers_per_column + 1, dtype=float)

    if not x_ranges:
        return np.linspace(0, h_band, answers_per_column + 1, dtype=float)

    # Usamos la columna central como referencia
    col_idx = len(x_ranges) // 2
    col_idx = max(0, min(col_idx, len(x_ranges) - 1))
    ref_range = x_ranges[col_idx]

    centers = _detectar_centros_filas(
        banda_gray,
        ref_range,
        answers_per_column,
        config,
    )

    if centers.size != answers_per_column:
        # Fallback: si algo falla, usamos reparto lineal anterior
        return np.linspace(0, h_band, answers_per_column + 1, dtype=float)

    # Distancia típica entre filas
    diffs = np.diff(centers)
    step = float(np.median(diffs)) if diffs.size > 0 else h_band / answers_per_column

    boundaries = np.zeros(answers_per_column + 1, dtype=float)

    # Primera frontera: un poco por encima del primer centro
    boundaries[0] = max(0.0, centers[0] - step / 2.0)

    # Fronteras intermedias = puntos medios entre centros
    for i in range(1, answers_per_column):
        boundaries[i] = (centers[i - 1] + centers[i]) / 2.0

    # Última frontera: un poco por debajo del último centro
    boundaries[-1] = min(float(h_band), centers[-1] + step / 2.0)

    boundaries = np.clip(boundaries, 0.0, float(h_band))
    boundaries = np.sort(boundaries)

    return boundaries


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
        # las compactamos en columnas lógicas (para tener un bounding box por columna)
        answer_columns = _compactar_columnas_respuestas(
            answer_anchors_raw,
            group_size=len(config.answer_labels),
        )

        # Y calculamos, a partir de las mismas barras, los límites X de A–E
        option_boundaries = _calcular_boundaries_opciones(
            answer_anchors_raw,
            group_size=len(config.answer_labels),
        )

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        dni = _leer_dni(gray, dni_anchors, config, index)
        respuestas = _leer_respuestas(
            gray,
            answer_columns,
            option_boundaries,
            config,
            index,
        )


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


def _calcular_boundaries_opciones(
    raw_anchors: Sequence[tuple[int, int, int, int]],
    group_size: int,
) -> List[List[int]]:
    """
    A partir de las barras inferiores de respuestas (raw_anchors) calcula,
    para cada columna de preguntas, los límites X de las opciones A–E.

    Devuelve una lista de columnas; cada columna es una lista de 6 enteros:
    boundaries[0..5] tales que:
        A -> [b0, b1)
        B -> [b1, b2)
        ...
        E -> [b4, b5)
    """

    if not raw_anchors or group_size <= 0:
        return []

    anchors = list(raw_anchors)
    if len(anchors) < group_size:
        return []

    # Si el número total no es múltiplo de 5, devolvemos lista vacía
    if len(anchors) % group_size != 0:
        return []

    columnas_boundaries: List[List[int]] = []

    # Se asume el mismo agrupamiento de 5 en 5 que en _compactar_columnas_respuestas
    for i in range(0, len(anchors), group_size):
        group = anchors[i : i + group_size]
        if len(group) < group_size:
            continue

        # Ordenamos dentro del grupo de izquierda a derecha
        group_sorted = sorted(group, key=lambda r: r[0])

        # Centros X de cada barra (A–E)
        centers = [a[0] + a[2] // 2 for a in group_sorted]
        centers_arr = np.array(centers, dtype=float)

        if centers_arr.size < 2:
            continue

        # Distancia típica entre opciones
        diffs = np.diff(centers_arr)
        step = float(np.median(diffs))

        # Construimos boundaries como puntos medios entre centros
        boundaries: List[int] = []

        # Primer límite algo antes del centro de A
        left0 = int(round(centers_arr[0] - step / 2.0))
        boundaries.append(left0)

        # Límites intermedios = punto medio entre centros consecutivos
        for c1, c2 in zip(centers_arr[:-1], centers_arr[1:]):
            boundaries.append(int(round((c1 + c2) / 2.0)))

        # Último límite algo después del centro de E
        right_last = int(round(centers_arr[-1] + step / 2.0))
        boundaries.append(right_last)

        columnas_boundaries.append(boundaries)

    return columnas_boundaries


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


def _calcular_boundaries_opciones(
    raw_anchors: Sequence[tuple[int, int, int, int]],
    group_size: int,
) -> List[List[int]]:
    """
    A partir de las barras inferiores de respuestas (raw_anchors) calcula,
    para cada columna de preguntas, los límites X de las opciones A–E.

    Devuelve una lista de columnas; cada columna es una lista de 6 enteros:
    boundaries[0..5] tales que:
        A -> [b0, b1)
        B -> [b1, b2)
        ...
        E -> [b4, b5)
    """

    if not raw_anchors or group_size <= 0:
        return []

    anchors = list(raw_anchors)
    if len(anchors) < group_size:
        return []

    # Si el número total no es múltiplo de 5, devolvemos lista vacía
    if len(anchors) % group_size != 0:
        return []

    columnas_boundaries: List[List[int]] = []

    # Se asume el mismo agrupamiento de 5 en 5 que en _compactar_columnas_respuestas
    for i in range(0, len(anchors), group_size):
        group = anchors[i : i + group_size]
        if len(group) < group_size:
            continue

        # Ordenamos dentro del grupo de izquierda a derecha
        group_sorted = sorted(group, key=lambda r: r[0])

        # Centros X de cada barra (A–E)
        centers = [a[0] + a[2] // 2 for a in group_sorted]
        centers_arr = np.array(centers, dtype=float)

        if centers_arr.size < 2:
            continue

        # Distancia típica entre opciones
        diffs = np.diff(centers_arr)
        step = float(np.median(diffs))

        # Construimos boundaries como puntos medios entre centros
        boundaries: List[int] = []

        # Primer límite algo antes del centro de A
        left0 = int(round(centers_arr[0] - step / 2.0))
        boundaries.append(left0)

        # Límites intermedios = punto medio entre centros consecutivos
        for c1, c2 in zip(centers_arr[:-1], centers_arr[1:]):
            boundaries.append(int(round((c1 + c2) / 2.0)))

        # Último límite algo después del centro de E
        right_last = int(round(centers_arr[-1] + step / 2.0))
        boundaries.append(right_last)

        columnas_boundaries.append(boundaries)

    return columnas_boundaries


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
    option_boundaries_per_column: Sequence[Sequence[int]] | None,
    config: OMRConfig,
    pagina: int,
) -> List[Respuesta]:

    if not columnas:
        return []

    h, w = gray.shape

    # Banda aproximada de respuestas
    y0 = int(h * config.answer_vertical_band[0])
    y1 = int(h * config.answer_vertical_band[1])
    if y1 - y0 < 1:
        return []

    banda = gray[y0:y1, :]
    band_height = banda.shape[0]

    num_columnas = len(columnas)
    answers_per_column = int(np.ceil(config.questions / num_columnas))
    column_width = _estimacion_ancho_columnas(columnas)
    resultados: List[Respuesta] = []

    # Rango X de cada columna lógica para perfilar filas
    x_ranges: List[tuple[int, int]] = []
    for idx, col in enumerate(columnas):
        x_center = col[0] + col[2] // 2

        # Si tenemos boundaries específicos A–E para esta columna,
        # usamos el mínimo y máximo como rango de columna.
        if (
            option_boundaries_per_column
            and idx < len(option_boundaries_per_column)
            and len(option_boundaries_per_column[idx]) >= 2
        ):
            bounds = option_boundaries_per_column[idx]
            x0 = max(int(bounds[0]), 0)
            x1 = min(int(bounds[-1]), w)
        else:
            # Fallback antiguo
            x0 = max(x_center - column_width // 2, 0)
            x1 = min(x_center + column_width // 2, w)

        if x1 <= x0:
            x1 = min(x0 + 1, w)

        x_ranges.append((x0, x1))

    # Afinar banda vertical
    adj_top, adj_bottom = _ajustar_banda_vertical(
        banda,
        x_ranges,
        min_height=max(int(band_height * 0.85), 400 // max(num_columnas, 1)),
        activation_ratio=0.10,
    )
    banda = banda[adj_top:adj_bottom, :]
    band_height = banda.shape[0]

    # boundaries de filas basados en perfil de burbujas
    row_boundaries = _calcular_row_boundaries(
        banda,
        x_ranges,
        answers_per_column,
        config,
    )

    for question_index in range(config.questions):
        # columna lógica
        column_index = min(
            question_index // answers_per_column,
            num_columnas - 1,
        )

        x0_col, x1_col = x_ranges[column_index]

        # fila dentro de la columna
        row_idx = question_index % answers_per_column
        row_top = float(row_boundaries[row_idx])
        row_bottom = float(row_boundaries[row_idx + 1])
        row_height = row_bottom - row_top

        center = (row_top + row_bottom) / 2.0
        half_height = row_height * 0.50

        local_top = int(max(center - half_height, 0))
        local_bottom = int(min(center + half_height, band_height))
        if local_bottom <= local_top:
            local_bottom = min(local_top + 1, band_height)

        sub = banda[local_top:local_bottom, x0_col:x1_col]

        # Boundaries específicos A–E para esta columna, en coordenadas locales
        local_boundaries: list[int] | None = None
        if (
            option_boundaries_per_column
            and column_index < len(option_boundaries_per_column)
            and len(option_boundaries_per_column[column_index]) == len(config.answer_labels) + 1
        ):
            global_bounds = option_boundaries_per_column[column_index]
            local_boundaries = [
                max(0, min(int(b) - x0_col, x1_col - x0_col))
                for b in global_bounds
            ]

        alternativa, estado, intensidad = _clasificar_alternativa(
            sub,
            config.answer_labels,
            config,
            x_boundaries=local_boundaries,
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


def _clasificar_alternativa(
    answer_img: np.ndarray,
    labels: Sequence[str],
    config: OMRConfig,
    x_boundaries: Sequence[int] | None = None,
) -> tuple[str, str, float]:
    """Determina la opción con más tinta y aplica reglas de negocio.

    Si x_boundaries no es None, se asume que contiene len(labels)+1 límites X
    (en coordenadas locales de answer_img) que definen exactamente las
    regiones A–E. En ese caso NO se aplica left_margin ni reparto uniforme.
    """

    if answer_img.size == 0:
        return ("-", "SIN RESPUESTA", 0.0)

    gray = answer_img
    h_sub, w_sub = gray.shape
    if w_sub <= 0 or h_sub <= 0:
        return ("-", "SIN RESPUESTA", 0.0)

    # ---------- Recorte horizontal ----------
    if x_boundaries is not None and len(x_boundaries) == len(labels) + 1:
        # No aplicamos margen izquierdo: los boundaries ya vienen “limpios”.
        right_img = gray
        left_offset = 0
        boundaries = np.array(
            [max(0, min(int(b), w_sub)) for b in x_boundaries],
            dtype=int,
        )
    else:
        # Comportamiento anterior (pero podemos bajar el margen)
        left_margin = int(w_sub * config.answer_left_margin_ratio)
        if left_margin >= w_sub:
            left_margin = 0
        right_img = gray[:, left_margin:]
        left_offset = left_margin
        h_use, w_use = right_img.shape
        if w_use <= 0:
            right_img = gray
            h_use, w_use = right_img.shape
            left_offset = 0
        boundaries = np.linspace(0, w_use, len(labels) + 1, dtype=int)

    h_use, w_use = right_img.shape

    # ---------- Medida de tinta ----------
    inv = cv2.bitwise_not(right_img).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(right_img, (5, 5), 0)
    _, bin_inv = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    scores: List[float] = []
    for idx in range(len(labels)):
        x0 = boundaries[idx]
        x1 = boundaries[idx + 1]
        if x1 <= x0:
            x1 = min(x0 + 1, w_use)

        cell_gray = inv[:, x0:x1]
        cell_bin = bin_inv[:, x0:x1]

        score_gray = float(cell_gray.mean()) if cell_gray.size else 0.0
        score_bin = float(cell_bin.mean()) / 255.0 if cell_bin.size else 0.0

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
        return ("-", "SIN RESPUESTA", best_score)

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
    if y1 - y0 < 1 or not answer_columns:
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

    # Afinar banda vertical igual que en _leer_respuestas
    adj_top, adj_bottom = _ajustar_banda_vertical(
        banda_gray,
        x_ranges,
        min_height=max(int(band_height * 0.85), 400 // max(len(answer_columns), 1)),
        activation_ratio=0.10,
    )

    band_color = band_color[adj_top:adj_bottom, :]
    banda_gray = banda_gray[adj_top:adj_bottom, :]
    band_height = band_color.shape[0]

    answers_per_column = int(np.ceil(config.questions / len(answer_columns)))

    # *** boundaries calibrados por perfil ***
    row_boundaries = _calcular_row_boundaries(
        banda_gray,
        x_ranges,
        answers_per_column,
        config,
    ).astype(int)

    # Dibujamos columnas y filas
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

        for rb in row_boundaries:
            y_rb = int(rb)
            cv2.line(band_color, (x0, y_rb), (x1, y_rb), (255, 0, 0), 1)

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

    col_index = min(
        question_index // answers_per_column,
        len(answer_columns) - 1,
    )
    x0, x1 = x_ranges[col_index]

    row_idx = question_index % answers_per_column
    local_top = int(np.floor(row_boundaries[row_idx]))
    local_bottom = int(np.ceil(row_boundaries[row_idx + 1]))
    if local_bottom <= local_top:
        local_bottom = min(local_top + 1, band_height)

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
