"""Alineación de la hoja mediante marcas inferiores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2  # type: ignore
import numpy as np

from template import Point, TemplateGeometry


@dataclass
class AlignmentResult:
    matrix: np.ndarray
    detected_marks: List[Point]

    def transform_point(self, point: Point) -> Point:
        x, y = point
        px = self.matrix[0, 0] * x + self.matrix[0, 1] * y + self.matrix[0, 2]
        py = self.matrix[1, 0] * x + self.matrix[1, 1] * y + self.matrix[1, 2]
        return float(px), float(py)


def preprocess_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    return gray


@dataclass
class AlignmentSettings:
    search_band_ratio: float = 0.25
    min_area_ratio: float = 0.0005
    aspect_ratio_range: Tuple[float, float] = (2.5, 20.0)
    blur_kernel: int = 5


def detect_alignment_marks(gray: np.ndarray, settings: AlignmentSettings) -> List[Point]:
    h, w = gray.shape
    band_start = int(h * (1 - settings.search_band_ratio))
    band = gray[band_start:, :]

    blur = cv2.GaussianBlur(band, (settings.blur_kernel, settings.blur_kernel), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = settings.min_area_ratio * w * h
    aspect_min, aspect_max = settings.aspect_ratio_range

    candidates: List[Tuple[float, float]] = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        if area < min_area:
            continue
        aspect = rw / rh if rh else 0
        if not (aspect_min <= aspect <= aspect_max):
            continue
        bottom = y + rh
        if bottom < 0.5 * band.shape[0]:
            continue
        cx = x + rw / 2
        cy = y + rh / 2 + band_start
        candidates.append((cx, cy))

    candidates.sort(key=lambda p: p[0])
    return candidates[:3]


def compute_alignment(template: TemplateGeometry, marks: Sequence[Point]) -> AlignmentResult | None:
    if len(marks) < 3:
        return None

    src = np.array(template.alignment_reference_points(), dtype=np.float32)
    dst = np.array(marks[:3], dtype=np.float32)

    matrix, _ = cv2.estimateAffinePartial2D(src, dst)
    if matrix is None:
        return None

    return AlignmentResult(matrix=matrix, detected_marks=list(marks[:3]))


__all__ = [
    "AlignmentResult",
    "AlignmentSettings",
    "detect_alignment_marks",
    "compute_alignment",
    "preprocess_gray",
]
