"""Lectura individual de burbujas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2  # type: ignore
import numpy as np

from template import Point


def _crop_region(gray: np.ndarray, center: Point, size: Tuple[float, float]) -> np.ndarray:
    cx, cy = center
    w, h = size
    x0 = int(max(cx - w / 2, 0))
    y0 = int(max(cy - h / 2, 0))
    x1 = int(min(cx + w / 2, gray.shape[1]))
    y1 = int(min(cy + h / 2, gray.shape[0]))
    return gray[y0:y1, x0:x1]


@dataclass
class BubbleMetrics:
    ink_level: float
    marked: bool


def measure_bubble(gray: np.ndarray, center: Point, size: Tuple[float, float], threshold: float) -> BubbleMetrics:
    region = _crop_region(gray, center, size)
    if region.size == 0:
        return BubbleMetrics(ink_level=0.0, marked=False)

    # Normalizar a [0,1] invertido para que más tinta => valor mayor
    inverted = cv2.bitwise_not(region).astype(np.float32) / 255.0
    ink = float(inverted.mean())
    return BubbleMetrics(ink_level=ink, marked=ink >= threshold)


__all__ = ["BubbleMetrics", "measure_bubble"]
