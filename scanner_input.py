"""Carga y preprocesado de imágenes escaneadas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2  # type: ignore
import numpy as np


@dataclass
class PreprocessSettings:
    blur_kernel: int = 3
    binarize: bool = False


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def preprocess_image(image: np.ndarray, settings: PreprocessSettings) -> Tuple[np.ndarray, np.ndarray]:
    gray = to_grayscale(image)
    if settings.blur_kernel > 1:
        k = settings.blur_kernel | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    processed = gray
    if settings.binarize:
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, processed


__all__ = ["PreprocessSettings", "load_image", "preprocess_image", "to_grayscale"]
