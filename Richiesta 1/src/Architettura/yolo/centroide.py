"""Utilita' per calcolare centroidi da box o maschere."""

from __future__ import annotations

from typing import Iterable, Tuple

try:
    import numpy as np
except Exception:
    np = None


def centroid_from_bbox(xyxy: Iterable[float]) -> Tuple[float, float]:
    """Calcola il centroide dal bounding box [x1, y1, x2, y2]."""
    # Logica (box):
    # 1) prendo i due angoli opposti (x1,y1) e (x2,y2),
    # 2) il centroide e' il punto medio: ((x1+x2)/2, (y1+y2)/2).
    # Questo funziona perche' il box e' un rettangolo allineato agli assi.
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def centroid_from_mask(mask) -> Tuple[float, float]:
    """Calcola il centroide da una maschera binaria."""
    # Logica (maschera):
    # 1) prendo tutti i pixel "attivi" (mask > 0),
    # 2) calcolo la media delle coordinate x e y di quei pixel,
    # 3) la media e' il centro di massa della maschera.
    # Se la maschera e' vuota, ritorna (0.0, 0.0).
    if np is None:
        raise RuntimeError("Serve numpy per calcolare il centroide da maschera.")
    mask = np.asarray(mask)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0.0, 0.0)
    return float(xs.mean()), float(ys.mean())
