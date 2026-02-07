"""Costruzione del contesto dal metadata AI2-THOR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except Exception:
    np = None

from Architettura.yolo.labels import normalize_target

_LAST_SEEN: Dict[str, Any] | None = None
_TARGET_MEMORY: Dict[str, Any] | None = None


def _read_yolo_json(yolo_json_path: str | None) -> List[Dict[str, Any]]:
    # Legge il file JSON di YOLO (se presente) e ritorna una lista di detection.
    # Serve per calcolare last_seen e posizione del target dal bounding box.
    # Se il file manca o e' invalido, ritorna lista vuota.
    if not yolo_json_path:
        return []
    path = Path(yolo_json_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _side_from_center(x_center: float, width: int) -> str:
    # Classifica la posizione orizzontale del target (left/center/right).
    # Usa una fascia centrale (40%-60%) per "center".
    # Viene usata per costruire last_seen nel contesto VLM.
    if width <= 0:
        return "unknown"
    left_bound = width * 0.4
    right_bound = width * 0.6
    if x_center < left_bound:
        return "left"
    if x_center > right_bound:
        return "right"
    return "center"


def _grid_position(
    x_center: float,
    y_center: float,
    width: int,
    height: int,
) -> str:
    # Classifica la posizione su una griglia 3x2 (top/bottom + left/center/right).
    # Serve per salvare una memoria grossolana della posizione del target.
    # Restituisce stringhe tipo "top-left", "bottom-center", ecc.
    if width <= 0 or height <= 0:
        return "center"
    x_ratio = x_center / float(width)
    y_ratio = y_center / float(height)
    if x_ratio < 1 / 3:
        x_pos = "left"
    elif x_ratio < 2 / 3:
        x_pos = "center"
    else:
        x_pos = "right"
    if y_ratio < 0.5:
        y_pos = "top"
    else:
        y_pos = "bottom"
    if x_pos == "center" and y_pos == "center":
        return "center"
    return f"{y_pos}-{x_pos}"


def _find_target_bbox(
    detections: List[Dict[str, Any]],
    target_norm: str,
) -> Tuple[list[float] | None, float | None]:
    # Cerca tra le detection YOLO il bbox del target con confidenza piu' alta.
    # Ritorna (bbox, conf) oppure (None, None) se non trovato.
    # Serve per calcolare last_seen e la memoria posizione.
    best_bbox = None
    best_conf = None
    for det in detections:
        label = str(det.get("class_name", "")).lower()
        if label != target_norm:
            continue
        conf = float(det.get("confidence", 0.0))
        bbox = det.get("bbox_xyxy") or []
        if len(bbox) != 4:
            continue
        if best_conf is None or conf > best_conf:
            best_conf = conf
            best_bbox = bbox
    return best_bbox, best_conf


def _target_distance_from_metadata(
    metadata: Dict[str, Any],
    target_norm: str,
) -> tuple[float | None, str | None]:
    # Cerca la distanza del target direttamente nei metadata di AI2-THOR.
    # Se ci sono piu' istanze, prende quella piu' vicina.
    # Ritorna (distanza, objectId) o (None, None) se non disponibile.
    objects = metadata.get("objects") or []
    best = None
    best_id = None
    for obj in objects:
        obj_type = str(obj.get("objectType", "")).lower()
        if not obj_type or obj_type != target_norm:
            continue
        try:
            dist = float(obj.get("distance"))
        except Exception:
            continue
        if best is None or dist < best:
            best = dist
            best_id = obj.get("objectId")
    return best, best_id


def _update_last_seen(
    *,
    target_label: str | None,
    yolo_json_path: str | None,
    frame_width: int | None,
    frame_idx: int | None,
) -> Dict[str, Any] | None:
    # Aggiorna la memoria "last_seen" usando il bbox YOLO del target.
    # Calcola il lato (left/center/right) con _side_from_center.
    # Ritorna un dict con label, side, confidence e frame_index.
    global _LAST_SEEN
    target_norm = normalize_target(target_label or "")
    if not target_norm:
        return _LAST_SEEN
    if frame_width is None:
        return _LAST_SEEN
    detections = _read_yolo_json(yolo_json_path)
    best, best_conf = _find_target_bbox(detections, target_norm)
    if best is None or best_conf is None:
        return _LAST_SEEN
    x1, _, x2, _ = best
    x_center = (float(x1) + float(x2)) * 0.5
    _LAST_SEEN = {
        "label": target_norm,
        "side": _side_from_center(x_center, int(frame_width)),
        "confidence": float(best_conf),
        "frame_index": frame_idx,
    }
    return _LAST_SEEN


def _depth_stats(depth_frame: Any) -> Dict[str, float]:
    # Calcola statistiche base sulla depth image (min/max/mean).
    # Utile per debug o per logiche future.
    # Se numpy manca o depth_frame e' None, ritorna dict vuoto.
    if depth_frame is None or np is None:
        return {}
    depth = np.asarray(depth_frame, dtype=float)
    return {
        "min_depth_m": float(np.min(depth)),
        "max_depth_m": float(np.max(depth)),
        "mean_depth_m": float(np.mean(depth)),
    }


def build_context(
    event: Any,
    scan_count: int | None = None,
    target_label: str | None = None,
    yolo_json_path: str | None = None,
    frame_idx: int | None = None,
) -> Dict[str, Any]:
    """Trasforma i dati grezzi in un report leggibile per l'IA."""
    # Costruisce il contesto da metadata AI2-THOR + YOLO per la VLM.
    # Include last_seen, distanza target, collisioni, target_position e telemetria.
    # E' la base testuale che alimenta il prompt della VLM.
    metadata = getattr(event, "metadata", {}) or {}
    objects = metadata.get("objects") or []
    frame_width = None
    frame_height = None
    try:
        frame = getattr(event, "frame", None)
        frame_width = int(frame.shape[1])
        frame_height = int(frame.shape[0])
    except Exception:
        frame_width = None
        frame_height = None
    last_seen = _update_last_seen(
        target_label=target_label,
        yolo_json_path=yolo_json_path,
        frame_width=frame_width,
        frame_idx=frame_idx,
    )
    target_object = (target_label or "").strip()
    target_norm = normalize_target(target_label or "")
    detections = _read_yolo_json(yolo_json_path)
    bbox, _ = _find_target_bbox(detections, target_norm) if target_norm else (None, None)
    # target_distance: distanza numerica (metri) dal target piu' vicino.
    # target_distance_source: objectId dell'istanza che fornisce quella distanza (debug).
    target_distance = None
    target_distance_source = None
    # Se abbiamo un target valido, leggiamo la distanza direttamente dai metadata.
    if target_norm:
        target_distance, target_distance_source = _target_distance_from_metadata(
            metadata, target_norm
        )

    # Memoria di posizione: quando abbiamo un bbox, salviamo una posizione grossolana
    # (top/bottom + left/center/right) per ricordare dove si trovava il target.
    global _TARGET_MEMORY
    if bbox is not None and frame_width and frame_height:
        x1, y1, x2, y2 = bbox
        x_center = (float(x1) + float(x2)) * 0.5
        y_center = (float(y1) + float(y2)) * 0.5
        _TARGET_MEMORY = {
            "position": _grid_position(
                x_center,
                y_center,
                int(frame_width),
                int(frame_height),
            ),
            "frame_index": frame_idx,
        }
    # Posizione "memorizzata" del target (puo' essere None se mai visto).
    target_known_position = (
        _TARGET_MEMORY.get("position") if _TARGET_MEMORY is not None else None
    )
    # last_seen: memoria dell'ultima volta che il target e' apparso (left/center/right).
    # Se frame_index differisce, viene marcato come "stale".
    last_seen_context = None
    if last_seen is not None:
        last_seen_context = dict(last_seen)
        if frame_idx is not None:
            last_seen_context["stale"] = last_seen.get("frame_index") != frame_idx

    # target_position: posizione 3D del target (se disponibile nei metadata AI2-THOR).
    target_position = None
    if target_norm:
        for obj in objects:
            obj_type = str(obj.get("objectType", "")).lower()
            if obj_type != target_norm:
                continue
            pos = obj.get("position")
            if isinstance(pos, dict):
                target_position = {
                    "objectId": obj.get("objectId"),
                    "objectType": obj.get("objectType"),
                    "position": {
                        "x": pos.get("x"),
                        "y": pos.get("y"),
                        "z": pos.get("z"),
                    },
                }
                break

    # Blocco dati grezzo: raccoglie stato dell'azione, collisioni e info target.
    context_data = {
        "last_action": metadata.get("lastAction"),
        "last_action_success": metadata.get("lastActionSuccess", True),
        "last_action_error": metadata.get("errorMessage") or "",
        "collided": metadata.get("collided"),
        "collidedObjects": metadata.get("collidedObjects") or [],
        "scan_count": scan_count,
        "last_seen": last_seen_context,
        "target_object": target_object,
        "target_distance": target_distance,
        "target_distance_source": target_distance_source,
        "target_position": target_position,
    }

    # Contesto finale passato alla VLM: solo campi strutturati (senza telemetria testuale).
    return {
        "last_action": context_data.get("last_action"),
        "last_action_success": context_data.get("last_action_success"),
        "last_action_error": context_data.get("last_action_error"),
        "collided": context_data.get("collided"),
        "collidedObjects": context_data.get("collidedObjects"),
        "scan_count": scan_count,
        "last_seen": context_data.get("last_seen"),
        "target_object": context_data.get("target_object"),
        "target_distance": context_data.get("target_distance"),
    }
