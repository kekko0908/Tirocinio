"""Calcolo stato VLM a livello di sistema."""
# Nota: questo modulo non e' attualmente operativo nel flusso principale.
# E' stato preparato per una futura logica a stati della VLM.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from Architettura.yolo.labels import normalize_target


def _read_yolo_labels(yolo_json_path: str | None) -> list[str]:
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
    labels = []
    for item in data or []:
        name = item.get("class_name")
        if name:
            labels.append(str(name).lower())
    return labels


def is_target_visible(target: str, yolo_json_path: str | None) -> bool:
    target_norm = normalize_target(target)
    if not target_norm:
        return False
    labels = _read_yolo_labels(yolo_json_path)
    return target_norm in labels


def compute_vlm_state(
    *,
    scan_count: int,
    scan_complete: bool,
    current_target: str,
    target_queue: list[str],
    search_near: bool,
    yolo_json_path: str | None,
    last_action_success: bool,
) -> Tuple[str, bool]:
    """Ritorna (state, target_visible)."""
    target_visible = is_target_visible(current_target, yolo_json_path)

    if not last_action_success:
        return "MODE_RECOVERY", target_visible

    if not scan_complete and scan_count < 4:
        return "MODE_INITIALIZATION", target_visible
    if target_visible:
        return "MODE_APPROACH", target_visible
    if search_near and target_queue:
        return "MODE_SEARCH_NEAR", target_visible
    return "MODE_NAVIGATE", target_visible
