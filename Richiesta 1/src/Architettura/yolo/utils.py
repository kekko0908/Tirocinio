"""Utility YOLO (engine17 + salvataggio frame con bbox)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import re

import numpy as np
try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

from Architettura.paths import ARTIFACTS_ROOT, rel_to_project
from Architettura.yolo.centroide import centroid_from_bbox
from pathlib import Path as _Path
import sys as _sys
import importlib

_REPO_ROOT = _Path(__file__).resolve().parents[4]
_TIROCINIO_SRC = _REPO_ROOT / "Tirocinio" / "src"
if _TIROCINIO_SRC.exists() and str(_TIROCINIO_SRC) not in _sys.path:
    _sys.path.insert(0, str(_TIROCINIO_SRC))

YOLO_OUT_DIR = ARTIFACTS_ROOT / "yolo_outputs" / "yolo"
YOLO_JSON_DIR = ARTIFACTS_ROOT / "yolo_outputs" / "json"
_YBB_MODULE = None


def _get_ybb_module():
    """Import lazy: evita di caricare torch/ultralytics prima di llama_cpp."""
    global _YBB_MODULE
    if _YBB_MODULE is None:
        _YBB_MODULE = importlib.import_module("YOLO_vs_VLM.yolo_bounding_box")
    return _YBB_MODULE


def _parse_conf_from_debug(debug_info: str | None) -> float | None:
    if not debug_info:
        return None
    match = re.search(r"conf=([0-9.]+)", debug_info)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _compute_target_center_x(det: Dict[str, Any], image_width: int) -> float | None:
    if image_width <= 0:
        return None
    centroid = det.get("centroid")
    cx = None
    if isinstance(centroid, dict):
        cx = centroid.get("x")
    elif isinstance(centroid, (list, tuple)) and len(centroid) == 2:
        cx = centroid[0]
    if cx is None:
        bbox = det.get("bbox_xyxy") or []
        if len(bbox) == 4:
            cx, _ = centroid_from_bbox(bbox)
    if cx is None:
        return None
    try:
        return float(cx) / float(image_width)
    except Exception:
        return None


def _draw_boxes(base: "Image.Image", det: Dict[str, Any]) -> "Image.Image":
    if ImageDraw is None:
        raise RuntimeError("Serve PIL.ImageDraw per disegnare i box.")
    draw = ImageDraw.Draw(base)
    x1, y1, x2, y2 = det["bbox_xyxy"]
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    label = f"{det['class_name']} {det['confidence']:.2f}"
    draw.text((int(x1), int(max(0, y1 - 12))), label, fill=(0, 255, 0))
    cx, cy = centroid_from_bbox([x1, y1, x2, y2])
    r = 4
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=(255, 0, 0),
        fill=(255, 0, 0),
        width=2,
    )
    return base


def save_yolo_png(
    image: Any,
    frame_idx: int,
    target_label: str | None = None,
    model_path: str | Path | None = None,
    conf: float = 0.20,
    output_path: Path | None = None,
    image_is_bgr: bool = False,
) -> Dict[str, Any]:
    if Image is None:
        raise RuntimeError("Serve PIL per salvare il PNG YOLO.")

    YOLO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = YOLO_OUT_DIR / f"yolo_{frame_idx:05d}.png"

    min_conf = float(conf)
    target_center_x = None

    if Image is not None and isinstance(image, Image.Image):
        image_rgb = np.asarray(image)
    else:
        image_rgb = image
    try:
        if isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
            color_hint = "BGR" if image_is_bgr else "RGB"
            print(
                f"[YOLO] input_shape={tuple(image_rgb.shape)} dtype={image_rgb.dtype} "
                f"input_color={color_hint}"
            )
    except Exception:
        pass

    if not target_label:
        save_img = image_rgb
        if image_is_bgr and isinstance(image_rgb, np.ndarray) and image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
            try:
                save_img = image_rgb[:, :, ::-1]
            except Exception:
                save_img = image_rgb
        Image.fromarray(save_img).convert("RGB").save(output_path.as_posix())
        return {
            "yolo": rel_to_project(output_path),
            "match": False,
            "track_id": None,
            "min_conf": None,
            "target_center_x": target_center_x,
            "detections": [],
        }

    ybb = _get_ybb_module()
    bbox, debug_info = ybb.save_yolo_png(
        image_rgb,
        frame_idx=frame_idx,
        target_name=str(target_label),
        model_path=str(model_path or "yolo11x-seg.pt"),
        output_path=output_path,
        conf=float(conf),
        image_is_bgr=bool(image_is_bgr),
    )
    if bbox is None:
        if debug_info:
            print(f"[YOLO DEBUG] no match | {debug_info}", flush=True)
        return {
            "yolo": rel_to_project(output_path),
            "match": False,
            "track_id": None,
            "min_conf": min_conf,
            "target_center_x": target_center_x,
            "detections": [],
        }

    det_conf = _parse_conf_from_debug(debug_info) or float(conf)
    x1, y1, x2, y2 = bbox
    cx, cy = centroid_from_bbox(bbox)
    selected = {
        "class_id": -1,
        "class_name": str(target_label),
        "confidence": float(det_conf),
        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "centroid": {"x": float(cx), "y": float(cy)},
        "track_id": None,
    }
    target_center_x = _compute_target_center_x(selected, image_rgb.shape[1])

    return {
        "yolo": rel_to_project(output_path),
        "match": True,
        "track_id": None,
        "min_conf": min_conf,
        "target_center_x": target_center_x,
        "detections": [selected],
    }


def save_artifacts_yolo(
    image: Any,
    frame_idx: int,
    target_label: str | None = None,
    oracle_data: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    """Salva i risultati YOLO e ritorna i percorsi."""
    del image
    del target_label
    del oracle_data
    YOLO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_JSON_DIR.mkdir(parents=True, exist_ok=True)
    yolo_frame_path = YOLO_OUT_DIR / f"yolo_{frame_idx:05d}.png"
    yolo_json_path = YOLO_JSON_DIR / f"yolo_{frame_idx:05d}.json"
    if not yolo_json_path.exists():
        yolo_json_path.write_text(json.dumps([], ensure_ascii=True), encoding="utf-8")
    return {
        "yolo_frame": rel_to_project(yolo_frame_path),
        "yolo_json": rel_to_project(yolo_json_path),
        "yolo_seg": None,
    }
