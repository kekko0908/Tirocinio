"""Funzioni YOLO per segmentazione."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

try:
    import numpy as np
except Exception:
    np = None

from Architettura.yolo.bounding_box import build_bbox_payload
from Architettura.yolo.centroide import centroid_from_bbox, centroid_from_mask
from Architettura.yolo.labels import normalize_target
from Architettura.yolo.utils import get_model

PRED_CONF_MIN = 0.05


def _draw_detections(
    base: "Image.Image",
    detections: List[Dict[str, Any]],
    include_labels: bool = False,
) -> "Image.Image":
    if ImageDraw is None:
        raise RuntimeError("Serve PIL.ImageDraw per disegnare le maschere.")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for det in detections:
        mask = det.get("mask")
        if mask is not None:
            if np is None:
                raise RuntimeError("Serve numpy per convertire la maschera.")
            mask_img = Image.fromarray((mask > 0).astype("uint8") * 160)
            green = Image.new("RGBA", base.size, (0, 255, 0, 140))
            overlay.paste(green, (0, 0), mask_img)
    draw = ImageDraw.Draw(overlay)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255, 255), width=4)
        if include_labels:
            label = f"{det['class_name']} {det['confidence']:.2f}"
            label_pos = (int(x1), int(max(0, y1 - 12)))
            draw.text(label_pos, label, fill=(0, 0, 255, 255))
        centroid = det.get("centroid")
        if isinstance(centroid, dict):
            cx = centroid.get("x")
            cy = centroid.get("y")
        else:
            if centroid is None:
                centroid = _compute_centroid(det)
            if centroid is not None:
                cx, cy = centroid
            else:
                cx = cy = None
        if cx is not None and cy is not None:
            r = 4
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=(255, 0, 0, 255),
                outline=(255, 0, 0, 255),
            )
    return Image.alpha_composite(base, overlay).convert("RGB")


def _compute_centroid(det: Dict[str, Any]) -> tuple[int, int] | None:
    xyxy = det.get("bbox_xyxy") or []
    if len(xyxy) != 4:
        return None
    mask = det.get("mask")
    if mask is not None:
        try:
            centroid = centroid_from_mask(mask)
            if centroid is not None:
                return int(centroid[0]), int(centroid[1])
        except Exception:
            pass
    centroid = centroid_from_bbox(xyxy)
    return int(centroid[0]), int(centroid[1])


def _oracle_to_detection(
    oracle_data: Dict[str, Any],
    target_label: str | None,
) -> Dict[str, Any]:
    # Converte i dati oracle (ground truth) in una detection "finta" stile YOLO.
    # Serve per riutilizzare lo stesso flusso di disegno/overlay quando YOLO non basta.
    # Include bbox, maschera e centroide se disponibili.
    name = target_label or oracle_data.get("object_type") or "target"
    det = {
        "class_id": -1,
        "class_name": str(name),
        "confidence": 1.0,
        "bbox_xyxy": oracle_data.get("bbox_xyxy") or [0.0, 0.0, 0.0, 0.0],
    }
    mask = oracle_data.get("mask")
    if mask is not None:
        det["mask"] = mask
    centroid = oracle_data.get("centroid")
    if centroid is not None:
        det["centroid"] = centroid
    return det


def _select_target_detection(
    detections: List[Dict[str, Any]],
    target_label: str | None,
) -> Dict[str, Any] | None:
    if not detections:
        return None
    target_norm = normalize_target(target_label or "")
    if not target_norm:
        return None
    matches = [
        det
        for det in detections
        if normalize_target(det.get("class_name", "")) == target_norm
    ]
    if not matches:
        return None
    return max(matches, key=lambda d: float(d.get("confidence", 0.0)))


def detect_segments(
    image: Any,
    conf: float | None = None,
) -> List[Dict[str, Any]]:
    """YOLO segmentazione: restituisce box + maschera per ogni detection."""
    # Logica della segmentazione:
    # 1) esegue il modello YOLO sull'immagine (predict),
    # 2) per ogni detection legge classe, confidenza e bounding box,
    # 3) se il modello produce maschere, associa la maschera al box,
    # 4) costruisce un dizionario per ogni oggetto (bbox + mask + label),
    # 5) ritorna la lista completa di detections.
    model = get_model()
    if conf is None:
        results = model(image, verbose=False)
    else:
        results = model.predict(source=image, conf=float(conf), verbose=False)
    if not results:
        return []
    result = results[0]
    if result.boxes is None:
        return []
    masks = result.masks
    detections: List[Dict[str, Any]] = []
    mask_data = masks.data if masks is not None else None
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item()) if box.cls is not None else -1
        conf = float(box.conf.item()) if box.conf is not None else 0.0
        xyxy = box.xyxy[0].tolist()
        name = str(result.names.get(cls_id, str(cls_id)))
        mask = None
        if mask_data is not None and i < len(mask_data):
            mask = mask_data[i].cpu().numpy()
        detections.append(build_bbox_payload(cls_id, name, conf, xyxy, mask))
    return detections


def save_segmented_png(
    image: Any,
    output_path: str | Path,
    target_label: str | None = None,
    oracle_data: Dict[str, Any] | None = None,
) -> Optional[Path]:
    """Esegue YOLO e salva un PNG con box + maschere sovrapposte."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if Image is None:
        raise RuntimeError("Serve PIL per salvare il PNG segmentato.")

    if oracle_data is not None:
        base = Image.fromarray(image).convert("RGBA")
        det = _oracle_to_detection(oracle_data, target_label)
        combined = _draw_detections(base, [det])
        combined.save(output_path.as_posix())
        return output_path

    detections = detect_segments(image, conf=float(PRED_CONF_MIN))
    if not detections:
        base = Image.fromarray(image).convert("RGB")
        base.save(output_path.as_posix())
        return output_path
    selected = _select_target_detection(detections, target_label)
    if selected is None:
        base = Image.fromarray(image).convert("RGB")
        base.save(output_path.as_posix())
        return output_path
    detections = [selected]

    base = Image.fromarray(image).convert("RGBA")
    if detections:
        combined = _draw_detections(base, detections)
    else:
        combined = base.convert("RGB")
    combined.save(output_path.as_posix())
    return output_path


def save_combined_frame(
    image: Any,
    output_path: str | Path,
    json_path: str | Path,
    target_label: str | None = None,
    oracle_data: Dict[str, Any] | None = None,
) -> Optional[Path]:
    """Salva un frame con box blu, maschera verde e centroidi rossi."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if Image is None:
        raise RuntimeError("Serve PIL per disegnare il frame combinato.")

    if oracle_data is not None:
        base = Image.fromarray(image).convert("RGBA")
        det = _oracle_to_detection(oracle_data, target_label)
        centroid = det.get("centroid")
        if centroid is None:
            centroid = _compute_centroid(det)
            if centroid is not None:
                det["centroid"] = {"x": float(centroid[0]), "y": float(centroid[1])}
        combined = _draw_detections(base, [det], include_labels=True)
        combined.save(output_path.as_posix())
        payload = dict(det)
        payload.pop("mask", None)
        json_path.write_text(json.dumps([payload], indent=2), encoding="utf-8")
        return output_path

    detections = detect_segments(image, conf=float(PRED_CONF_MIN))
    selected = _select_target_detection(detections, target_label)

    base = Image.fromarray(image).convert("RGBA")
    if selected is None:
        combined = base.convert("RGB")
        combined.save(output_path.as_posix())
        json_path.write_text("[]", encoding="utf-8")
        return output_path

    centroid = _compute_centroid(selected)
    if centroid is not None:
        selected["centroid"] = {"x": float(centroid[0]), "y": float(centroid[1])}

    combined = _draw_detections(base, [selected], include_labels=True)
    combined.save(output_path.as_posix())

    payload = dict(selected)
    payload.pop("mask", None)
    json_path.write_text(json.dumps([payload], indent=2), encoding="utf-8")
    return output_path
