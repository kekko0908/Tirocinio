"""Wrapper minimale per AI2-THOR con primitive base."""
# Nota su semantic vs instance:
# - semantic segmentation colora per classe (tutto il pavimento stesso colore),
# - instance segmentation separa ogni oggetto singolo (id diversi).
# Qui privilegiamo la semantic perche' evidenzia il pavimento/ostacoli in modo uniforme.

import json
import time
from typing import Iterable
import os

from pathlib import Path

from Architettura.paths import ARTIFACTS_ROOT, rel_to_project
VISION_ROOT = ARTIFACTS_ROOT / "vision_outputs"
FIRST_PERSON_ROOT = VISION_ROOT / "first_person"
RGB_DIR = FIRST_PERSON_ROOT / "rgb"
DEPTH_DIR = FIRST_PERSON_ROOT / "depth"
INSTANCE_DIR = FIRST_PERSON_ROOT / "instance"
SEMANTIC_DIR = FIRST_PERSON_ROOT / "semantic"
THIRD_PERSON_DIR = VISION_ROOT / "third_person"

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

from ai2thor.controller import Controller
from Architettura.yolo.utils import save_artifacts_yolo
from Architettura.yolo.labels import normalize_target
from Architettura.yolo.utils import save_yolo_png




def get_movement_primitives() -> list[str]:
    """
    Restituisce la lista delle primitive di movimento standard per l'agente iTHOR (default).
    """
    return [
        # Movimento Corpo
        "MoveAhead",
        "MoveBack",
        "MoveLeft",
        "MoveRight",
        
        # Rotazione
        "RotateRight",
        "RotateLeft",
        
        # Movimento Testa (Camera)
        "LookUp",
        "LookDown",
        
        # Postura
        "Crouch",
        "Stand",
        
        # Teletrasporto (God Mode/Debug)
        "Teleport",
        "TeleportFull"
    ]

def _save_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        Image.fromarray(image).save(path.as_posix())
        return
    if plt is not None:
        plt.imsave(path.as_posix(), image)
        return
    print(f"Salvataggio immagine saltato (manca PIL/matplotlib): {path}")


def _save_depth(path: Path, depth):
    if np is None:
        print("Salvataggio depth saltato (manca numpy).")
        return
    depth = np.asarray(depth, dtype=np.float32)
    dmin = float(np.min(depth))
    dmax = float(np.max(depth))
    if dmax - dmin < 1e-6:
        print("Depth uniforme: non salvo immagine.")
        return
    norm = (depth - dmin) / (dmax - dmin)
    img = (norm * 255).astype(np.uint8)
    _save_image(path, img)


def show_third_person_frame(event, frame_idx: int | None = None):
    """Salva il frame della third-person camera, se disponibile."""
    frames = getattr(event, "third_party_camera_frames", None)
    if not frames:
        return
    frame = frames[0]
    if frame_idx is None:
        if not hasattr(show_third_person_frame, "_idx"):
            show_third_person_frame._idx = 0
        frame_idx = show_third_person_frame._idx
        show_third_person_frame._idx += 1
    fname = THIRD_PERSON_DIR / f"third_person_frame_{frame_idx:05d}.png"
    _save_image(fname, frame)


def show_preview_frame(event, max_width: int = 640):
    """Mostra una preview ridotta del frame (utile con rendering alto)."""
    if cv2 is None:
        return
    frame = getattr(event, "frame", None)
    if frame is None:
        return
    try:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        scale = min(1.0, float(max_width) / float(w)) if w else 1.0
        if scale < 1.0:
            bgr = cv2.resize(
                bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow("AI2-THOR Preview", bgr)
        cv2.waitKey(1)
    except Exception:
        return


def update_third_person_camera(controller, event, position=None, rotation=None):
    if position is None:
        position = {"x": -1.25, "y": 1, "z": -1}
    if rotation is None:
        rotation = {"x": 90, "y": 0, "z": 0}
    controller.step(
        action="UpdateThirdPartyCamera",
        thirdPartyCameraId=0,
        position=position,
        rotation=rotation,
    )


def call_action(controller: Controller, action: str, **kwargs):
    """Chiama una primitiva AI2-THOR e ritorna l'evento."""
    return controller.step(action=action, **kwargs)


def print_primitives():
    actions=get_movement_primitives()
    """Stampa le primitive supportate."""
    for action in actions:
        print(f"- {action}")


def save_artifacts(
    event,
    frame_idx: int,
    target_label: str | None = None,
) -> dict:
    """Salva i risultati di visione e ritorna i percorsi."""
    show_third_person_frame(event, frame_idx=frame_idx)
    rgb_path = RGB_DIR / f"rgb_frame_{frame_idx:05d}.png"
    _save_image(rgb_path, event.frame)
    depth_frame = getattr(event, "depth_frame", None)
    depth_path = None
    if depth_frame is not None:
        depth_path = DEPTH_DIR / f"depth_frame_{frame_idx:05d}.png"
        _save_depth(depth_path, depth_frame)
    instance_frame = getattr(event, "instance_segmentation_frame", None)
    instance_path = None
    if instance_frame is not None:
        instance_path = INSTANCE_DIR / f"instance_frame_{frame_idx:05d}.png"
        _save_image(instance_path, instance_frame)
    semantic_frame = getattr(event, "semantic_segmentation_frame", None)
    semantic_path = None
    if semantic_frame is not None:
        semantic_path = SEMANTIC_DIR / f"semantic_frame_{frame_idx:05d}.png"
        _save_image(semantic_path, semantic_frame)
    oracle_data = None
    yolo_target_match = False
    yolo_paths = save_artifacts_yolo(
        event.frame,
        frame_idx=frame_idx,
        target_label=target_label,
        oracle_data=oracle_data,
    )
    oracle_path = None
    # Oracle disabilitato.
    oracle_path = None

    yolo = None
    try:
        yolo_conf = float(os.environ.get("YOLO_TARGET_CONF", "0.25"))
    except Exception:
        yolo_conf = 0.25
    try:
        yolo = save_yolo_png(
            event.frame,
            frame_idx=frame_idx,
            target_label=target_label,
            conf=yolo_conf,
            image_is_bgr=False,
        )
    except Exception:
        yolo = None
    yolo_match = (
        bool(yolo.get("match")) if isinstance(yolo, dict) else False
    )
    yolo_track_id = (
        yolo.get("track_id") if isinstance(yolo, dict) else None
    )
    yolo_target_center_x = (
        yolo.get("target_center_x") if isinstance(yolo, dict) else None
    )
    yolo_detections = []
    if isinstance(yolo, dict):
        raw_detections = yolo.get("detections")
        if isinstance(raw_detections, list):
            yolo_detections = raw_detections
    if isinstance(yolo, dict):
        yolo_min_conf = yolo.get("min_conf")
        target_center_x_fmt = (
            f"{float(yolo_target_center_x):.2f}"
            if yolo_target_center_x is not None
            else None
        )
        print(
            f"[YOLO] match={yolo_match} "
            f"track_id={yolo_track_id} "
            f"min_conf={yolo_min_conf} "
            f"target_center_x={target_center_x_fmt}"
        )
    # Fallback: usa metadata AI2-THOR per visibilità target (debug/ground truth).
    metadata_visible = False
    if target_label:
        target_norm = normalize_target(target_label)
        objects = event.metadata.get("objects") or []
        for obj in objects:
            obj_type = str(obj.get("objectType", "")).lower()
            if obj_type == target_norm and bool(obj.get("visible", False)):
                metadata_visible = True
                break
    if yolo_match:
        yolo_target_match = True
    elif metadata_visible:
        yolo_target_match = True
    yolo_json_rel = yolo_paths.get("yolo_json")
    if yolo_json_rel:
        yolo_json_abs = Path(yolo_json_rel)
        if not yolo_json_abs.is_absolute():
            yolo_json_abs = Path(__file__).resolve().parents[2] / yolo_json_abs
        try:
            yolo_json_abs.parent.mkdir(parents=True, exist_ok=True)
            yolo_json_abs.write_text(
                json.dumps(yolo_detections, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[WARN] Scrittura yolo_json fallita: {exc}", flush=True)
    return {
        "rgb": rel_to_project(rgb_path),
        "depth": (
            rel_to_project(depth_path) if depth_path is not None else None
        ),
        "instance": (
            rel_to_project(instance_path) if instance_path is not None else None
        ),
        "semantic": (
            rel_to_project(semantic_path) if semantic_path is not None else None
        ),
        "third_person": (
            rel_to_project(
                THIRD_PERSON_DIR / f"third_person_frame_{frame_idx:05d}.png"
            )
        ),
        "yolo_frame": yolo_paths.get("yolo_frame"),
        "yolo_json": yolo_paths.get("yolo_json"),
        "yolo_seg": yolo_paths.get("yolo_seg"),
        "yolo": (
            yolo.get("yolo") if isinstance(yolo, dict) else None
        ),
        "yolo_track_id": yolo_track_id,
        "yolo_target_center_x": yolo_target_center_x,
        "oracle_overlay": None,
        "oracle_object_id": None,
        "yolo_target_match": yolo_target_match,
        "metadata_target_visible": metadata_visible,
    }


def demo_actions(
    controller: Controller,
    actions: Iterable[tuple[str, dict]],
    delay_sec: float = 0.0,
    on_step=None,
):
    """Esegue una lista di azioni (nome, parametri) con callback opzionale."""
    if not hasattr(demo_actions, "_idx"):
        demo_actions._idx = 0
    for action_name, params in actions:
        event = call_action(controller, action_name, **params)
        update_third_person_camera(controller, event)
        frame_idx = demo_actions._idx
        paths = save_artifacts(event, frame_idx)
        if on_step is not None:
            on_step(event, action_name, frame_idx, paths)
        demo_actions._idx += 1
        if not event.metadata.get("lastActionSuccess", True):
            print(
                f"Azione fallita: {action_name} | errore: {event.metadata.get('errorMessage')}"
            )
        if delay_sec > 0:
            time.sleep(delay_sec)
