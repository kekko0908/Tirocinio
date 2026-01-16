# Scopo: logiche specifiche dello stato APPROACH.
from typing import Any, Dict, Optional, Tuple

import math
import numpy as np

from pipeline_modules.sensors import estimate_target_world

from .state_names import STATE_LOCALIZE, STATE_NAVIGATE, STATE_SEARCH


def get_oracle_target(event, target_type: str) -> Optional[Dict[str, Any]]:
    """
    Estrae informazioni del target dai metadata visibili.
    Calcola centroid, area maschera e bbox se disponibili.
    Ritorna un dict con dettagli o None se non visibile.
    """
    meta = getattr(event, "metadata", {}) or {}
    objs = meta.get("objects", []) or []
    target_l = str(target_type or "").lower()
    visible = [o for o in objs if o.get("visible") and str(o.get("objectType", "")).lower() == target_l]
    if not visible:
        return None
    obj = sorted(visible, key=lambda x: float(x.get("distance", 1e9)))[0]
    centroid = None
    mask_area = None
    bbox = None
    masks = getattr(event, "instance_masks", None) or {}
    if obj.get("objectId") in masks:
        mask = masks.get(obj.get("objectId"))
        if mask is not None and mask.size:
            ys, xs = np.where(mask)
            if xs.size > 0:
                centroid = (int(xs.mean()), int(ys.mean()))
                mask_area = int(mask.sum())
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {"object": obj, "centroid": centroid, "mask_area": mask_area, "bbox_xyxy": bbox}


def compute_approach_action(
    event,
    det: Optional[Dict[str, Any]],
    sensor: Dict[str, Any],
    args,
    target_type: str,
    approach_confirm: int,
    approach_stuck_steps: int,
) -> Tuple[Dict[str, Any], bool, int]:
    """
    Calcola la prossima azione nello stato APPROACH.
    Integra centramento, distanza, sicurezza e recovery.
    Ritorna spec azione, flag close_enough e contatore conferma.
    """
    # Logica APPROACH: prima centriamo, poi avviciniamo con step piccoli, con evasive laterali se bloccati.
    oracle = get_oracle_target(event, target_type)
    if oracle is None:
        det = None
    target_visible = oracle is not None
    if not target_visible:
        # Perso target? Piccola rotazione locale per ritrovarlo.
        search_deg = max(10, int(args.approach_rotate_degrees))
        return {"action": "RotateRight", "degrees": search_deg}, False, approach_confirm

    if sensor.get("collision") or not sensor.get("last_action_success", True):
        # Recovery deterministico: schivata laterale o rotazione verso il lato piu' libero.
        left = float(sensor.get("dist_left_m", 0.0))
        right = float(sensor.get("dist_right_m", 0.0))
        safe_m = float(args.safe_front_m)
        if left >= right and left >= safe_m:
            return {"action": "MoveLeft", "moveMagnitude": 0.25}, False, approach_confirm
        if right >= safe_m:
            return {"action": "MoveRight", "moveMagnitude": 0.25}, False, approach_confirm
        turn = "RotateLeft" if left >= right else "RotateRight"
        return {"action": turn, "degrees": int(args.approach_rotate_degrees)}, False, approach_confirm

    h, w = event.frame.shape[:2]
    center_x = int(w / 2)
    cx = None
    if oracle and oracle.get("centroid") is not None:
        cx = int(oracle["centroid"][0])
    elif det is not None:
        cx = int(det["centroid_px"][0])

    if cx is None and oracle and oracle.get("object"):
        obj_pos = (oracle.get("object") or {}).get("position") or {}
        agent = (event.metadata.get("agent") or {})
        agent_pos = agent.get("position") or {}
        agent_rot = agent.get("rotation") or {}
        if obj_pos and agent_pos:
            dx = float(obj_pos.get("x", 0.0)) - float(agent_pos.get("x", 0.0))
            dz = float(obj_pos.get("z", 0.0)) - float(agent_pos.get("z", 0.0))
            angle_to_obj = math.degrees(math.atan2(dx, dz))
            yaw = float(agent_rot.get("y", 0.0) or 0.0)
            bearing_err = (angle_to_obj - yaw + 180.0) % 360.0 - 180.0
            fov = float((event.metadata or {}).get("fov", (event.metadata or {}).get("cameraFOV", 90.0)) or 90.0)
            tol_deg = max(2.0, (float(args.approach_center_tol_px) / max(1.0, w / 2.0)) * (fov / 2.0))
            if abs(bearing_err) > tol_deg:
                direction = "RotateRight" if bearing_err > 0 else "RotateLeft"
                return {"action": direction, "degrees": int(args.approach_rotate_degrees)}, False, approach_confirm

    if cx is None:
        cx = center_x
    err = cx - center_x
    if abs(err) > int(args.approach_center_tol_px):
        direction = "RotateRight" if err > 0 else "RotateLeft"
        return {"action": direction, "degrees": int(args.approach_rotate_degrees)}, False, approach_confirm

    dist = None
    if oracle and oracle.get("object"):
        try:
            dist = float(oracle["object"].get("distance", 0.0))
        except Exception:
            dist = None
    if dist is None and det is not None:
        world = estimate_target_world(event, target_type, det["centroid_px"], depth_radius=args.depth_radius)
        if world and world.get("position"):
            pos = (event.metadata.get("agent") or {}).get("position", {}) or {}
            dx = float(world["position"].get("x", 0.0)) - float(pos.get("x", 0.0))
            dz = float(world["position"].get("z", 0.0)) - float(pos.get("z", 0.0))
            dist = (dx * dx + dz * dz) ** 0.5
    if dist is None:
        dist = float(sensor.get("dist_front_m", 0.0))

    front = float(sensor.get("dist_front_m", 0.0))
    left = float(sensor.get("dist_left_m", 0.0))
    right = float(sensor.get("dist_right_m", 0.0))
    safe_m = float(args.safe_front_m)

    bbox_area = 0.0
    if oracle and oracle.get("mask_area") is not None:
        bbox_area = float(oracle["mask_area"])
    elif det is not None:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        bbox_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    close_enough = False
    if dist is not None and dist <= float(args.approach_dist_thresh_m):
        close_enough = True
    if bbox_area >= float(args.approach_bbox_area_thresh):
        close_enough = True

    if not close_enough and front < min(0.35, safe_m):
        # Stop anticipato se un ostacolo e' troppo vicino al centro.
        approach_confirm += 1
        return (
            {"action": "RotateLeft", "degrees": int(args.approach_rotate_degrees)},
            approach_confirm >= int(args.approach_confirm_k),
            approach_confirm,
        )

    if close_enough:
        approach_confirm += 1
        return (
            {"action": "RotateLeft", "degrees": int(args.approach_rotate_degrees)},
            approach_confirm >= int(args.approach_confirm_k),
            approach_confirm,
        )

    approach_confirm = max(0, approach_confirm - 1)

    # Se siamo bloccati, proviamo una schivata laterale (logica simile a 13_autonomous_exploration).
    if approach_stuck_steps >= max(1, int(args.nav_stuck_steps) - 1):
        if left >= right and left >= safe_m:
            return {"action": "MoveLeft", "moveMagnitude": 0.25}, False, approach_confirm
        if right >= safe_m:
            return {"action": "MoveRight", "moveMagnitude": 0.25}, False, approach_confirm
        turn = "RotateLeft" if left >= right else "RotateRight"
        return {"action": turn, "degrees": int(args.approach_rotate_degrees)}, False, approach_confirm

    # Se c'è un ostacolo frontale, meglio una piccola traslazione laterale.
    if front < safe_m:
        if left >= right and left >= safe_m:
            return {"action": "MoveLeft", "moveMagnitude": 0.25}, False, approach_confirm
        if right >= safe_m:
            return {"action": "MoveRight", "moveMagnitude": 0.25}, False, approach_confirm
        turn = "RotateLeft" if left >= right else "RotateRight"
        return {"action": turn, "degrees": int(args.approach_rotate_degrees)}, False, approach_confirm

    step_sz = 0.15 if dist is not None and dist < 1.2 else 0.25
    return {"action": "MoveAhead", "moveMagnitude": step_sz}, False, approach_confirm


def next_state_approach(
    close_enough: bool, approach_stuck_steps: int, nav_stuck_steps: int, lost_target: bool
) -> Optional[Tuple[str, str]]:
    """
    Decide la transizione dallo stato APPROACH.
    Passa a LOCALIZE se vicino, a NAVIGATE se bloccato.
    Passa a SEARCH se il target e perso.
    """
    if close_enough:
        return STATE_LOCALIZE, "approach_close"
    if approach_stuck_steps >= int(nav_stuck_steps):
        return STATE_NAVIGATE, "approach_stuck"
    if lost_target:
        return STATE_SEARCH, "lost_target"
    return None
