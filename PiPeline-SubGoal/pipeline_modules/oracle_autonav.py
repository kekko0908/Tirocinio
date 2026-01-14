# Scopo: logica "oracle" stile 13_autonomous_exploration per l'approach.
import math
from typing import Dict, Optional, Tuple

import numpy as np


def dist_xz(p1: Dict, p2: Dict) -> float:
    """
    Calcola distanza euclidea sul piano xz.
    Usa coordinate x e z dei dizionari.
    Ritorna un float in metri.
    """
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["z"] - p2["z"]) ** 2)


def get_depth_center_distance(event) -> float:
    """
    Stima distanza media dal patch centrale depth.
    Filtra valori validi e usa fallback se vuoto.
    Ritorna una distanza robusta per ostacoli.
    """
    depth = getattr(event, "depth_frame", None)
    if depth is None:
        return 10.0
    h, w = depth.shape[:2]
    center_patch = depth[h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10]
    valid = center_patch[center_patch > 0]
    return float(np.mean(valid)) if len(valid) > 0 else 10.0


def find_target_in_view(event, target_type: str) -> Tuple[bool, Optional[Dict], Optional[int]]:
    """
    Cerca il target visibile nei metadata dell'evento.
    Se presente, calcola il centro della maschera.
    Ritorna (seen, obj_meta, cx).
    """
    meta = (getattr(event, "metadata", {}) or {}).get("objects", []) or []
    target_l = str(target_type or "").lower()
    visible = [o for o in meta if str(o.get("objectType", "")).lower() == target_l and o.get("visible")]
    if not visible:
        return False, None, None
    target = sorted(visible, key=lambda x: float(x.get("distance", 1e9)))[0]
    tid = target.get("objectId")

    cx = None
    masks = getattr(event, "instance_masks", None) or {}
    if tid in masks:
        ys, xs = np.where(masks[tid])
        if len(xs) > 0:
            cx = int(np.mean(xs))
    return True, target, cx


def oracle_approach_step(
    event,
    target_type: str,
    args,
    sensor: Dict,
    last_pos: Optional[Dict],
    stuck_counter: int,
) -> Tuple[Dict, bool, Optional[Dict], int, Dict]:
    """
    Esegue uno step di approach deterministico "oracle".
    Gestisce stuck, centramento e avanzamento sicuro.
    Ritorna action spec, close flag, pos e debug.
    """
    # Implementazione deterministica ispirata al file 13.
    curr_pos = (event.metadata.get("agent") or {}).get("position", {}) or {}
    if last_pos and dist_xz(curr_pos, last_pos) < 0.01:
        stuck_counter += 1
    else:
        stuck_counter = 0
    last_pos = curr_pos

    obstacle_dist = get_depth_center_distance(event)
    seen, obj_meta, cx = find_target_in_view(event, target_type)
    debug = {
        "seen": bool(seen),
        "cx": cx,
        "dist": float(obj_meta.get("distance")) if obj_meta else None,
        "obstacle_dist": float(obstacle_dist),
        "stuck_counter": int(stuck_counter),
    }

    # Target non visibile: micro-rotazione locale.
    if not seen:
        search_deg = max(10, int(args.approach_rotate_degrees))
        return {"action": "RotateRight", "degrees": search_deg}, False, last_pos, stuck_counter, debug

    dist = float(obj_meta.get("distance", 0.0))

    # Se bloccato mentre vede il target, schivata laterale deterministica.
    if stuck_counter > 3:
        left = float(sensor.get("dist_left_m", 0.0))
        right = float(sensor.get("dist_right_m", 0.0))
        if right >= left:
            return {"action": "MoveRight", "moveMagnitude": 0.25}, False, last_pos, 0, debug
        return {"action": "MoveLeft", "moveMagnitude": 0.25}, False, last_pos, 0, debug

    # 1) Centramento preciso con mask (se presente).
    center_x = int(event.frame.shape[1] / 2)
    tol = int(args.approach_center_tol_px)
    if cx is not None:
        if cx < (center_x - tol):
            return {"action": "RotateLeft", "degrees": 5}, False, last_pos, stuck_counter, debug
        if cx > (center_x + tol):
            return {"action": "RotateRight", "degrees": 5}, False, last_pos, stuck_counter, debug

    # 2) Avanzamento se centrato.
    if dist > float(args.approach_dist_thresh_m):
        if obstacle_dist < 0.35:
            return {"action": "RotateLeft", "degrees": int(args.approach_rotate_degrees)}, True, last_pos, stuck_counter, debug
        step_sz = 0.15 if dist < 1.2 else 0.25
        if not sensor.get("last_action_success", True):
            return {"action": "MoveRight", "moveMagnitude": 0.10}, False, last_pos, stuck_counter, debug
        return {"action": "MoveAhead", "moveMagnitude": step_sz}, False, last_pos, stuck_counter, debug

    # Target vicino: considerato raggiunto.
    return {"action": "RotateLeft", "degrees": int(args.approach_rotate_degrees)}, True, last_pos, stuck_counter, debug
