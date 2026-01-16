# Scopo: lettura dello stato robot e distanze da depth.
import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def read_agent_pose(event) -> Tuple[Dict, Dict, float]:
    """
    Legge posizione, rotazione e horizon dai metadata.
    Normalizza i dizionari con default sicuri.
    Ritorna (pos, rot, horizon).
    """
    meta = getattr(event, "metadata", {}) or {}
    agent = meta.get("agent", {}) or {}
    pos = agent.get("position", {}) or {}
    rot = agent.get("rotation", {}) or {}
    horizon = float(agent.get("cameraHorizon", 0.0) or 0.0)
    return pos, rot, horizon


def sample_depth_window(depth: np.ndarray, cx: float, cy: float, radius: int = 12) -> Optional[float]:
    """
    Campiona una finestra depth intorno a un punto.
    Filtra valori validi e calcola un percentile robusto.
    Ritorna la distanza stimata o None.
    """
    if depth is None or depth.size == 0:
        return None
    h, w = depth.shape[:2]
    x0 = max(0, int(cx - radius))
    x1 = min(w, int(cx + radius + 1))
    y0 = max(0, int(cy - radius))
    y1 = min(h, int(cy + radius + 1))
    patch = depth[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    valid = patch[np.isfinite(patch)]
    if valid.size == 0:
        return None
    return float(np.percentile(valid, 10))


def estimate_target_world(event, target_type: str, centroid_px, depth_radius: int = 12) -> Optional[Dict]:
    """
    Stima la posizione mondo del target.
    Usa metadata se visibile, altrimenti depth e intrinsics.
    Ritorna dict con position/source o None.
    """
    meta = getattr(event, "metadata", {}) or {}
    objs = meta.get("objects", []) or []
    visible = [o for o in objs if o.get("visible") and o.get("objectType") == target_type]
    if visible:
        best = sorted(visible, key=lambda x: float(x.get("distance", 1e9)))[0]
        pos = best.get("position", {}) or {}
        return {"position": pos, "source": "metadata", "object_id": best.get("objectId")}

    depth = getattr(event, "depth_frame", None)
    if depth is None:
        return None
    h, w = depth.shape[:2]
    u, v = int(centroid_px[0]), int(centroid_px[1])
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))
    dist = sample_depth_window(depth, u, v, radius=depth_radius)
    if dist is None:
        return None

    fov = float(meta.get("fov", meta.get("cameraFOV", 90.0)) or 90.0)
    fx = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    fy = fx
    x_cam = (u - (w / 2.0)) / fx * dist
    z_cam = dist

    agent = (meta.get("agent") or {})
    pos = agent.get("position", {}) or {}
    rot = agent.get("rotation", {}) or {}
    yaw = math.radians(float(rot.get("y", 0.0) or 0.0))

    world_x = float(pos.get("x", 0.0)) + x_cam * math.cos(yaw) + z_cam * math.sin(yaw)
    world_z = float(pos.get("z", 0.0)) + z_cam * math.cos(yaw) - x_cam * math.sin(yaw)
    world_y = float(pos.get("y", 0.0))
    return {"position": {"x": world_x, "y": world_y, "z": world_z}, "source": "depth"}


def build_sensor_state(event, depth_radius: int = 12) -> Dict:
    """
    Costruisce lo stato sensori dal frame corrente.
    Calcola distanze front/left/right dal depth.
    Ritorna un dict con collisione e pose.
    """
    meta = getattr(event, "metadata", {}) or {}
    collided = bool(meta.get("collided", False))
    last_action_success = bool(meta.get("lastActionSuccess", True))
    pos, rot, horizon = read_agent_pose(event)
    yaw = float((rot or {}).get("y", 0.0) or 0.0)
    depth = getattr(event, "depth_frame", None)

    dist_front = dist_left = dist_right = None
    if depth is not None:
        h, w = depth.shape[:2]
        dist_front = sample_depth_window(depth, w * 0.5, h * 0.5, radius=depth_radius)
        dist_left = sample_depth_window(depth, w * 0.25, h * 0.5, radius=depth_radius)
        dist_right = sample_depth_window(depth, w * 0.75, h * 0.5, radius=depth_radius)

    default_dist = 2.5
    return {
        "dist_front_m": float(dist_front if dist_front is not None else default_dist),
        "dist_left_m": float(dist_left if dist_left is not None else default_dist),
        "dist_right_m": float(dist_right if dist_right is not None else default_dist),
        "collision": collided,
        "last_action_success": last_action_success,
        "position": pos,
        "yaw": yaw,
        "horizon": horizon,
    }


def save_depth_frame(depth: np.ndarray, out_path) -> None:
    """
    Salva il depth frame normalizzato in scala di grigi.
    Clampa range e gestisce NaN/inf prima di scrivere.
    Scrive l'immagine sul percorso specificato.
    """
    if depth is None or depth.size == 0:
        return
    d = depth.copy()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip(d, 0.0, 5.0)
    d = (d / 5.0 * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_path), d)
