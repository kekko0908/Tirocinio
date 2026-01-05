# Scopo: overlay e frame di evidenza/telemetria.
from typing import Dict, List, Optional

import cv2
import numpy as np

from .utils import format_action_spec


def draw_detection(frame_bgr: np.ndarray, det: dict, mask01: Optional[np.ndarray] = None):
    x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
    cx, cy = det["centroid_px"]
    vis = frame_bgr.copy()
    if mask01 is not None and mask01.size > 0:
        mask8 = (mask01 > 0.5).astype(np.uint8) * 255
        color = np.zeros_like(vis)
        color[:, :, 1] = mask8
        vis = cv2.addWeighted(vis, 1.0, color, 0.35, 0.0)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    label = f"{det['label']} {det['score']:.2f}"
    cv2.putText(vis, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def draw_yolo_evidence(frame_bgr: np.ndarray, det: dict, draw_centroid: bool = True):
    x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
    cx, cy = det["centroid_px"]
    vis = frame_bgr.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if draw_centroid:
        cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    label = f"{det['label']} {det['score']:.2f}"
    cv2.putText(vis, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def draw_comparison(frame_bgr: np.ndarray, yolo_det: dict, vlm_bbox: List[float]):
    vis = frame_bgr.copy()
    yx1, yy1, yx2, yy2 = [int(v) for v in yolo_det["bbox_xyxy"]]
    cv2.rectangle(vis, (yx1, yy1), (yx2, yy2), (0, 255, 255), 2)
    vx1, vy1, vx2, vy2 = [int(v) for v in vlm_bbox]
    cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), (255, 0, 255), 2)
    cv2.putText(vis, "YOLO", (yx1, max(20, yy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, "VLM", (vx1, max(20, vy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return vis


def draw_vlm_bbox(frame_bgr: np.ndarray, vlm_bbox: List[float], label: str = "VLM"):
    vis = frame_bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in vlm_bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(vis, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return vis


def draw_vlm_not_visible(frame_bgr: np.ndarray, message: str = "VLM NOT_VISIBLE"):
    vis = frame_bgr.copy()
    cv2.putText(vis, message, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(vis, message, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return vis


def annotate_telemetry(
    frame_bgr: np.ndarray,
    step_idx: int,
    action_spec: Dict,
    sensor: Dict,
    mem_summary: Dict,
) -> np.ndarray:
    vis = frame_bgr.copy()
    action_txt = format_action_spec(action_spec)
    cell = mem_summary.get("pose_discrete", {}).get("cell", {})
    cell_txt = f"cell=({cell.get('x', 0)},{cell.get('z', 0)})"
    novelty = mem_summary.get("novelty_score", 0.0)
    coverage = mem_summary.get("coverage_pct", 0.0)
    collision = "YES" if sensor.get("collision", False) else "NO"
    dist_f = sensor.get("dist_front_m", 0.0)
    dist_l = sensor.get("dist_left_m", 0.0)
    dist_r = sensor.get("dist_right_m", 0.0)
    lines = [
        f"step={step_idx} action={action_txt} collision={collision}",
        f"{cell_txt} coverage={coverage:.1f}% novelty={novelty:.2f}",
        f"dist(m) F={dist_f:.2f} L={dist_l:.2f} R={dist_r:.2f}",
    ]
    y = 24
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24
    return vis
