# Scopo: funzioni di utilita per parsing, normalizzazione e json.
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def extract_json_anywhere(text: str):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def safe_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", text.strip().lower())


def guess_target_from_goal(goal_text: str) -> str:
    goal_l = goal_text.lower()
    known = {
        "apple": ["apple", "mela"],
        "mug": ["mug", "tazza", "cup"],
        "bottle": ["bottle", "bottiglia"],
        "lettuce": ["lettuce", "insalata"],
        "banana": ["banana"],
        "book": ["book", "libro"],
        "bowl": ["bowl", "ciotola"],
    }
    for label, keys in known.items():
        if any(k in goal_l for k in keys):
            return label.title()
    return ""


def normalize_action(action: str) -> Optional[str]:
    if not action:
        return None
    a = re.sub(r"\s+", "", str(action)).lower()
    mapping = {
        "moveahead": "MoveAhead",
        "rotateleft": "RotateLeft",
        "rotateright": "RotateRight",
        "lookup": "LookUp",
        "lookdown": "LookDown",
        "done": "Done",
        "stop": "Done",
    }
    return mapping.get(a)


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def parse_vlm_bbox(text: str, width: int, height: int) -> Optional[List[float]]:
    if not text:
        return None
    if "NOT_VISIBLE" in text.upper():
        return None
    data = extract_json_anywhere(text)
    if isinstance(data, dict):
        if all(k in data for k in ["x1", "y1", "x2", "y2"]):
            box = [data["x1"], data["y1"], data["x2"], data["y2"]]
        elif "bbox" in data and isinstance(data["bbox"], list) and len(data["bbox"]) == 4:
            box = data["bbox"]
        else:
            return None
    elif isinstance(data, list) and len(data) == 4:
        box = data
    else:
        return None

    def clamp_box(vals):
        x1, y1, x2, y2 = vals
        x1 = max(0.0, min(x1, float(width - 1)))
        y1 = max(0.0, min(y1, float(height - 1)))
        x2 = max(0.0, min(x2, float(width)))
        y2 = max(0.0, min(y2, float(height)))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    try:
        vals = [float(v) for v in box]
    except Exception:
        vals = []
    if len(vals) == 4:
        clamped = clamp_box(vals)
        if clamped:
            return clamped

    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(numbers) >= 4:
        nums = [float(n) for n in numbers]
        for i in range(0, len(nums) - 3):
            candidate = clamp_box(nums[i : i + 4])
            if candidate:
                return candidate
    return None


def format_action_spec(spec: Dict) -> str:
    action = str(spec.get("action", ""))
    degrees = spec.get("degrees", None)
    if degrees is not None and action in {"RotateLeft", "RotateRight"}:
        return f"{action}({int(degrees)})"
    return action
