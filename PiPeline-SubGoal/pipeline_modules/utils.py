# Scopo: funzioni di utilita per parsing, normalizzazione e json.
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

KNOWN_TARGETS = {
    "apple": ["apple", "mela"],
    "mug": ["mug", "tazza", "cup"],
    "bottle": ["bottle", "bottiglia"],
    "lettuce": ["lettuce", "insalata"],
    "banana": ["banana"],
    "book": ["book", "libro"],
    "bowl": ["bowl", "ciotola"],
}


def _extract_balanced_json(text: str):
    """
    Estrae un oggetto JSON bilanciato da testo.
    Gestisce stringhe e escape per non rompere le graffe.
    Ritorna dict/list o None se non valido.
    """
    start = text.find("{")
    if start == -1:
        return None
    in_str = False
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def extract_json_anywhere(text: str):
    """
    Cerca un JSON valido in un testo libero.
    Supporta blocchi fenced e ricerca da destra.
    Ritorna l'oggetto JSON o None.
    """
    # 1) Fenced JSON blocks (anche senza fence di chiusura).
    for m in re.finditer(r"```json\s*(\{.*?)(?:```|$)", text, flags=re.DOTALL):
        candidate = m.group(1).strip()
        data = _extract_balanced_json(candidate)
        if data is not None:
            return data
    # 2) Cerca da destra l'ultimo oggetto JSON valido.
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(starts):
        data = _extract_balanced_json(text[start:])
        if data is not None:
            return data
    return None


def safe_write_json(path: Path, obj) -> None:
    """
    Scrive JSON su disco creando le directory.
    Serializza con indent e encoding utf-8.
    Ritorna None dopo la scrittura.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def normalize_label(text: str) -> str:
    """
    Normalizza etichette in formato alfanumerico.
    Rimuove spazi e caratteri non validi.
    Ritorna la label in lower case.
    """
    return re.sub(r"[^a-z0-9_]+", "", text.strip().lower())


def guess_target_from_goal(goal_text: str) -> str:
    """
    Prova a indovinare il target dal testo del goal.
    Confronta parole chiave note per oggetto.
    Ritorna la label Title Case o stringa vuota.
    """
    goal_l = goal_text.lower()
    for label, keys in KNOWN_TARGETS.items():
        if any(k in goal_l for k in keys):
            return label.title()
    return ""


def _find_first_label(text: str) -> str:
    """
    Trova la prima label presente nel testo.
    Usa la posizione piu a sinistra tra le keyword.
    Ritorna la label Title Case o stringa vuota.
    """
    text_l = text.lower()
    best = None
    for label, keys in KNOWN_TARGETS.items():
        for k in keys:
            idx = text_l.find(k)
            if idx == -1:
                continue
            if best is None or idx < best[0]:
                best = (idx, label.title())
            break
    if not best:
        return ""
    return best[1]


def extract_targets_from_goal(goal_text: str) -> List[str]:
    """
    Estrae tutte le label presenti nel goal.
    Ordina per posizione e rimuove duplicati.
    Ritorna la lista ordinata di target.
    """
    goal_l = goal_text.lower()
    hits = []
    for label, keys in KNOWN_TARGETS.items():
        idx = None
        for k in keys:
            pos = goal_l.find(k)
            if pos != -1:
                idx = pos if idx is None else min(idx, pos)
        if idx is not None:
            hits.append((idx, label.title()))
    hits.sort(key=lambda x: x[0])
    ordered = []
    seen = set()
    for _, label in hits:
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def parse_near_relation(goal_text: str) -> Optional[Dict[str, str]]:
    """
    Parsa una relazione "near/vicino" dal goal.
    Identifica target e reference distinti.
    Ritorna dict con target/reference o None.
    """
    goal_l = goal_text.lower()
    if "vicino" in goal_l:
        left, right = goal_l.split("vicino", 1)
    elif "near" in goal_l:
        left, right = goal_l.split("near", 1)
    else:
        return None
    target = _find_first_label(left)
    reference = _find_first_label(right)
    if not target or not reference or target == reference:
        return None
    return {"target": target, "reference": reference}


def normalize_action(action: str) -> Optional[str]:
    """
    Normalizza nome azione a formato canonico.
    Mappa sinonimi e varianti a ActionSet.
    Ritorna stringa azione o None.
    """
    if not action:
        return None
    a = re.sub(r"\s+", "", str(action)).lower()
    mapping = {
        "moveahead": "MoveAhead",
        "moveback": "MoveBack",
        "movebackward": "MoveBack",
        "movebackwards": "MoveBack",
        "moveleft": "MoveLeft",
        "moveright": "MoveRight",
        "strafeleft": "MoveLeft",
        "straferight": "MoveRight",
        "rotateleft": "RotateLeft",
        "rotateright": "RotateRight",
    }
    return mapping.get(a)


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Calcola IoU tra due bounding box xyxy.
    Gestisce intersezione e area di unione.
    Ritorna un valore float tra 0 e 1.
    """
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
    """
    Parsa una bbox da testo libero o JSON.
    Clampa ai limiti immagine e valida il box.
    Ritorna lista [x1,y1,x2,y2] o None.
    """
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
        """
        Convalida e limita una bbox ai limiti immagine.
        Corregge coordinate fuori range e verifica area positiva.
        Ritorna la bbox clamped o None.
        """
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
    """
    Formatta uno spec azione in stringa compatta.
    Include gradi o magnitudine se presenti.
    Ritorna la stringa pronta per log.
    """
    action = str(spec.get("action", ""))
    degrees = spec.get("degrees", None)
    if degrees is not None and action in {"RotateLeft", "RotateRight"}:
        return f"{action}({int(degrees)})"
    move_mag = spec.get("moveMagnitude", None)
    if move_mag is not None and action in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
        try:
            mag_val = float(move_mag)
        except Exception:
            mag_val = None
        if mag_val is not None:
            return f"{action}({mag_val:.2f})"
    return action
