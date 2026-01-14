# Scopo: logiche specifiche dello stato NAVIGATE.
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from pipeline_modules.utils import normalize_action

from .state_names import STATE_APPROACH


def build_nav_scan_queue(degrees: int) -> deque:
    """
    Costruisce una coda di micro-rotazioni di scan.
    Usa rotazioni simmetriche intorno alla direzione corrente.
    Ritorna una deque di action spec.
    """
    # Micro-scan simmetrico per cercare il target dopo il nav plan.
    deg = int(max(5, min(45, degrees)))
    return deque(
        [
            {"action": "RotateLeft", "degrees": deg},
            {"action": "RotateRight", "degrees": deg * 2},
            {"action": "RotateLeft", "degrees": deg},
        ]
    )


def parse_nav_plan(raw: Dict[str, Any], max_steps: int) -> List[Dict[str, Any]]:
    """
    Parsa e normalizza un piano di navigazione VLM.
    Filtra azioni non valide e clampa gradi/magnitudine.
    Limita la lunghezza del piano a max_steps.
    """
    if not isinstance(raw, dict):
        return []
    plan = raw.get("nav_plan")
    if not isinstance(plan, list):
        return []
    parsed = []
    for item in plan:
        if not isinstance(item, dict):
            continue
        action = normalize_action(item.get("action"))
        if not action:
            continue
        spec = {"action": action}
        if action in {"RotateLeft", "RotateRight"}:
            deg = item.get("degrees")
            if deg is not None:
                try:
                    deg_val = int(float(deg))
                except Exception:
                    deg_val = None
                if deg_val is not None:
                    spec["degrees"] = max(1, min(90, deg_val))
        if action in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
            mag = item.get("moveMagnitude")
            if mag is not None:
                try:
                    mag_val = float(mag)
                except Exception:
                    mag_val = None
                if mag_val is not None:
                    spec["moveMagnitude"] = max(0.05, min(0.5, mag_val))
        parsed.append(spec)
        if len(parsed) >= int(max_steps):
            break
    return parsed


def parse_nav_subgoals(raw: Dict[str, Any], max_subgoals: int, max_steps: int) -> List[Dict[str, Any]]:
    """
    Parsa e normalizza i subgoal di navigazione.
    Pulisce testo e pianifica steps con parse_nav_plan.
    Limita il numero di subgoal a max_subgoals.
    """
    if not isinstance(raw, dict):
        return []
    items = raw.get("nav_subgoals")
    if not isinstance(items, list):
        return []
    parsed = []
    for item in items:
        if not isinstance(item, dict):
            continue
        goal = str(item.get("goal", "") or "").strip()
        expectation = str(item.get("expectation", "") or "").strip()
        plan = parse_nav_plan({"nav_plan": item.get("plan")}, max_steps)
        parsed.append(
            {
                "id": item.get("id"),
                "goal": goal,
                "expectation": expectation,
                "plan": plan,
            }
        )
        if len(parsed) >= int(max_subgoals):
            break
    return parsed


def next_state_navigate(
    recent_detection: bool,
    det_dist: Optional[float],
    approach_dist_thresh: float,
    navigate_dist_thresh: float,
) -> Optional[Tuple[str, str]]:
    """
    Decide la transizione da NAVIGATE verso APPROACH.
    Usa rilevamento recente e distanza stimata dal target.
    Ritorna None se deve continuare a navigare.
    """
    if not recent_detection:
        return None
    if det_dist is not None and det_dist <= float(approach_dist_thresh):
        return STATE_APPROACH, "nav_close"
    return None
