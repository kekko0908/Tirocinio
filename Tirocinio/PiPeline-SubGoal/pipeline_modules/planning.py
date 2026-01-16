# Scopo: logica di planning e parsing dei target.
import re

from .utils import extract_targets_from_goal, guess_target_from_goal, parse_near_relation


def fallback_plan(goal_text: str, target_hint: str):
    """
    Costruisce un piano di fallback senza VLM.
    Sceglie un target da hint o euristica.
    Ritorna dict con subgoals base.
    """
    target = target_hint or guess_target_from_goal(goal_text) or "Apple"
    return {
        "target_type": target,
        "subgoals": [
            {"id": 1, "type": "explore", "description": "Explore the scene and scan for the target."},
            {"id": 2, "type": "search", "description": "Move to keep the target centered in view."},
            {"id": 3, "type": "localize", "description": "Run YOLO to localize the target."},
        ],
    }


def build_target_queue(goal_text: str, target_hint: str):
    """
    Costruisce la coda target in modo deterministico.
    Gestisce relazione near e deduplicazione target.
    Ritorna (queue, near_relation).
    """
    # Costruisce la coda dei target da cercare in modo deterministico.
    targets = []
    queue = []
    near_relation = None

    if target_hint:
        raw = [t.strip() for t in re.split(r"[;,]", target_hint) if t.strip()]
        targets = [t.title() for t in raw if t]
    else:
        near_relation = parse_near_relation(goal_text or "")
        if near_relation:
            queue = [
                {"label": near_relation["reference"], "mode": "global"},
                {"label": near_relation["target"], "mode": "near", "reference": near_relation["reference"]},
            ]
        else:
            targets = extract_targets_from_goal(goal_text or "")
            if not targets:
                hint = guess_target_from_goal(goal_text or "")
                targets = [hint] if hint else []

    if not queue:
        if not targets:
            targets = ["Apple"]
        seen = set()
        for label in targets:
            if not label or label in seen:
                continue
            seen.add(label)
            queue.append({"label": label, "mode": "global"})
    return queue, near_relation


def build_fsm_plan(goal_text: str, target_queue, near_relation):
    """
    Costruisce il piano statico della FSM.
    Include obiettivo, target e lista stati.
    Ritorna un dict serializzabile.
    """
    return {
        "goal": goal_text,
        "targets": target_queue,
        "near_relation": near_relation,
        "states": [
            "SELECT_TARGET",
            "EXPLORE",
            "SEARCH",
            "SEARCH_NEAR",
            "NAVIGATE",
            "APPROACH",
            "LOCALIZE",
            "DONE",
            "FAIL",
        ],
    }
