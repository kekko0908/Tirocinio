# Scopo: template e prompt VLM centralizzati.
import json
from typing import Dict, List, Tuple


def build_plan_subgoals_messages(goal_text: str) -> Tuple[str, str]:
    """
    Costruisce i messaggi di prompt per subgoal.
    Genera system e user in formato testo.
    Ritorna una tupla (system, user).
    """
    system = (
        "You are a planner for an AI2-THOR robot. "
        "Return JSON only with keys: target_type, subgoals. "
        "subgoals is a list of objects {id, type, description}."
    )
    user = (
        "Goal: "
        + goal_text
        + "\nCreate 3-5 subgoals for search and localization. "
        "Use types: explore, search, approach, localize."
    )
    return system, user


def build_choose_action_messages(
    action_set: List[str], state_desc: str, history, context_summary: Dict
) -> Tuple[str, str]:
    """
    Costruisce i messaggi per scegliere un'azione.
    Include action_set, stato e context_summary.
    Ritorna una tupla (system, user).
    """
    system = (
        "You control an AI2-THOR robot at low level. "
        "Pick exactly one action from: "
        + ", ".join(action_set)
        + ". Return JSON only: "
        "{\"action\": \"...\", \"degrees\": optional int, \"reason\": \"short\", "
        "\"target_confidence\": 0.0-1.0, \"request_yolo\": true|false}. "
        "Use the Context JSON as authoritative facts. "
        "Hard rule: NEVER MoveAhead if dist_front_m < safe_front_m. "
        "If front is clear and the target is far, prefer MoveAhead to reduce distance. "
        "If front is clear and oracle_target.distance_m is known, keep advancing even if the target is not visible. "
        "Do NOT rotate just to re-acquire a target that is already known from Context. "
        "If distance to target does not decrease for multiple steps, assume an obstacle and choose a lateral move or rotation. "
        "Safety: if dist_front_m < safe_front_m then do NOT MoveAhead; "
        "if dist_left_m < safe_front_m then do NOT MoveLeft; "
        "if dist_right_m < safe_front_m then do NOT MoveRight; "
        "if collision or last_action_success=false then prefer MoveLeft/MoveRight/MoveBack; "
        "rotate only if no side/back is safe. "
        "Use target_location_hint only when oracle_target.distance_m is unknown or the target is visible; "
        "then: left/right -> rotate toward hint; center -> move ahead if safe. "
        "Do NOT use LookUp or LookDown. "
        "State policy: EXPLORE = scan and widen coverage; SEARCH = scan and adjust to maximize recall; "
        "APPROACH = center target and reduce distance with small rotations; "
        "LOCALIZE = keep target stable and visible. "
        "Use MoveBack or small side-steps only when you need more space and it is safe. "
        "If oracle_target.distance_m is known and state is NAVIGATE, avoid request_yolo. "
        "Set request_yolo=true when the target is likely visible or probe is positive."
    )
    hist = ", ".join(history[-6:]) if history else "none"
    ctx = json.dumps(context_summary, ensure_ascii=True)
    user = (
        f"State: {state_desc}\n"
        f"Recent actions: {hist}\n"
        f"Context: {ctx}\n"
        "Pick the best next action to find the target with the current state policy."
    )
    return system, user


def build_plan_navigation_messages(
    action_set: List[str], target_label: str, context_summary: Dict
) -> Tuple[str, str]:
    """
    Costruisce il prompt per un nav_plan breve.
    Usa action_set, target_label e contesto.
    Ritorna una tupla (system, user).
    """
    system = (
        "You control an AI2-THOR robot at low level. "
        "Task: generate a short navigation plan to reach a far or occluded target. "
        "Return JSON only with keys: nav_plan, confidence, rationale, route_side. "
        "route_side is one of [\"left\",\"right\",\"straight\",\"unknown\"]. "
        "rationale must be 1-2 concise sentences that mention obstacles and free space. "
        "nav_plan is a list of 3-6 actions. Each action is "
        "{\"action\": \"...\", \"degrees\": optional int}. "
        "Allowed actions: "
        + ", ".join(action_set)
        + ". "
        "Use Context as facts. Avoid collisions: if dist_front_m < safe_front_m then do NOT MoveAhead; "
        "if dist_left_m < safe_front_m then do NOT MoveLeft; "
        "if dist_right_m < safe_front_m then do NOT MoveRight. "
        "If front is clear and oracle_target.distance_m is known, prioritize MoveAhead steps even if the target is not visible. "
        "Do NOT add rotations just to re-acquire a target that is already known from Context. "
        "If blocked or progress stalls, prefer MoveLeft/MoveRight/MoveBack before rotations. "
        "Prefer small rotations (5-45 degrees). "
        "Do NOT use LookUp or LookDown. "
        "If uncertain, return an empty nav_plan with low confidence."
    )
    ctx = json.dumps(context_summary, ensure_ascii=True)
    user = (
        f"TARGET: {target_label}\n"
        f"Context: {ctx}\n"
        "Return a precise nav_plan to get closer to the target, using visible landmarks when possible."
    )
    return system, user


def build_plan_navigation_subgoals_messages(
    action_set: List[str], target_label: str, context_summary: Dict
) -> Tuple[str, str]:
    """
    Costruisce il prompt per nav_subgoals.
    Specifica formato JSON e vincoli di sicurezza.
    Ritorna una tupla (system, user).
    """
    system = (
        "You control an AI2-THOR robot at low level. "
        "Task: break navigation into 3-5 subgoals to reach a far or occluded target. "
        "Return JSON only with keys: nav_subgoals, confidence, rationale, route_side. "
        "route_side is one of [\"left\",\"right\",\"straight\",\"unknown\"]. "
        "rationale must be 1-2 concise sentences mentioning obstacles and free space. "
        "nav_subgoals is a list of objects with keys: id, goal, plan, expectation. "
        "plan is a list of 2-5 actions; each action is {\"action\": \"...\", \"degrees\": optional int}. "
        "Allowed actions: "
        + ", ".join(action_set)
        + ". "
        "Avoid collisions: if dist_front_m < safe_front_m then do NOT MoveAhead; "
        "if dist_left_m < safe_front_m then do NOT MoveLeft; "
        "if dist_right_m < safe_front_m then do NOT MoveRight. "
        "If front is clear and target distance is high, include MoveAhead in the plan to reduce distance. "
        "If front is clear and oracle_target.distance_m is known, prioritize consecutive MoveAhead steps. "
        "Do NOT add rotations just to re-acquire a target that is already known from Context. "
        "Prefer small rotations (5-45 degrees). "
        "If progress is not happening, include a lateral move first; "
        "rotate only if no lateral movement is safe. "
        "Do NOT use LookUp or LookDown. "
        "If uncertain, return an empty nav_subgoals list with low confidence. "
        "Do NOT repeat the prompt or context. Output JSON only."
    )
    ctx = json.dumps(context_summary, ensure_ascii=True)
    user = (
        f"TARGET: {target_label}\n"
        f"Context: {ctx}\n"
        "Return navigation subgoals that keep the target likely visible or re-acquirable."
    )
    return system, user


def build_assess_approach_messages(target_label: str, context_summary: Dict) -> Tuple[str, str]:
    """
    Costruisce il prompt per valutare approach.
    Usa target_label e context_summary.
    Ritorna una tupla (system, user).
    """
    system = (
        "You control an AI2-THOR robot at low level. "
        "Task: decide if it is feasible to approach the target directly from the current view. "
        "Return JSON only with keys: approach_possible (true/false), confidence (0-1), reason (short). "
        "Use the image and Context facts. If the target is behind a counter/wall or clearly blocked, "
        "set approach_possible=false. "
        "If oracle_target.distance_m is known but the path is obstructed, set approach_possible=false. "
        "Set approach_possible=true only when the front path is free and direct approach is feasible."
    )
    ctx = json.dumps(context_summary, ensure_ascii=True)
    user = (
        f"TARGET: {target_label}\n"
        f"Context: {ctx}\n"
        "Decide if the robot can approach the target directly without needing to navigate around obstacles."
    )
    return system, user


def build_predict_bbox_messages(target_label: str) -> Tuple[str, str]:
    """
    Costruisce il prompt per stimare la bbox.
    Include target_label e formato atteso.
    Ritorna una tupla (system, user).
    """
    system = (
        "Task: Bounding-box annotation. "
        f"You are given an RGB frame from AI2-THOR and a target object class: {target_label}. "
        "Provide a single bounding box tightly enclosing the target object. "
        "Output must be in pixel coordinates relative to the image: "
        "{x1, y1, x2, y2} with 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height. "
        "If the object is not visible, output NOT_VISIBLE."
    )
    user = f"Target: {target_label}"
    return system, user


def build_probe_scene_messages(target_label: str) -> Tuple[str, str]:
    """
    Costruisce il prompt per probe della scena.
    Definisce i campi JSON richiesti.
    Ritorna una tupla (system, user).
    """
    system = (
        "You are given a single RGB frame from AI2-THOR and a target object class. "
        "Return a compact structured assessment with focus on the target. "
        "Output only valid JSON, no extra text."
    )
    user = (
        f"TARGET_OBJECT: {target_label}\n"
        "Fields:\n"
        "target_visible: true/false\n"
        "target_visibility: one of [\"clear\",\"partial\",\"uncertain\",\"none\"]\n"
        "target_location_hint: one of [\"top_left\",\"top\",\"top_right\",\"left\",\"center\",\"right\","
        "\"bottom_left\",\"bottom\",\"bottom_right\",\"unknown\"]\n"
        "related_objects: array of up to 5 strings\n"
        "confidence: integer 0-100\n"
        "If uncertain, set target_visible=false and target_visibility=\"uncertain\"."
    )
    return system, user
