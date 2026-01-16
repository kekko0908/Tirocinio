# Scopo: template e prompt VLM centralizzati.
import json
from typing import Dict, List, Tuple

# --- SYSTEM CONSTANTS ---
SYSTEM_INSTRUCTIONS = (
    "YOU ARE A DUAL-VIEW ROBOT NAVIGATOR. "
    "You receive two images: (1) [EGO-VIEW] robot view, (2) [TOP-DOWN MAP] tactical map. "
    "On the top-down map: orange points are safe/reachable; everything else is lava/wall. "
    "A black arrow shows robot position and facing; a gold star (★) marks the target if present. "
    "If the robot is not on orange, prioritize returning to orange. "
)

# vlm.py

NAVIGATION_LOGIC = (
    "NAVIGATION LOGIC:\n"
    "1. FOLLOW GUIDANCE (MAP): Look for a WHITE LINE on the map. This is your calculated GPS path. "
    "Follow the white line strictly. If no white line, follow the ORANGE dots towards the STAR.\n"
    "Blue circles on the white line are CHECKPOINTS: use the nearest one as your current subgoal.\n"
    "2. SAFETY CHECK (EGO-VIEW): Before moving, verify with your eyes (Ego-View). "
    "Is the path physically clear? If you see an immediate obstacle (e.g., table edge) that the map missed, STOP or detour.\n"
    "3. TARGET ACQUISITION: If the Target Object is clearly visible in EGO-VIEW (not just as a Star on the map), "
    "ignore the white line and approach the target directly to center it."
)

JSON_FORMAT_CMD = "Output JSON only, no markdown, no extra text. "


def build_plan_subgoals_messages(goal_text: str) -> Tuple[str, str]:
    """
    Costruisce i messaggi di prompt per subgoal.
    Genera system e user in formato testo.
    Ritorna una tupla (system, user).
    """
    # Legacy: non usato dalla FSM corrente, tenuto per compatibilita.
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
        SYSTEM_INSTRUCTIONS
        + NAVIGATION_LOGIC
        + "Pick exactly one action from: "
        + ", ".join(action_set)
        + ". Return JSON only: "
        "{\"action\": \"...\", \"degrees\": optional int, \"reason\": \"short\", "
        "\"target_confidence\": 0.0-1.0, \"request_yolo\": true|false}. "
        + JSON_FORMAT_CMD
        + "Use the Context JSON as authoritative facts. "
        "Hard rule: NEVER MoveAhead if dist_front_m < safe_front_m. "
        "If front is clear and the target is far, prefer MoveAhead to reduce distance. "
        "If front is clear and oracle_target.distance_m is known, keep advancing even if the target is not visible. "
        "Do NOT rotate just to re-acquire a target that is already known from Context. "
        "If distance to target does not decrease for multiple steps, assume an obstacle and choose a lateral move or rotation. "
        "If Context.sensor.last_action_success is false, you are blocked; use the map to find open orange space. "
        "Safety: if dist_front_m < safe_front_m then do NOT MoveAhead; "
        "if dist_left_m < safe_front_m then do NOT MoveLeft; "
        "if dist_right_m < safe_front_m then do NOT MoveRight; "
        "if collision or last_action_success=false then prefer MoveLeft/MoveRight/MoveBack; "
        "rotate only if no side/back is safe. "
        "Use target_location_hint only when oracle_target.distance_m is unknown or the target is visible; "
        "then: left/right -> rotate toward hint; center -> move ahead if safe. "
        "If the map shows a clear orange corridor straight ahead and front is safe, prefer MoveAhead. "
        "Do NOT use LookUp. "
        "Only use LookDown when Context.flags.nav_obstacle_replan is true, and then use LookDown with degrees=20. "
        "State policy: EXPLORE = scan and widen coverage; SEARCH = scan and adjust to maximize recall; "
        "APPROACH = center target and reduce distance with small rotations; "
        "LOCALIZE = keep target stable and visible. "
        "Use MoveBack or small side-steps only when you need more space and it is safe. "
        "If oracle_target.distance_m is known and state is NAVIGATE, avoid request_yolo. "
        "Set request_yolo=true when the target is likely visible or probe is positive."
    )
    # history arriva da ActionManager.history (runner.py) ed e' una coda delle ultime azioni eseguite.
    # Lo usiamo solo come contesto testuale per evitare loop e per dare memoria breve alla VLM.
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
    # Legacy: non usato nella pipeline attuale (usa nav_subgoals).
    system = (
        SYSTEM_INSTRUCTIONS
        + NAVIGATION_LOGIC
        + "Task: generate a short navigation plan to get closer to the target. "
        "Return JSON only: {nav_plan, confidence, rationale, route_side}. "
        "nav_plan is 3-6 actions like {\"action\":\"...\",\"degrees\":optional int}. "
        "route_side in [\"left\",\"right\",\"straight\",\"unknown\"]. "
        "Keep the plan simple and prefer wide, open spaces. "
        "Avoid tight spaces; if the target is in a narrow area, first align/center with rotations or side steps, "
        "then advance. "
        "Use Context as facts. Safety: if dist_front_m < safe_front_m then no MoveAhead; "
        "if dist_left_m < safe_front_m then no MoveLeft; "
        "if dist_right_m < safe_front_m then no MoveRight. "
        "If Context.failed_actions_recent is non-empty, avoid repeating those actions. "
        "If Context.behavior_memory.avoid_actions is non-empty, avoid those actions. "
        "If Context.behavior_memory.avoid_zones is non-empty, avoid moving toward those zones. "
        "Prefer lateral moves before rotations when blocked. Prefer small rotations (5-45). "
        "If Context.flags.nav_obstacle_replan is true, the first subgoal must have id=\"subgoal-ostacolo\" "
        "and focus on avoiding the obstacle using the current view. "
        + JSON_FORMAT_CMD
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
        SYSTEM_INSTRUCTIONS
        + NAVIGATION_LOGIC
        + "Task: break navigation into 3-5 subgoals to get closer to the target. "
        "Return JSON only: {nav_subgoals, confidence, rationale, route_side}. "
        "Each subgoal has {id, goal, plan, expectation} and plan is 2-5 actions. "
        "Keep the plan simple and prefer wide, open spaces. "
        "Avoid tight spaces; if the target is in a narrow area, first align/center, then advance. "
        "Use Context as facts. Safety: if dist_front_m < safe_front_m then no MoveAhead; "
        "if dist_left_m < safe_front_m then no MoveLeft; "
        "if dist_right_m < safe_front_m then no MoveRight. "
        "If Context.failed_actions_recent is non-empty, avoid repeating those actions. "
        "If Context.behavior_memory.avoid_actions is non-empty, avoid those actions. "
        "If Context.behavior_memory.avoid_zones is non-empty, avoid moving toward those zones. "
        "Prefer lateral moves before rotations when blocked. Prefer small rotations (5-45). "
        "Do NOT return an empty nav_subgoals list. "
        "Example JSON: "
        "{\"nav_subgoals\":[{\"id\":\"1\",\"goal\":\"skirt obstacle\",\"expectation\":\"clear path\","
        "\"plan\":[{\"action\":\"MoveRight\"},{\"action\":\"MoveAhead\"}]}],"
        "\"confidence\":0.6,\"rationale\":\"simple detour\",\"route_side\":\"right\"}. "
        + JSON_FORMAT_CMD
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
