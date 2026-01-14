# Scopo: costruzione del context_summary per la VLM.
from typing import Any, Dict, Optional


def build_context_summary(
    args,
    total_steps: int,
    current_state: str,
    state_steps: int,
    target_type: str,
    target_idx: int,
    target_spec: Optional[Dict[str, Any]],
    nav_attempts: int,
    nav_need_plan: bool,
    last_nav_plan: list,
    sensor: Dict[str, Any],
    mem_summary: Dict[str, Any],
    probe_info: Optional[Dict[str, Any]],
    probe_positive: bool,
    probe_hint: Optional[str],
    last_detection: Optional[Dict[str, Any]],
    last_det_age: Optional[int],
    approach_active: bool,
    object_summary: Dict[str, Any],
    long_term_priors: Dict[str, Any],
    action_mgr,
    vlm_memory_summary: Optional[Dict[str, Any]] = None,
    oracle_target: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Costruisce il context_summary per la VLM.
    Aggrega stato FSM, sensori, memoria e probe.
    Ritorna un dict serializzabile per i prompt.
    """
    last_det_summary = None
    if last_detection:
        det = last_detection.get("det") or {}
        last_det_summary = {
            "score": det.get("score"),
            "centroid_px": det.get("centroid_px"),
            "bbox_xyxy": det.get("bbox_xyxy"),
            "world": (last_detection.get("world") or {}).get("position"),
            "world_source": (last_detection.get("world") or {}).get("source"),
            "dist_m": last_detection.get("dist_m"),
        }
    return {
        "episode_context": {
            "scene_id": args.scene,
            "step_id": total_steps,
            "current_state": current_state,
            "target": target_type,
            "target_index": target_idx,
        },
        "fsm": {
            "state": current_state,
            "state_steps": state_steps,
            "mode": target_spec.get("mode") if target_spec else "global",
            "reference": target_spec.get("reference") if target_spec else None,
            "nav_attempts": int(nav_attempts),
            "nav_need_plan": bool(nav_need_plan),
            "nav_plan_len": len(last_nav_plan),
        },
        "sensor": {
            "dist_front_m": round(sensor["dist_front_m"], 2),
            "dist_left_m": round(sensor["dist_left_m"], 2),
            "dist_right_m": round(sensor["dist_right_m"], 2),
            "collision": bool(sensor["collision"]),
            "last_action_success": bool(sensor["last_action_success"]),
            "pose_discrete": mem_summary.get("pose_discrete", {}),
        },
        "memory": {
            "visited": mem_summary["visited"],
            "coverage_pct": mem_summary["coverage_pct"],
            "ranked_directions": mem_summary["ranked_directions"],
            "novelty_score": mem_summary["novelty_score"],
        },
        "probe": {
            "info": probe_info,
            "positive": probe_positive,
            "hint": probe_hint,
        },
        "last_detection": last_det_summary,
        "last_detection_age": last_det_age,
        "approach_state": {
            "active": approach_active,
            "last_detection_step": last_detection["step"] if last_detection else None,
        },
        "object_memory": object_summary,
        "long_term_priors": long_term_priors,
        "action_manager": action_mgr.get_state(),
        "vlm_memory": vlm_memory_summary or {},
        "oracle_target": oracle_target,
        "constraints": {
            "safe_front_m": round(float(args.safe_front_m), 2),
            "safe_side_m": round(float(args.safe_front_m), 2),
            "navigate_dist_thresh_m": round(float(args.navigate_dist_thresh_m), 2),
            "approach_dist_thresh_m": round(float(args.approach_dist_thresh_m), 2),
        },
    }
