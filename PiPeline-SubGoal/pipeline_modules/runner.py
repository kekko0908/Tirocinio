# Scopo: orchestratore della pipeline (ciclo percezione/azione).
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common import EnvConfig, make_controller, get_rgb_bgr, print_ok, print_warn

from pipeline_modules.action_manager import ActionManager
from pipeline_modules.constants import ACTION_SET
from pipeline_modules.context import build_context_summary
from pipeline_modules.memory import ExplorationMemory
from pipeline_modules.sensors import build_sensor_state, estimate_target_world, save_depth_frame
from pipeline_modules.debug_logger import DebugLogger
from pipeline_modules.object_memory import ObjectMemory
from pipeline_modules.vlm_memory import VlmMemory
from pipeline_modules.planning import build_fsm_plan, build_target_queue
from pipeline_modules.utils import bbox_iou, format_action_spec, normalize_label, normalize_action, safe_write_json
from pipeline_modules.visualization import (
    annotate_telemetry,
    draw_comparison,
    draw_detection,
    draw_probe_overlay,
    draw_vlm_bbox,
    draw_vlm_not_visible,
    draw_yolo_evidence,
)
from pipeline_modules.vlm import VLMEngine
from pipeline_modules.yolo import YOLODetector
from pipeline_modules.oracle_autonav import oracle_approach_step
from fsm import (
    STATE_APPROACH,
    STATE_DESCRIPTIONS,
    STATE_DONE,
    STATE_EXPLORE,
    STATE_FAIL,
    STATE_LOCALIZE,
    STATE_NAVIGATE,
    STATE_SEARCH,
    STATE_SEARCH_NEAR,
    STATE_SELECT_TARGET,
    apply_explore_macros,
    build_nav_scan_queue,
    compute_approach_action,
    get_oracle_target,
    confirmed_from_hits,
    next_state_approach,
    next_state_explore,
    next_state_localize,
    next_state_navigate,
    next_state_search,
    next_state_search_near,
    parse_nav_plan,
    parse_nav_subgoals,
    reduce_hint_deg,
)
def run_pipeline(args) -> None:
    """
    Esegue l'intera pipeline percezione/azione.
    Imposta ambiente, moduli e ciclo FSM principale.
    Scrive log e risultati negli outputs.
    """

    if args.max_explore_steps is None:
        args.max_explore_steps = int(args.max_steps_per_subgoal)
    if args.max_search_steps is None:
        args.max_search_steps = int(args.max_steps_per_subgoal)
    if args.max_localize_steps is None:
        args.max_localize_steps = int(args.max_steps_per_subgoal)
    if args.near_max_steps is None:
        args.near_max_steps = int(args.max_steps_per_subgoal)

    if not args.goal and not args.target_type:
        raise SystemExit("Provide --goal or --target_type.")

    run_dir = PIPELINE_DIR / "outputs"
    frames_dir = run_dir / "frames"
    debug_dir = run_dir / "debug"
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    vlm = None
    if not args.no_vlm:
        try:
            vlm = VLMEngine(args.vlm_model, action_set=ACTION_SET)
        except Exception as e:
            print_warn(f"VLM load failed, fallback planner/action. Err={e}")
            vlm = None

    target_queue, near_relation = build_target_queue(args.goal or "", args.target_type or "")
    plan = build_fsm_plan(args.goal or "", target_queue, near_relation)

    safe_write_json(run_dir / "plan.json", plan)
    print_ok(f"Plan saved: {run_dir / 'plan.json'}")
    print_ok(f"Targets: {', '.join([t['label'] for t in target_queue])}")
    debug_logger = DebugLogger(run_dir / "debug_trace.jsonl", run_dir / "debug_trace.txt")
    subgoal_logger = DebugLogger(run_dir / "vlm_subgoals.jsonl")
    debug_logger.log(
        {
            "event": "start",
            "scene": args.scene,
            "targets": target_queue,
            "near_relation": near_relation,
            "args": vars(args),
        }
    )
    subgoal_logger.log(
        {
            "event": "start",
            "scene": args.scene,
            "targets": target_queue,
            "near_relation": near_relation,
            "args": vars(args),
        }
    )

    target_type = target_queue[0]["label"]
    detector = YOLODetector(args.yolo_model, args.yolo_conf, args.imgsz, target_type)

    cfg = EnvConfig(scene=args.scene, render_depth=True, render_instance_segmentation=True)
    controller = make_controller(cfg)

    total_steps = 0
    history = []
    log = []
    found_targets = []
    failed_targets = []
    last_yolo_step = -999
    pending_scan_yolo = False
    approach_confirm = 0
    last_detection = None
    episode_id = str(int(time.time()))
    object_memory = ObjectMemory(run_dir / "long_term_memory.json", scene_id=args.scene, episode_id=episode_id)
    vlm_memory = VlmMemory(run_dir / "memory_vlm.json", scene_id=args.scene, episode_id=episode_id)

    dir_bins = max(4, int(round(360 / float(args.scan_degrees))))
    memory = ExplorationMemory(cell_size=args.cell_size, dir_bins=dir_bins)
    action_mgr = ActionManager(
        safe_front_m=args.safe_front_m,
        scan_degrees=args.scan_degrees,
        scan_trigger=args.scan_trigger,
        scan_cooldown=args.scan_cooldown,
        advance_steps=args.advance_steps,
        advance_min_front=args.advance_min_front,
    )
    state = STATE_SELECT_TARGET
    state_steps = 0
    target_idx = -1
    target_spec = None
    near_anchor = None
    last_hint = {}
    hint_turn_deg = {}
    last_seen_hint = {}
    last_seen_step = {}
    nav_subgoals = deque()
    nav_subgoal_plan_queue = deque()
    nav_scan_queue = deque()
    nav_reacquire_queue = deque()
    nav_subgoal_idx = 0
    nav_last_subgoal = None
    nav_reacquire_pending_check = False
    nav_reacquire_side = None
    nav_reacquire_degrees = None
    nav_need_plan = True
    nav_attempts = 0
    last_nav_plan = []
    nav_yolo_ready_step = -1
    nav_no_progress_steps = 0
    nav_last_dist = None
    approach_stuck_steps = 0
    last_approach_dist = None
    approach_last_pos = None
    approach_stuck_counter = 0
    search_hits = deque(maxlen=int(args.search_confirm_n))
    localize_hits = deque(maxlen=int(args.localize_confirm_n))
    target_start_step = 0
    init_event = controller.last_event
    init_sensor = build_sensor_state(init_event, depth_radius=args.depth_radius)
    # Stampa posa iniziale per capire dove spawna il robot.
    init_pos = init_sensor.get("position", {}) or {}
    init_yaw = float(init_sensor.get("yaw", 0.0))
    init_horizon = float(init_sensor.get("horizon", 0.0))
    print_ok(
        f"Spawn pose | pos=({init_pos.get('x', 0.0):.2f}, {init_pos.get('y', 0.0):.2f}, {init_pos.get('z', 0.0):.2f}) "
        f"yaw={init_yaw:.1f} horizon={init_horizon:.1f}"
    )
    if args.spawn_yaw is not None:
        controller.step(action="RotateLeft", degrees=float(args.spawn_yaw))
        init_event = controller.last_event
        init_sensor = build_sensor_state(init_event, depth_radius=args.depth_radius)
        init_pos = init_sensor.get("position", {}) or {}
        init_yaw = float(init_sensor.get("yaw", 0.0))
        init_horizon = float(init_sensor.get("horizon", 0.0))
        print_ok(
            f"Spawn yaw applied | pos=({init_pos.get('x', 0.0):.2f}, {init_pos.get('y', 0.0):.2f}, {init_pos.get('z', 0.0):.2f}) "
            f"yaw={init_yaw:.1f} horizon={init_horizon:.1f}"
        )
    memory.update(init_sensor["position"], init_sensor["yaw"])
    init_mem = memory.summarize(init_sensor["position"], init_sensor["yaw"], init_sensor)
    if args.save_frames:
        init_bgr = get_rgb_bgr(init_event)
        init_frame = annotate_telemetry(init_bgr, 0, {"action": "INIT"}, init_sensor, init_mem)
        cv2.imwrite(str(frames_dir / "step_init_0000.png"), init_frame)
        if args.save_depth:
            save_depth_frame(getattr(init_event, "depth_frame", None), frames_dir / "depth_init_0000.png")

    def enter_state(new_state: str, reason: str) -> None:
        """
        Gestisce la transizione fra stati FSM.
        Resetta contatori e code secondo lo stato.
        Registra log e stampa il motivo del cambio.
        """
        nonlocal state, state_steps, nav_need_plan, nav_yolo_ready_step, approach_stuck_steps, last_approach_dist
        nonlocal approach_last_pos, approach_stuck_counter
        nonlocal nav_subgoals, nav_subgoal_plan_queue, nav_scan_queue, nav_reacquire_queue
        nonlocal nav_subgoal_idx, nav_last_subgoal, nav_reacquire_pending_check, nav_reacquire_side
        nonlocal nav_reacquire_degrees
        nonlocal nav_no_progress_steps, nav_last_dist
        if new_state == state:
            return
        prev = state
        state = new_state
        state_steps = 0
        if new_state in {STATE_APPROACH, STATE_LOCALIZE}:
            action_mgr.queue = []
            action_mgr.scan_remaining = 0
        if new_state in {STATE_SEARCH, STATE_SEARCH_NEAR}:
            search_hits.clear()
        if new_state == STATE_LOCALIZE:
            localize_hits.clear()
        if new_state == STATE_APPROACH:
            approach_stuck_steps = 0
            last_approach_dist = None
            approach_last_pos = None
            approach_stuck_counter = 0
        if new_state == STATE_NAVIGATE:
            nav_subgoals.clear()
            nav_subgoal_plan_queue.clear()
            nav_scan_queue.clear()
            nav_reacquire_queue.clear()
            nav_need_plan = True
            nav_yolo_ready_step = -1
            approach_stuck_steps = 0
            nav_subgoal_idx = 0
            nav_last_subgoal = None
            nav_reacquire_pending_check = False
            nav_reacquire_side = None
            nav_reacquire_degrees = None
            nav_no_progress_steps = 0
            nav_last_dist = None
        debug_logger.log(
            {"event": "state_transition", "from": prev, "to": new_state, "reason": reason, "step": total_steps}
        )
        print_ok(f"FSM {prev} -> {new_state} | motivo: {reason}")

    def select_next_target() -> bool:
        """
        Seleziona il prossimo target dalla coda.
        Resetta variabili e contatori per il nuovo target.
        Ritorna True se un target e disponibile.
        """
        # Seleziona il prossimo target e resetta lo stato operativo.
        nonlocal target_idx, target_spec, target_type, last_detection, approach_confirm, pending_scan_yolo, near_anchor
        nonlocal nav_need_plan, nav_attempts, last_nav_plan, nav_yolo_ready_step
        nonlocal nav_subgoals, nav_subgoal_plan_queue, nav_scan_queue, nav_reacquire_queue
        nonlocal nav_subgoal_idx, nav_last_subgoal, nav_reacquire_pending_check, nav_reacquire_side
        nonlocal nav_reacquire_degrees
        nonlocal nav_no_progress_steps, nav_last_dist
        nonlocal approach_stuck_steps, last_approach_dist, approach_last_pos, approach_stuck_counter
        nonlocal target_start_step
        target_idx += 1
        if target_idx >= len(target_queue):
            return False
        target_spec = target_queue[target_idx]
        target_type = target_spec["label"]
        last_hint.pop(target_type, None)
        last_seen_hint.pop(target_type, None)
        last_seen_step.pop(target_type, None)
        hint_turn_deg.pop(target_type, None)
        nav_subgoals.clear()
        nav_subgoal_plan_queue.clear()
        nav_scan_queue.clear()
        nav_reacquire_queue.clear()
        nav_need_plan = True
        nav_attempts = 0
        last_nav_plan = []
        nav_yolo_ready_step = -1
        nav_subgoal_idx = 0
        nav_last_subgoal = None
        nav_reacquire_pending_check = False
        nav_reacquire_side = None
        nav_reacquire_degrees = None
        nav_no_progress_steps = 0
        nav_last_dist = None
        approach_stuck_steps = 0
        last_approach_dist = None
        approach_last_pos = None
        approach_stuck_counter = 0
        detector.target_label = normalize_label(target_type)
        last_detection = None
        approach_confirm = 0
        pending_scan_yolo = False
        near_anchor = None
        action_mgr.history = []
        target_start_step = total_steps
        if target_spec.get("mode") == "near":
            near_anchor = {"label": target_spec.get("reference"), "position": None}
            enter_state(STATE_SEARCH_NEAR, "start_target_near")
        else:
            enter_state(STATE_EXPLORE, "start_target")
        debug_logger.log(
            {"event": "target_selected", "target": target_type, "spec": target_spec, "step": total_steps}
        )
        return True


    def try_yolo(frame_rgb, frame_bgr, reason: str):
        """
        Esegue una detection YOLO rispettando il cooldown.
        Salva evidenze e aggiorna file di output.
        Ritorna il risultato della detection o None.
        """
        nonlocal last_yolo_step
        if total_steps - last_yolo_step < int(args.yolo_cooldown):
            return None
        det, mask01 = detector.detect(frame_rgb)
        last_yolo_step = total_steps
        if det is None:
            return None
        if det["score"] < float(args.yolo_conf):
            return None
        evidence_frame = draw_yolo_evidence(frame_bgr, det, draw_centroid=True)
        evidence_name = f"yolo_evidence_step_{total_steps:04d}.png"
        evidence_path = frames_dir / evidence_name
        cv2.imwrite(str(evidence_path), evidence_frame)
        det_out = {
            "goal": args.goal,
            "scene": args.scene,
            "target_type": target_type,
            "step": total_steps,
            "time": time.time(),
            "reason": reason,
            "detection": det,
            "detections": [
                {
                    "class": det["label"],
                    "bbox": det["bbox_xyxy"],
                    "confidence": det["score"],
                    "centroid": det["centroid_px"],
                }
            ],
            "yolo_bbox_frame": str(evidence_path),
            "yolo_mask_frame": str(frames_dir / f"yolo_mask_step_{total_steps:04d}.png"),
        }
        safe_write_json(run_dir / "detection.json", det_out)
        vis = draw_detection(frame_bgr, det, mask01=mask01)
        cv2.imwrite(str(debug_dir / f"det_step_{total_steps:04d}.png"), vis)
        cv2.imwrite(str(frames_dir / f"yolo_mask_step_{total_steps:04d}.png"), vis)
        if vlm is not None and not args.no_vlm_bbox:
            img = Image.fromarray(frame_rgb)
            vlm_bbox, vlm_raw = vlm.predict_bbox(img, target_type)
            det_out["vlm_bbox"] = vlm_bbox if vlm_bbox else "NOT_VISIBLE"
            det_out["vlm_raw"] = vlm_raw
            if vlm_bbox:
                iou = bbox_iou(det["bbox_xyxy"], vlm_bbox)
                det_out["iou_score"] = round(float(iou), 4)
                det_out["test_result"] = "PASS" if iou >= float(args.iou_threshold) else "FAIL"
                vlm_frame = draw_vlm_bbox(frame_bgr, vlm_bbox, label="VLM")
                vlm_name = f"vlm_bbox_step_{total_steps:04d}.png"
                vlm_path = frames_dir / vlm_name
                cv2.imwrite(str(vlm_path), vlm_frame)
                det_out["vlm_bbox_frame"] = str(vlm_path)
                comp = draw_comparison(frame_bgr, det, vlm_bbox)
                comp_name = f"compare_step_{total_steps:04d}.png"
                comp_path = debug_dir / comp_name
                cv2.imwrite(str(comp_path), comp)
                det_out["comparison_frame"] = str(comp_path)
            else:
                vlm_frame = draw_vlm_not_visible(frame_bgr, message="VLM NOT_VISIBLE")
                vlm_name = f"vlm_bbox_step_{total_steps:04d}.png"
                vlm_path = frames_dir / vlm_name
                cv2.imwrite(str(vlm_path), vlm_frame)
                det_out["vlm_bbox_frame"] = str(vlm_path)
                det_out["iou_score"] = 0.0
                det_out["test_result"] = "FAIL"
            safe_write_json(run_dir / "detection.json", det_out)
        print_ok("Target localized with YOLO.")
        return {"det": det, "mask": mask01, "state": det_out}

    def compute_horizontal_hint(centroid_px, frame_width: int) -> Optional[str]:
        """
        Calcola un hint left/center/right dal centroid.
        Usa soglie relative alla larghezza del frame.
        Ritorna la stringa hint o None se non valido.
        """
        if not centroid_px or frame_width <= 0:
            return None
        try:
            cx = float(centroid_px[0])
        except Exception:
            return None
        if cx < 0.4 * frame_width:
            return "left"
        if cx > 0.6 * frame_width:
            return "right"
        return "center"

    def build_reacquire_queue(hint: Optional[str], degrees: int) -> deque:
        """
        Costruisce una coda per riacquisire il target.
        Preferisce la rotazione verso l'ultimo hint.
        Fallback su micro-scan se hint assente.
        """
        # Dopo un subgoal, ruoto verso l'ultimo lato visto per riacquisire il target.
        deg = int(max(5, min(45, degrees)))
        if hint == "left":
            return deque([{"action": "RotateLeft", "degrees": deg}])
        if hint == "right":
            return deque([{"action": "RotateRight", "degrees": deg}])
        if hint == "center":
            return deque()
        return build_nav_scan_queue(deg)

    def build_nav_fallback_queue(mem_summary: Dict, sensor: Dict) -> deque:
        """
        Costruisce un piano di navigazione fallback.
        Usa memoria e distanze per scegliere movimenti.
        Ritorna una deque di action spec.
        """
        # Fallback NAVIGATE: usa direzione piu' libera/nuova e avanza se sicuro.
        ranked = (mem_summary or {}).get("ranked_directions") or []
        pose = (mem_summary or {}).get("pose_discrete") or {}
        current_dir = int(pose.get("dir_idx", 0))
        safe_front = float(args.safe_front_m)
        dist_front = float(sensor.get("dist_front_m", 0.0))
        dist_left = float(sensor.get("dist_left_m", 0.0))
        dist_right = float(sensor.get("dist_right_m", 0.0))

        if dist_front < safe_front:
            # Se davanti e' stretto, ruoto verso il lato piu' libero.
            rot_action = "RotateLeft" if dist_left >= dist_right else "RotateRight"
            return deque([{"action": rot_action, "degrees": int(args.nav_scan_degrees)}])

        if not ranked:
            return deque([{"action": "MoveAhead"}])

        target_dir = int(ranked[0].get("dir_idx", current_dir))
        diff = (target_dir - current_dir) % dir_bins
        step_deg = int(round(360 / float(dir_bins)))
        if diff == 0:
            return deque([{"action": "MoveAhead"}])
        if diff <= dir_bins / 2:
            deg = int(min(90, max(5, diff * step_deg)))
            return deque([{"action": "RotateRight", "degrees": deg}, {"action": "MoveAhead"}])
        deg = int(min(90, max(5, (dir_bins - diff) * step_deg)))
        return deque([{"action": "RotateLeft", "degrees": deg}, {"action": "MoveAhead"}])

    def get_oracle_nav_summary(event, target_type: str) -> Optional[Dict[str, Any]]:
        """
        Estrae un riepilogo di navigazione dai metadata.
        Usa distanza e posizione anche se non visibile.
        Ritorna dict con info target o None.
        """
        # Oracle "soft": usa metadata per distanza/posizione anche se non visibile.
        meta = getattr(event, "metadata", {}) or {}
        objs = meta.get("objects", []) or []
        target_l = str(target_type or "").lower()
        candidates = [o for o in objs if str(o.get("objectType", "")).lower() == target_l]
        if not candidates:
            return None
        obj = sorted(candidates, key=lambda x: float(x.get("distance", 1e9)))[0]
        dist = None
        try:
            dist = float(obj.get("distance"))
        except Exception:
            dist = None
        return {
            "visible": bool(obj.get("visible")),
            "distance_m": dist,
            "position": obj.get("position"),
            "object_id": obj.get("objectId"),
        }

    def update_last_seen(target_type: str, hint: Optional[str], source: str, centroid_px: Optional[list] = None) -> None:
        """
        Aggiorna l'ultimo hint visto con debounce.
        Evita cambi repentini di lato in 1 step.
        Propaga le info alla memoria VLM.
        """
        # Debounce: evita flip left/right immediati.
        if not hint:
            return
        prev = last_seen_hint.get(target_type)
        prev_step = last_seen_step.get(target_type)
        if prev and hint != prev and prev_step is not None:
            if int(total_steps) - int(prev_step) <= 1:
                return
        last_seen_hint[target_type] = hint
        last_seen_step[target_type] = int(total_steps)
        vlm_memory.update_last_seen(target_type, hint, total_steps, source, centroid_px=centroid_px)

    def handle_detection(event, yolo_res):
        """
        Gestisce una detection valida e aggiorna le memorie.
        Stima posizione mondo, salva record e aggiorna hint.
        Aggiorna last_detection e logga l'evento.
        """
        nonlocal last_detection, approach_confirm
        det = yolo_res["det"]
        centroid = det["centroid_px"]
        world = estimate_target_world(event, target_type, centroid, depth_radius=args.depth_radius)
        world_source = None
        dist_m = None
        if world:
            world_source = world.get("source", "unknown")
            pos = (event.metadata.get("agent") or {}).get("position", {}) or {}
            dx = float(world["position"].get("x", 0.0)) - float(pos.get("x", 0.0))
            dz = float(world["position"].get("z", 0.0)) - float(pos.get("z", 0.0))
            dist_m = (dx * dx + dz * dz) ** 0.5
            object_memory.record_detection(
                target_type,
                world["position"],
                total_steps,
                det["score"],
                centroid_px=centroid,
                source=world_source,
            )
            yolo_res["state"]["world_position"] = world["position"]
            yolo_res["state"]["world_source"] = world_source
            safe_write_json(run_dir / "detection.json", yolo_res["state"])
        else:
            object_memory.record_detection(
                target_type,
                None,
                total_steps,
                det["score"],
                centroid_px=centroid,
                source="none",
            )

        last_detection = {
            "det": det,
            "world": world,
            "step": total_steps,
            "dist_m": dist_m,
            "source": world_source,
        }
        frame_w = int(event.frame.shape[1]) if hasattr(event, "frame") else 0
        hint = compute_horizontal_hint(centroid, frame_w)
        if hint:
            update_last_seen(target_type, hint, "yolo", centroid_px=list(centroid))
        approach_confirm = 0
        debug_logger.log(
            {
                "event": "detection",
                "step": total_steps,
                "target": target_type,
                "score": det["score"],
                "centroid_px": centroid,
                "world": world,
            }
        )

    try:
        while total_steps < int(args.max_steps):
            if state == STATE_SELECT_TARGET:
                if not select_next_target():
                    enter_state(STATE_DONE, "all_targets_done")
                    break

            event = controller.last_event
            frame_rgb = event.frame
            frame_bgr = get_rgb_bgr(event)
            sensor = build_sensor_state(event, depth_radius=args.depth_radius)
            mem_summary = memory.summarize(sensor["position"], sensor["yaw"], sensor)
            approach_active = state == STATE_APPROACH
            current_state = state
            state_desc = STATE_DESCRIPTIONS.get(current_state, "Search the target.")
            object_summary = object_memory.get_summary(target_type)
            long_term_priors = object_memory.get_long_term_priors(target_type)
            use_oracle_nav = bool(args.oracle_approach)
            oracle_nav_summary = get_oracle_nav_summary(event, target_type) if use_oracle_nav else None

            anchor_dist = None
            anchor_missing = False
            if current_state == STATE_SEARCH_NEAR and near_anchor and near_anchor.get("label"):
                anchor_info = object_memory.get_summary(near_anchor["label"]).get("target_last_seen")
                near_anchor["position"] = (anchor_info or {}).get("position")
                if near_anchor["position"]:
                    pos = sensor.get("position", {}) or {}
                    dx = float(near_anchor["position"].get("x", 0.0)) - float(pos.get("x", 0.0))
                    dz = float(near_anchor["position"].get("z", 0.0)) - float(pos.get("z", 0.0))
                    anchor_dist = (dx * dx + dz * dz) ** 0.5
                else:
                    anchor_missing = True

            probe_info = None
            probe_raw = ""
            probe_positive = False
            probe_hint = None
            hint_action = None
            probe_conf = 0
            if (
                vlm is not None
                and current_state != STATE_APPROACH
                and int(args.probe_every) > 0
                and total_steps % int(args.probe_every) == 0
            ):
                img = Image.fromarray(frame_rgb)
                probe_info, probe_raw = vlm.probe_scene(img, target_type)
                if isinstance(probe_info, dict):
                    probe_hint = str(probe_info.get("target_location_hint", "") or "").lower()
                    probe_hint = probe_hint.replace("-", "_").replace(" ", "_")
                    try:
                        conf = int(probe_info.get("confidence", 0) or 0)
                    except Exception:
                        conf = 0
                    probe_conf = conf
                    visible = bool(probe_info.get("target_visible", False))
                    probe_positive = visible or conf >= int(args.probe_conf_thresh)
                    if probe_positive:
                        # Se ho un hint direzionale, lo uso per orientarmi invece di fare scan cieco.
                        strong_hint = conf >= int(args.hint_strong_conf)
                        left_hints = {"left", "top_left", "bottom_left"}
                        right_hints = {"right", "top_right", "bottom_right"}
                        bottom_hints = {"bottom", "bottom_left", "bottom_right"}
                        horizontal_hint = None
                        if probe_hint in left_hints:
                            horizontal_hint = "left"
                        elif probe_hint in right_hints:
                            horizontal_hint = "right"
                        if probe_hint == "center":
                            horizontal_hint = "center"

                        if horizontal_hint:
                            update_last_seen(target_type, horizontal_hint, "probe")

                        if strong_hint and action_mgr.queue:
                            action_mgr.queue = []
                            action_mgr.scan_remaining = 0

                        if probe_hint == "center" and current_state in {STATE_EXPLORE, STATE_SEARCH, STATE_APPROACH}:
                            last_hint[target_type] = "center"
                            hint_turn_deg[target_type] = int(args.probe_scan_degrees)
                            if float(sensor.get("dist_front_m", 0.0)) >= float(args.safe_front_m):
                                hint_action = {"action": "MoveAhead"}
                        elif probe_hint in bottom_hints and current_state in {STATE_SEARCH, STATE_APPROACH}:
                            back_est = max(
                                float(sensor.get("dist_left_m", 0.0)),
                                float(sensor.get("dist_right_m", 0.0)),
                            )
                            if back_est >= float(args.safe_front_m):
                                hint_action = {"action": "MoveBack"}
                            elif horizontal_hint:
                                hint_action = {
                                    "action": "RotateLeft" if horizontal_hint == "left" else "RotateRight",
                                    "degrees": int(args.approach_rotate_degrees),
                                }
                        elif horizontal_hint and current_state in {STATE_EXPLORE, STATE_SEARCH, STATE_APPROACH}:
                            base_deg = (
                                int(args.approach_rotate_degrees)
                                if current_state == STATE_APPROACH
                                else int(args.probe_scan_degrees)
                            )
                            prev_hint = last_hint.get(target_type)
                            current_deg = hint_turn_deg.get(target_type, base_deg)
                            if prev_hint in {"left", "right"} and prev_hint != horizontal_hint:
                                current_deg = reduce_hint_deg(current_deg, base_deg)
                            hint_turn_deg[target_type] = current_deg
                            last_hint[target_type] = horizontal_hint
                            hint_action = {
                                "action": "RotateLeft" if horizontal_hint == "left" else "RotateRight",
                                "degrees": current_deg,
                            }
                        elif current_state == STATE_EXPLORE and not strong_hint:
                            action_mgr.enqueue_probe_scan(
                                steps=int(args.probe_scan_steps),
                                degrees=int(args.probe_scan_degrees),
                                step=total_steps,
                            )
                if args.save_frames and isinstance(probe_info, dict):
                    probe_frame = draw_probe_overlay(frame_bgr, target_type, probe_info)
                    cv2.imwrite(str(frames_dir / f"probe_step_{total_steps:04d}.png"), probe_frame)

            # Fallback: se non abbiamo un hint recente dal probe, usiamo l'ultimo lato visto.
            if hint_action is None:
                last_hint_side = last_seen_hint.get(target_type)
                last_step = last_seen_step.get(target_type)
                if last_hint_side and last_step is not None:
                    age = int(total_steps) - int(last_step)
                    if age <= int(args.lost_target_frames):
                        turn_deg = (
                            int(args.approach_rotate_degrees)
                            if current_state == STATE_APPROACH
                            else int(args.probe_scan_degrees)
                        )
                        if last_hint_side == "center":
                            if float(sensor.get("dist_front_m", 0.0)) >= float(args.safe_front_m):
                                last_seen_action = {"action": "MoveAhead"}
                        elif last_hint_side in {"left", "right"}:
                            last_seen_action = {
                                "action": "RotateLeft" if last_hint_side == "left" else "RotateRight",
                                "degrees": int(turn_deg),
                            }

            if current_state == STATE_EXPLORE:
                apply_explore_macros(
                    action_mgr,
                    mem_summary,
                    sensor,
                    total_steps,
                    probe_positive,
                    probe_conf,
                    args.hint_strong_conf,
                )

            yolo_res = None
            if pending_scan_yolo and total_steps - last_yolo_step >= int(args.yolo_cooldown):
                pending_scan_yolo = False
                yolo_res = try_yolo(frame_rgb, frame_bgr, "scan_complete")
                if yolo_res:
                    handle_detection(event, yolo_res)
                    object_summary = object_memory.get_summary(target_type)
                    long_term_priors = object_memory.get_long_term_priors(target_type)

            action = None
            action_data = None
            vlm_action_spec = None
            nav_action_spec = None
            nav_reacquire_action = False
            last_seen_action = None
            vlm_conf = 0.0
            vlm_request = False
            vlm_reason = ""
            context_summary = None

            if vlm is not None and current_state != STATE_APPROACH:
                last_det_age = None
                if last_detection:
                    last_det_age = total_steps - int(last_detection.get("step", 0))
                context_summary = build_context_summary(
                    args,
                    total_steps,
                    current_state,
                    state_steps,
                    target_type,
                    target_idx,
                    target_spec,
                    nav_attempts,
                    nav_need_plan,
                    last_nav_plan,
                    sensor,
                    mem_summary,
                    probe_info,
                    probe_positive,
                    probe_hint,
                    last_detection,
                    last_det_age,
                    approach_active,
                    object_summary,
                    long_term_priors,
                    action_mgr,
                    vlm_memory.get_summary(target_type),
                    oracle_nav_summary,
                )
                img = Image.fromarray(frame_rgb)
                if current_state == STATE_NAVIGATE:
                    if nav_reacquire_pending_check:
                        if not sensor.get("last_action_success", True):
                            back_clear = max(
                                float(sensor.get("dist_left_m", 0.0)),
                                float(sensor.get("dist_right_m", 0.0)),
                            ) >= float(args.safe_front_m)
                            if back_clear and nav_reacquire_side:
                                retry_deg = nav_reacquire_degrees or int(args.probe_scan_degrees)
                                nav_reacquire_queue.appendleft(
                                    {
                                        "action": "RotateLeft" if nav_reacquire_side == "left" else "RotateRight",
                                        "degrees": int(retry_deg),
                                    }
                                )
                                nav_reacquire_queue.appendleft({"action": "MoveBack"})
                                print_warn("NAV: rotazione bloccata, faccio MoveBack e riprovo.")
                            else:
                                print_warn("NAV: rotazione bloccata e spazio dietro insufficiente.")
                        nav_reacquire_pending_check = False

                    if (
                        nav_need_plan
                        and not nav_subgoal_plan_queue
                        and not nav_subgoals
                        and not nav_reacquire_queue
                        and not nav_scan_queue
                    ):
                        nav_data = vlm.plan_navigation_subgoals(img, target_type, context_summary)
                        nav_rationale = ""
                        nav_route = "unknown"
                        nav_confidence = None
                        if isinstance(nav_data, dict):
                            nav_rationale = str(nav_data.get("rationale", "") or "").strip()
                            nav_route = str(nav_data.get("route_side", "unknown") or "unknown").strip().lower()
                            if nav_route not in {"left", "right", "straight", "unknown"}:
                                nav_route = "unknown"
                            try:
                                nav_confidence = float(nav_data.get("confidence", 0.0) or 0.0)
                            except Exception:
                                nav_confidence = None
                        subgoals = parse_nav_subgoals(nav_data, int(args.nav_subgoals_max), args.nav_plan_max_steps)
                        nav_attempts += 1
                        nav_need_plan = False
                        nav_subgoals.extend(subgoals)
                        last_nav_plan = list(subgoals[0]["plan"]) if subgoals else []
                        if nav_rationale or nav_route != "unknown":
                            vlm_memory.update_nav_summary(target_type, nav_route, nav_rationale)
                        if subgoals:
                            rationale_txt = nav_rationale[:180] if nav_rationale else "-"
                            conf_txt = f"{nav_confidence:.2f}" if nav_confidence is not None else "-"
                            print_ok(
                                f"[VLM] NAV_GOAL#{nav_attempts} | T={target_type} | side={nav_route} | "
                                f"conf={conf_txt} | count={len(subgoals)} | rationale={rationale_txt}"
                            )
                            for sg in subgoals:
                                sg_id = sg.get("id") or "?"
                                sg_goal = sg.get("goal") or "-"
                                sg_expect = sg.get("expectation") or "-"
                                sg_steps = len(sg.get("plan") or [])
                                print_ok(f"[VLM]  - SG{sg_id}: goal={sg_goal} | steps={sg_steps} | expect={sg_expect}")
                            debug_logger.log(
                                {
                                    "event": "vlm_nav_goal",
                                    "step": total_steps,
                                    "target": target_type,
                                    "nav_attempt": nav_attempts,
                                    "nav_subgoals": subgoals,
                                    "route_side": nav_route,
                                    "rationale": nav_rationale,
                                    "confidence": nav_confidence,
                                    "raw": nav_data,
                                }
                            )
                            subgoal_logger.log(
                                {
                                    "event": "nav_goal_plan",
                                    "step": total_steps,
                                    "target": target_type,
                                    "nav_attempt": nav_attempts,
                                    "route_side": nav_route,
                                    "rationale": nav_rationale,
                                    "confidence": nav_confidence,
                                    "subgoals": subgoals,
                                }
                            )
                        else:
                            fallback_plan = parse_nav_plan(nav_data, args.nav_plan_max_steps)
                            if fallback_plan:
                                nav_subgoals.append(
                                    {
                                        "id": "fallback_plan",
                                        "goal": "fallback nav_plan",
                                        "expectation": "",
                                        "plan": fallback_plan,
                                    }
                                )
                            else:
                                print_warn("[VLM] NAV_GOAL empty -> fallback scan")
                                nav_scan_queue = build_nav_fallback_queue(mem_summary, sensor)
                                debug_logger.log(
                                    {
                                        "event": "vlm_nav_goal_empty",
                                        "step": total_steps,
                                        "target": target_type,
                                        "nav_attempt": nav_attempts,
                                        "raw": nav_data,
                                    }
                                )
                                subgoal_logger.log(
                                    {
                                        "event": "nav_goal_empty",
                                        "step": total_steps,
                                        "target": target_type,
                                        "nav_attempt": nav_attempts,
                                        "raw": nav_data,
                                    }
                                )
                                subgoal_logger.log(
                                    {
                                        "event": "nav_goal_fallback",
                                        "step": total_steps,
                                        "target": target_type,
                                        "actions": list(nav_scan_queue),
                                    }
                                )

                    if nav_subgoal_plan_queue:
                        nav_action_spec = nav_subgoal_plan_queue.popleft()
                        if not nav_subgoal_plan_queue:
                            done_goal = (nav_last_subgoal or {}).get("goal") or "-"
                            done_expect = (nav_last_subgoal or {}).get("expectation") or "-"
                            done_id = (nav_last_subgoal or {}).get("id") or nav_subgoal_idx
                            print_ok(
                                f"[VLM] NAV_GOAL_SUBGOAL_DONE#{nav_subgoal_idx} (id={done_id}) | T={target_type} | "
                                f"goal={done_goal}"
                            )
                            subgoal_logger.log(
                                {
                                    "event": "nav_goal_subgoal_done",
                                    "step": total_steps,
                                    "target": target_type,
                                    "nav_index": nav_subgoal_idx,
                                    "id": done_id,
                                    "goal": done_goal,
                                    "expectation": done_expect,
                                }
                            )
                            if use_oracle_nav and oracle_nav_summary and oracle_nav_summary.get("distance_m") is not None:
                                if not nav_subgoals and not nav_subgoal_plan_queue:
                                    nav_need_plan = True
                            else:
                                nav_reacquire_queue = build_reacquire_queue(
                                    last_seen_hint.get(target_type),
                                    int(args.probe_scan_degrees),
                                )
                    elif nav_reacquire_queue:
                        nav_action_spec = nav_reacquire_queue.popleft()
                        nav_reacquire_action = True
                        if not nav_reacquire_queue:
                            if not use_oracle_nav:
                                nav_yolo_ready_step = int(total_steps) + 1
                            if not nav_subgoals and not nav_subgoal_plan_queue:
                                nav_need_plan = True
                        subgoal_logger.log(
                            {
                                "event": "nav_goal_subgoal_reacquire",
                                "step": total_steps,
                                "target": target_type,
                                "last_seen_hint": last_seen_hint.get(target_type),
                                "action": nav_action_spec,
                            }
                        )
                    elif nav_subgoals:
                        subgoal = nav_subgoals.popleft()
                        nav_last_subgoal = subgoal
                        nav_subgoal_idx += 1
                        nav_subgoal_plan_queue = deque(subgoal.get("plan") or [])
                        plan_txt = ", ".join([format_action_spec(s) for s in nav_subgoal_plan_queue]) or "-"
                        goal_txt = subgoal.get("goal") or "-"
                        expect_txt = subgoal.get("expectation") or "-"
                        subgoal_id = subgoal.get("id") or nav_subgoal_idx
                        print_ok(
                            f"[VLM] NAV_GOAL_SUBGOAL#{nav_subgoal_idx} (id={subgoal_id}) | T={target_type} | "
                            f"goal={goal_txt} | expect={expect_txt} | plan={plan_txt}"
                        )
                        debug_logger.log(
                            {
                                "event": "vlm_nav_goal_subgoal",
                                "step": total_steps,
                                "target": target_type,
                                "nav_index": nav_subgoal_idx,
                                "goal": goal_txt,
                                "expectation": expect_txt,
                                "plan": list(nav_subgoal_plan_queue),
                            }
                        )
                        subgoal_logger.log(
                            {
                                "event": "nav_goal_subgoal_start",
                                "step": total_steps,
                                "target": target_type,
                                "nav_index": nav_subgoal_idx,
                                "goal": goal_txt,
                                "expectation": expect_txt,
                                "plan": list(nav_subgoal_plan_queue),
                            }
                        )
                        if not nav_subgoal_plan_queue:
                            if use_oracle_nav and oracle_nav_summary and oracle_nav_summary.get("distance_m") is not None:
                                if not nav_subgoals:
                                    nav_need_plan = True
                            else:
                                nav_reacquire_queue = build_reacquire_queue(
                                    last_seen_hint.get(target_type),
                                    int(args.probe_scan_degrees),
                                )
                                if not nav_reacquire_queue and not use_oracle_nav:
                                    nav_yolo_ready_step = int(total_steps) + 1
                    elif nav_scan_queue:
                        nav_action_spec = nav_scan_queue.popleft()
                        if not nav_scan_queue:
                            if not use_oracle_nav:
                                nav_yolo_ready_step = int(total_steps) + 1
                            nav_need_plan = True

                    if nav_action_spec is None:
                        action_data = vlm.choose_action(img, state_desc, history, context_summary)
                else:
                    action_data = vlm.choose_action(img, state_desc, history, context_summary)

                if isinstance(action_data, dict):
                    action = normalize_action(action_data.get("action"))
                    degrees = action_data.get("degrees")
                    if action in {"RotateLeft", "RotateRight"} and degrees is not None:
                        try:
                            deg_val = int(float(degrees))
                        except Exception:
                            deg_val = None
                        if deg_val is not None:
                            deg_val = max(1, min(90, deg_val))
                            vlm_action_spec = {"action": action, "degrees": deg_val}
                        else:
                            vlm_action_spec = action
                    else:
                        vlm_action_spec = action
                    try:
                        vlm_conf = float(action_data.get("target_confidence", 0.0) or 0.0)
                    except Exception:
                        vlm_conf = 0.0
                    vlm_conf = max(0.0, min(1.0, vlm_conf))
                    vlm_request = bool(action_data.get("request_yolo", False))
                    vlm_reason = str(action_data.get("reason", ""))[:120]

            if (vlm_action_spec is not None or nav_action_spec is not None) and action_mgr.queue:
                # Se la VLM propone un'azione, evito che le macro la sovrascrivano.
                action_mgr.queue = []
                action_mgr.scan_remaining = 0

            yolo_reason = None
            nav_yolo_ready = (
                current_state == STATE_NAVIGATE
                and not use_oracle_nav
                and nav_yolo_ready_step >= 0
                and total_steps >= int(nav_yolo_ready_step)
            )
            if current_state != STATE_APPROACH:
                if nav_yolo_ready:
                    yolo_reason = "nav_reacquire"
                elif vlm_request:
                    yolo_reason = "vlm_request"
                elif vlm_conf >= float(args.target_conf_thresh):
                    yolo_reason = f"vlm_conf_{vlm_conf:.2f}"
                elif probe_positive:
                    yolo_reason = "probe_positive"
                elif current_state == STATE_LOCALIZE:
                    yolo_reason = "state_localize"
                elif current_state in {STATE_SEARCH, STATE_SEARCH_NEAR}:
                    yolo_reason = "state_search"
                elif int(args.yolo_every) > 0 and total_steps % int(args.yolo_every) == 0:
                    yolo_reason = "periodic"

            if yolo_res is None and yolo_reason and total_steps - last_yolo_step >= int(args.yolo_cooldown):
                yolo_res = try_yolo(frame_rgb, frame_bgr, yolo_reason)
                if yolo_res:
                    handle_detection(event, yolo_res)
                    object_summary = object_memory.get_summary(target_type)
                    long_term_priors = object_memory.get_long_term_priors(target_type)
                if yolo_reason == "nav_reacquire":
                    nav_yolo_ready_step = -1

            candidate_seen = probe_positive
            if yolo_res:
                hit = yolo_res["det"]["score"] >= float(args.target_conf_thresh)
                low_hit = yolo_res["det"]["score"] >= float(args.yolo_low_conf)
                candidate_seen = candidate_seen or low_hit
                if current_state in {STATE_SEARCH, STATE_SEARCH_NEAR}:
                    search_hits.append(hit)
                if current_state == STATE_LOCALIZE:
                    localize_hits.append(hit)
                if (
                    vlm is not None
                    and context_summary is not None
                    and current_state in {STATE_EXPLORE, STATE_SEARCH, STATE_NAVIGATE}
                    and last_detection is not None
                    and last_detection.get("approach_assessed_step") != total_steps
                ):
                    assess = vlm.assess_approach(Image.fromarray(frame_rgb), target_type, context_summary)
                    if isinstance(assess, dict):
                        approach_possible = assess.get("approach_possible", None)
                        if approach_possible is not None:
                            approach_possible = bool(approach_possible)
                        last_detection["approach_possible"] = approach_possible
                        last_detection["approach_assessment"] = assess
                        last_detection["approach_assessed_step"] = total_steps

            search_confirmed = confirmed_from_hits(search_hits, args.search_confirm_k, args.search_confirm_n)
            localize_confirmed = confirmed_from_hits(localize_hits, args.localize_confirm_k, args.localize_confirm_n)
            lost_target = (
                last_detection is not None and total_steps - int(last_detection.get("step", 0)) > int(args.lost_target_frames)
            )

            if current_state == STATE_LOCALIZE and localize_confirmed:
                trace_actions = []
                for entry in log:
                    step = entry.get("step")
                    if step is None or step < target_start_step or step > total_steps:
                        continue
                    action = entry.get("action", {}) or {}
                    action_item = {"step": int(step), "state": entry.get("state"), "action": action.get("action")}
                    if "degrees" in action:
                        action_item["degrees"] = action.get("degrees")
                    if "moveMagnitude" in action:
                        action_item["moveMagnitude"] = action.get("moveMagnitude")
                    trace_actions.append(action_item)
                object_memory.record_action_trace(
                    target_type,
                    trace_actions,
                    target_start_step,
                    total_steps,
                    success=True,
                    mode=target_spec.get("mode") if target_spec else "global",
                    reference=target_spec.get("reference") if target_spec else None,
                )
                found_targets.append(
                    {
                        "target_type": target_type,
                        "step": total_steps,
                        "reason": "localize_confirmed",
                        "last_detection": last_detection,
                        "mode": target_spec.get("mode") if target_spec else "global",
                        "reference": target_spec.get("reference") if target_spec else None,
                    }
                )
                enter_state(STATE_SELECT_TARGET, "target_done")
                continue

            forced_action = None
            close_enough = False
            approach_oracle_info = None
            if current_state == STATE_APPROACH:
                oracle = get_oracle_target(event, target_type)
                if oracle:
                    if last_detection is None:
                        last_detection = {"det": None, "world": None, "step": total_steps, "dist_m": None}
                    last_detection["step"] = total_steps
                    obj = oracle.get("object") or {}
                    pos = obj.get("position")
                    if pos:
                        last_detection["world"] = {"position": pos, "source": "metadata", "object_id": obj.get("objectId")}
                    if "distance" in obj:
                        try:
                            last_detection["dist_m"] = float(obj.get("distance"))
                        except Exception:
                            pass
                    centroid = oracle.get("centroid")
                    bbox = oracle.get("bbox_xyxy")
                    if centroid is not None and bbox is not None:
                        last_detection["det"] = {
                            "centroid_px": list(centroid),
                            "bbox_xyxy": list(bbox),
                            "score": 1.0,
                            "label": target_type,
                        }
                if args.oracle_approach:
                    forced_action, close_enough, approach_last_pos, approach_stuck_counter, approach_oracle_info = (
                        oracle_approach_step(
                            event,
                            target_type,
                            args,
                            sensor,
                            approach_last_pos,
                            approach_stuck_counter,
                        )
                    )
                    approach_confirm = approach_confirm + 1 if close_enough else max(0, approach_confirm - 1)
                else:
                    det = last_detection["det"] if oracle and last_detection else None
                    forced_action, close_enough, approach_confirm = compute_approach_action(
                        event,
                        det,
                        sensor,
                        args,
                        target_type,
                        approach_confirm,
                        approach_stuck_steps,
                    )

            strong_action = (
                forced_action
                or nav_action_spec
                or hint_action
                or last_seen_action
            )
            if strong_action and action_mgr.queue:
                # Se abbiamo una guida forte, svuotiamo le macro in coda per non ignorarla.
                action_mgr.queue = []
                action_mgr.scan_remaining = 0

            if current_state == STATE_APPROACH and forced_action:
                action_spec = dict(forced_action)
                decision = {"source": "approach", "overrides": []}
                scan_completed = False
                action_mgr.history.append(action_spec.get("action", ""))
            else:
                action_spec, decision, scan_completed = action_mgr.select_action(
                    forced_action
                    if forced_action
                    else (
                        nav_action_spec
                        if nav_action_spec
                        else (hint_action if hint_action else last_seen_action)
                        or vlm_action_spec
                    ),
                    sensor,
                    total_steps,
                )
            if current_state in {STATE_LOCALIZE, STATE_SEARCH_NEAR} and action_spec.get("action") in {
                "MoveAhead",
                "MoveBack",
                "MoveLeft",
                "MoveRight",
            }:
                action_spec = {"action": "RotateLeft", "degrees": int(args.approach_rotate_degrees)}
                decision["overrides"].append("state_no_move")
            if action_spec.get("action") == "Done" and current_state not in {STATE_DONE, STATE_FAIL}:
                action_spec = {"action": "RotateLeft", "degrees": int(args.approach_rotate_degrees)}
                decision["overrides"].append("state_no_done")

            if nav_reacquire_action and action_spec.get("action") in {"RotateLeft", "RotateRight"}:
                nav_reacquire_pending_check = True
                nav_reacquire_side = "left" if action_spec["action"] == "RotateLeft" else "right"
                nav_reacquire_degrees = action_spec.get("degrees", int(args.probe_scan_degrees))

            if int(args.log_every) > 0 and total_steps % int(args.log_every) == 0:
                yolo_flag = "Y" if yolo_res else "-"
                probe_flag = "P" if probe_positive else "-"
                action_txt = format_action_spec(action_spec)
                reason_txt = yolo_reason or ""
                hint_deg = hint_turn_deg.get(target_type)
                hint_txt = ""
                if probe_hint:
                    hint_txt = f" | Hint={probe_hint}"
                    if hint_deg is not None:
                        hint_txt += f"({int(hint_deg)})"
                last_seen_txt = ""
                if last_seen_hint.get(target_type):
                    last_seen_txt = f" | LastSeen={last_seen_hint.get(target_type)}"
                dist_target = None
                if oracle_nav_summary and oracle_nav_summary.get("distance_m") is not None:
                    try:
                        dist_target = float(oracle_nav_summary.get("distance_m"))
                    except Exception:
                        dist_target = None
                elif last_detection:
                    try:
                        dist_target = float(last_detection.get("dist_m"))
                    except Exception:
                        dist_target = None
                dist_txt = f" | dist_target={dist_target:.2f}m" if dist_target is not None else " | dist_target=-"
                assess_txt = ""
                assess = (last_detection or {}).get("approach_assessment") or {}
                if assess:
                    ap = assess.get("approach_possible")
                    conf = assess.get("confidence")
                    if ap is not None:
                        assess_txt = f" | Assess={str(bool(ap))}"
                        if conf is not None:
                            try:
                                conf_val = float(conf)
                            except Exception:
                                conf_val = None
                            if conf_val is not None:
                                assess_txt += f"({conf_val:.2f})"
                macro_txt = "ON" if action_mgr.queue else "OFF"
                print_ok(
                    f"S{total_steps:03d} | {current_state:<11} | T={target_type} | A={action_txt} | "
                    f"Y={yolo_flag}({reason_txt}) | VLM={vlm_conf:.2f} | Probe={probe_flag}{hint_txt}{last_seen_txt}{dist_txt}{assess_txt} | Macro={macro_txt}"
                )

            debug_logger.log(
                {
                    "event": "step_decision",
                    "step": total_steps,
                    "state": current_state,
                    "state_steps": state_steps,
                    "target": target_type,
                    "target_index": target_idx,
                    "approach_active": approach_active,
                    "probe": probe_info,
                    "probe_hint": probe_hint,
                    "hint_action": hint_action,
                    "probe_positive": probe_positive,
                    "last_seen_hint": last_seen_hint.get(target_type),
                    "last_seen_age": (total_steps - last_seen_step.get(target_type))
                    if last_seen_step.get(target_type) is not None
                    else None,
                    "yolo_reason": yolo_reason,
                    "yolo_detected": bool(yolo_res),
                    "vlm_action": action,
                    "forced_action": forced_action,
                    "approach_oracle": approach_oracle_info,
                    "approach_assessment": (last_detection or {}).get("approach_assessment"),
                    "nav_action": nav_action_spec,
                    "final_action": action_spec,
                    "action_meta": decision,
                    "mem_summary": {
                        "coverage": mem_summary.get("coverage_pct"),
                        "novelty": mem_summary.get("novelty_score"),
                    },
                    "distances": {
                        "front": sensor.get("dist_front_m"),
                        "left": sensor.get("dist_left_m"),
                        "right": sensor.get("dist_right_m"),
                    },
                    "action_manager": action_mgr.get_state(),
                }
            )

            next_event = controller.step(**action_spec)
            post_bgr = get_rgb_bgr(next_event)
            post_sensor = build_sensor_state(next_event, depth_radius=args.depth_radius)
            memory.update(post_sensor["position"], post_sensor["yaw"])
            post_mem = memory.summarize(post_sensor["position"], post_sensor["yaw"], post_sensor)
            if args.save_frames:
                telem = annotate_telemetry(post_bgr, total_steps, action_spec, post_sensor, post_mem)
                frame_path = frames_dir / f"step_{total_steps:04d}.png"
                cv2.imwrite(str(frame_path), telem)
                if args.save_depth:
                    depth_path = frames_dir / f"depth_step_{total_steps:04d}.png"
                    save_depth_frame(getattr(next_event, "depth_frame", None), depth_path)
            history.append(action_spec.get("action", ""))
            log.append(
                {
                    "step": total_steps,
                    "state": current_state,
                    "state_steps": state_steps,
                    "target": target_type,
                    "target_index": target_idx,
                    "approach_active": approach_active,
                    "action": action_spec,
                    "action_meta": decision,
                    "nav_action": nav_action_spec,
                    "approach_assessment": (last_detection or {}).get("approach_assessment"),
                    "vlm": {
                        "raw": action_data,
                        "target_confidence": vlm_conf,
                        "request_yolo": vlm_request,
                        "reason": vlm_reason,
                    },
                    "probe": probe_info,
                    "probe_hint": probe_hint,
                    "hint_action": hint_action,
                    "probe_raw": probe_raw,
                    "object_memory": object_summary,
                    "long_term_priors": long_term_priors,
                    "robot_state": post_sensor,
                    "memory": post_mem,
                }
            )
            dist_now = None
            if use_oracle_nav:
                oracle_post = get_oracle_nav_summary(next_event, target_type)
                if oracle_post and oracle_post.get("distance_m") is not None:
                    try:
                        dist_now = float(oracle_post.get("distance_m"))
                    except Exception:
                        dist_now = None
            if dist_now is None and last_detection and (last_detection.get("world") or {}).get("position"):
                pos = post_sensor.get("position", {}) or {}
                dx = float(last_detection["world"]["position"].get("x", 0.0)) - float(pos.get("x", 0.0))
                dz = float(last_detection["world"]["position"].get("z", 0.0)) - float(pos.get("z", 0.0))
                dist_now = (dx * dx + dz * dz) ** 0.5
                last_detection["dist_m"] = dist_now
            if dist_now is not None:
                if current_state == STATE_APPROACH and not args.oracle_approach:
                    if last_approach_dist is not None:
                        if dist_now >= float(last_approach_dist) - float(args.nav_min_progress_m):
                            approach_stuck_steps += 1
                        else:
                            approach_stuck_steps = 0
                    last_approach_dist = dist_now
                else:
                    approach_stuck_steps = 0
                    last_approach_dist = None
                if current_state == STATE_NAVIGATE and action_spec.get("action") in {
                    "MoveAhead",
                    "MoveBack",
                    "MoveLeft",
                    "MoveRight",
                }:
                    if nav_last_dist is not None:
                        if dist_now >= float(nav_last_dist) - float(args.nav_min_progress_m):
                            nav_no_progress_steps += 1
                        else:
                            nav_no_progress_steps = 0
                    nav_last_dist = dist_now
            if scan_completed:
                pending_scan_yolo = True
            total_steps += 1
            state_steps += 1

            # Transizioni FSM basate su segnali misurabili.
            recent_detection = (
                last_detection is not None and int(last_detection.get("step", -1)) == int(total_steps - 1)
            )
            if current_state in {STATE_EXPLORE, STATE_SEARCH} and recent_detection:
                det_score = float((last_detection.get("det") or {}).get("score", 0.0))
                det_source = last_detection.get("source")
                det_dist = last_detection.get("dist_m")
                approach_possible = last_detection.get("approach_possible")
                if approach_possible is False:
                    enter_state(STATE_NAVIGATE, "vlm_blocked")
                    continue
                if det_dist is not None and approach_possible is True:
                    if det_dist > float(args.navigate_dist_thresh_m):
                        enter_state(STATE_NAVIGATE, "vlm_far")
                        continue
                    enter_state(STATE_APPROACH, "vlm_ok")
                    continue
                if det_dist is not None and approach_possible is None:
                    if det_source == "metadata" or det_score >= float(args.yolo_low_conf):
                        if det_dist > float(args.navigate_dist_thresh_m):
                            enter_state(STATE_NAVIGATE, "yolo_far")
                            continue
                        if det_dist > float(args.approach_dist_thresh_m):
                            enter_state(STATE_APPROACH, "yolo_mid")
                            continue

            if current_state == STATE_EXPLORE:
                transition = next_state_explore(candidate_seen, state_steps, args.max_explore_steps)
                if transition:
                    enter_state(*transition)
            elif current_state == STATE_SEARCH:
                transition = next_state_search(search_confirmed, state_steps, args.max_search_steps)
                if transition:
                    enter_state(*transition)
            elif current_state == STATE_SEARCH_NEAR:
                transition = next_state_search_near(
                    anchor_missing,
                    anchor_dist,
                    args.near_radius,
                    search_confirmed,
                    state_steps,
                    args.near_max_steps,
                )
                if transition:
                    enter_state(*transition)
            elif current_state == STATE_NAVIGATE:
                oracle_dist = None
                if oracle_nav_summary and oracle_nav_summary.get("distance_m") is not None:
                    try:
                        oracle_dist = float(oracle_nav_summary.get("distance_m"))
                    except Exception:
                        oracle_dist = None
                det_dist = (
                    oracle_dist
                    if oracle_dist is not None
                    else (last_detection.get("dist_m") if last_detection else None)
                )
                if nav_no_progress_steps >= int(args.nav_no_progress_steps):
                    print_warn("NAVIGATE: no progress, forcing replan.")
                    nav_subgoals.clear()
                    nav_subgoal_plan_queue.clear()
                    nav_scan_queue.clear()
                    nav_reacquire_queue.clear()
                    nav_need_plan = True
                    nav_no_progress_steps = 0
                    nav_last_dist = None
                    nav_yolo_ready_step = -1
                transition = next_state_navigate(
                    det_dist is not None,
                    det_dist,
                    args.approach_dist_thresh_m,
                    args.navigate_dist_thresh_m,
                )
                if transition:
                    enter_state(*transition)
            elif current_state == STATE_APPROACH:
                transition = next_state_approach(
                    close_enough,
                    0 if args.oracle_approach else approach_stuck_steps,
                    args.nav_stuck_steps,
                    lost_target,
                )
                if transition:
                    enter_state(*transition)
            elif current_state == STATE_LOCALIZE:
                transition = next_state_localize(state_steps, args.max_localize_steps)
                if transition:
                    enter_state(*transition)

    finally:
        controller.stop()

    safe_write_json(run_dir / "run_log.json", log)
    found_labels = {t.get("target_type") for t in found_targets}
    missing_targets = [t["label"] for t in target_queue if t.get("label") not in found_labels]
    if missing_targets:
        print_warn("Targets not fully found.")
        fail_out = {
            "goal": args.goal,
            "scene": args.scene,
            "targets": target_queue,
            "found_targets": found_targets,
            "missing_targets": missing_targets,
            "success": False,
            "reason": "not_detected",
            "step": total_steps,
        }
        safe_write_json(run_dir / "detection.json", fail_out)
    else:
        try:
            det_path = run_dir / "detection.json"
            det_data = {}
            if det_path.exists():
                det_data = json.loads(det_path.read_text(encoding="utf-8"))
            det_data["success"] = True
            det_data["final_reason"] = "all_targets_found"
            det_data["targets"] = target_queue
            det_data["found_targets"] = found_targets
            det_data["step"] = total_steps
            safe_write_json(det_path, det_data)
        except Exception:
            pass
        print_ok(f"Detection saved: {run_dir / 'detection.json'}")
