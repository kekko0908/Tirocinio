# Scopo: orchestratore della pipeline (CLI + ciclo percezione/azione).
import argparse
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common import EnvConfig, make_controller, get_rgb_bgr, print_ok, print_warn

from pipeline_modules.action_manager import ActionManager
from pipeline_modules.memory import ExplorationMemory
from pipeline_modules.sensors import build_sensor_state, save_depth_frame
from pipeline_modules.utils import (
    bbox_iou,
    guess_target_from_goal,
    normalize_action,
    safe_write_json,
)
from pipeline_modules.visualization import (
    annotate_telemetry,
    draw_comparison,
    draw_detection,
    draw_vlm_bbox,
    draw_vlm_not_visible,
    draw_yolo_evidence,
)
from pipeline_modules.vlm import VLMEngine
from pipeline_modules.yolo import YOLODetector


DEFAULT_VLM_MODEL = "google/gemma-3-4b-it"
DEFAULT_YOLO_MODEL = "yolo11x-seg.pt"

ACTION_SET = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Done"]


def fallback_plan(goal_text: str, target_hint: str):
    target = target_hint or guess_target_from_goal(goal_text) or "Apple"
    return {
        "target_type": target,
        "subgoals": [
            {"id": 1, "type": "explore", "description": "Explore the scene and scan for the target."},
            {"id": 2, "type": "search", "description": "Move to keep the target centered in view."},
            {"id": 3, "type": "localize", "description": "Run YOLO to localize the target."},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="", help="Natural language goal, e.g. 'Cercami la mela'.")
    ap.add_argument("--scene", default="FloorPlan1")
    ap.add_argument("--target_type", default="", help="Optional override for the target label.")
    ap.add_argument("--vlm_model", default=DEFAULT_VLM_MODEL)
    ap.add_argument("--yolo_model", default=DEFAULT_YOLO_MODEL)
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--max_steps_per_subgoal", type=int, default=40)
    ap.add_argument("--yolo_every", type=int, default=1, help="Force periodic YOLO every N steps (0=disabled).")
    ap.add_argument("--yolo_cooldown", type=int, default=4)
    ap.add_argument("--target_conf_thresh", type=float, default=0.6)
    ap.add_argument("--cell_size", type=float, default=0.5)
    ap.add_argument("--scan_degrees", type=int, default=60)
    ap.add_argument("--scan_trigger", type=float, default=0.5)
    ap.add_argument("--scan_cooldown", type=int, default=15)
    ap.add_argument("--safe_front_m", type=float, default=0.6)
    ap.add_argument("--advance_steps", type=int, default=2)
    ap.add_argument("--advance_min_front", type=float, default=1.2)
    ap.add_argument("--depth_radius", type=int, default=12)
    ap.add_argument("--save_depth", action="store_true")
    ap.add_argument("--iou_threshold", type=float, default=0.4)
    ap.add_argument("--no_vlm_bbox", action="store_true")
    ap.add_argument("--save_frames", action="store_true", default=True, help="Save frames (default).")
    ap.add_argument("--no_save_frames", action="store_false", dest="save_frames", help="Disable saving frames.")
    ap.add_argument("--no_vlm", action="store_true")
    args = ap.parse_args()

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

    plan = None
    if vlm is not None and args.goal:
        try:
            plan = vlm.plan_subgoals(args.goal)
        except Exception as e:
            print_warn(f"Plan failed, fallback. Err={e}")
            plan = None
    if not plan or "subgoals" not in plan:
        plan = fallback_plan(args.goal or "", args.target_type)
    if args.target_type:
        plan["target_type"] = args.target_type
    target_type = plan.get("target_type") or guess_target_from_goal(args.goal) or "Apple"

    safe_write_json(run_dir / "plan.json", plan)
    print_ok(f"Plan saved: {run_dir / 'plan.json'}")
    print_ok(f"Target: {target_type}")

    detector = YOLODetector(args.yolo_model, args.yolo_conf, args.imgsz, target_type)

    cfg = EnvConfig(scene=args.scene, render_depth=True, render_instance_segmentation=False)
    controller = make_controller(cfg)

    total_steps = 0
    history = []
    log = []
    found = None
    last_yolo_step = -999
    pending_scan_yolo = False

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
    init_event = controller.last_event
    init_sensor = build_sensor_state(init_event, depth_radius=args.depth_radius)
    memory.update(init_sensor["position"], init_sensor["yaw"])
    init_mem = memory.summarize(init_sensor["position"], init_sensor["yaw"], init_sensor)
    if args.save_frames:
        init_bgr = get_rgb_bgr(init_event)
        init_frame = annotate_telemetry(init_bgr, 0, {"action": "INIT"}, init_sensor, init_mem)
        cv2.imwrite(str(frames_dir / "step_init_0000.png"), init_frame)
        if args.save_depth:
            save_depth_frame(getattr(init_event, "depth_frame", None), frames_dir / "depth_init_0000.png")

    def try_yolo(frame_rgb, frame_bgr, reason: str) -> bool:
        nonlocal found, last_yolo_step
        if total_steps - last_yolo_step < int(args.yolo_cooldown):
            return False
        det, mask01 = detector.detect(frame_rgb)
        last_yolo_step = total_steps
        if det is None:
            return False
        if det["score"] < float(args.yolo_conf):
            return False
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
        found = det_out
        print_ok("Target localized with YOLO.")
        return True

    try:
        for sg in plan["subgoals"]:
            sg_id = sg.get("id", "")
            sg_desc = sg.get("description", "")
            sg_type = (sg.get("type", "") or "").lower()
            print_ok(f"Subgoal {sg_id}: {sg_type} | {sg_desc}")

            for _ in range(int(args.max_steps_per_subgoal)):
                if total_steps >= int(args.max_steps):
                    break
                event = controller.last_event
                frame_rgb = event.frame
                frame_bgr = get_rgb_bgr(event)
                sensor = build_sensor_state(event, depth_radius=args.depth_radius)
                mem_summary = memory.summarize(sensor["position"], sensor["yaw"], sensor)

                action_mgr.maybe_enqueue_scan(mem_summary["coverage_ratio"], total_steps)
                action_mgr.maybe_enqueue_advance(sensor["dist_front_m"], mem_summary["coverage_ratio"])

                if pending_scan_yolo and total_steps - last_yolo_step >= int(args.yolo_cooldown):
                    pending_scan_yolo = False
                    if try_yolo(frame_rgb, frame_bgr, "scan_complete"):
                        break

                action = None
                action_data = None
                vlm_conf = 0.0
                vlm_request = False
                vlm_reason = ""

                if vlm is not None:
                    context_summary = {
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
                        "constraints": {"safe_front_m": round(float(args.safe_front_m), 2)},
                    }
                    img = Image.fromarray(frame_rgb)
                    action_data = vlm.choose_action(img, sg_desc, history, context_summary)
                    if isinstance(action_data, dict):
                        action = normalize_action(action_data.get("action"))
                        try:
                            vlm_conf = float(action_data.get("target_confidence", 0.0) or 0.0)
                        except Exception:
                            vlm_conf = 0.0
                        vlm_conf = max(0.0, min(1.0, vlm_conf))
                        vlm_request = bool(action_data.get("request_yolo", False))
                        vlm_reason = str(action_data.get("reason", ""))[:120]

                yolo_reason = None
                if vlm_request:
                    yolo_reason = "vlm_request"
                elif vlm_conf >= float(args.target_conf_thresh):
                    yolo_reason = f"vlm_conf_{vlm_conf:.2f}"
                elif int(args.yolo_every) > 0 and total_steps % int(args.yolo_every) == 0:
                    yolo_reason = "periodic"
                elif sg_type == "localize":
                    yolo_reason = "subgoal_localize"

                if yolo_reason and total_steps - last_yolo_step >= int(args.yolo_cooldown):
                    if try_yolo(frame_rgb, frame_bgr, yolo_reason):
                        break

                action_spec, decision, scan_completed = action_mgr.select_action(action, sensor, total_steps)
                if action_spec.get("action") == "Done":
                    break

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
                        "subgoal": sg_id,
                        "action": action_spec,
                        "action_meta": decision,
                        "vlm": {
                            "raw": action_data,
                            "target_confidence": vlm_conf,
                            "request_yolo": vlm_request,
                            "reason": vlm_reason,
                        },
                        "robot_state": post_sensor,
                        "memory": post_mem,
                    }
                )
                if scan_completed:
                    pending_scan_yolo = True
                total_steps += 1

            if found is not None or total_steps >= int(args.max_steps):
                break

    finally:
        controller.stop()

    safe_write_json(run_dir / "run_log.json", log)
    if found is None:
        print_warn("Target not found.")
        fail_out = {
            "goal": args.goal,
            "scene": args.scene,
            "target_type": target_type,
            "found": False,
            "reason": "not_detected",
            "step": total_steps,
        }
        safe_write_json(run_dir / "detection.json", fail_out)
    else:
        print_ok(f"Detection saved: {run_dir / 'detection.json'}")


if __name__ == "__main__":
    main()
