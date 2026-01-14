# Scopo: definizione CLI (argparse) con default centralizzati.
import argparse


def build_arg_parser(default_vlm_model: str, default_yolo_model: str) -> argparse.ArgumentParser:
    """
    Costruisce il parser CLI della pipeline.
    Imposta default per modelli, soglie e limiti di step.
    Ritorna un argparse.ArgumentParser pronto all'uso.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="", help="Natural language goal, e.g. 'Cercami la mela'.")
    ap.add_argument("--scene", default="FloorPlan1")
    ap.add_argument("--target_type", default="", help="Optional override for the target label.")
    ap.add_argument("--vlm_model", default=default_vlm_model)
    ap.add_argument("--yolo_model", default=default_yolo_model)
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--max_steps_per_subgoal", type=int, default=40)
    ap.add_argument("--max_explore_steps", type=int, default=None)
    ap.add_argument("--max_search_steps", type=int, default=None)
    ap.add_argument("--max_localize_steps", type=int, default=None)
    ap.add_argument("--near_max_steps", type=int, default=None)
    ap.add_argument("--yolo_every", type=int, default=0, help="Force periodic YOLO every N steps (0=disabled).")
    ap.add_argument("--yolo_cooldown", type=int, default=0)
    ap.add_argument("--target_conf_thresh", type=float, default=0.6)
    ap.add_argument("--yolo_low_conf", type=float, default=0.2)
    ap.add_argument("--search_confirm_k", type=int, default=1)
    ap.add_argument("--search_confirm_n", type=int, default=1)
    ap.add_argument("--localize_confirm_k", type=int, default=1)
    ap.add_argument("--localize_confirm_n", type=int, default=1)
    ap.add_argument("--lost_target_frames", type=int, default=10)
    ap.add_argument("--near_radius", type=float, default=1.5)
    ap.add_argument("--approach_center_tol_px", type=int, default=40)
    ap.add_argument("--approach_dist_thresh_m", type=float, default=0.7)
    ap.add_argument("--approach_bbox_area_thresh", type=float, default=4000)
    ap.add_argument("--approach_confirm_k", type=int, default=2)
    ap.add_argument("--approach_rotate_degrees", type=int, default=5)
    ap.add_argument("--navigate_dist_thresh_m", type=float, default=0.8)
    ap.add_argument("--nav_plan_max_steps", type=int, default=6)
    ap.add_argument("--nav_subgoals_max", type=int, default=5)
    ap.add_argument("--nav_scan_degrees", type=int, default=15)
    ap.add_argument("--nav_stuck_steps", type=int, default=200)
    ap.add_argument("--nav_min_progress_m", type=float, default=0.05)
    ap.add_argument("--nav_no_progress_steps", type=int, default=3)
    ap.add_argument("--hint_strong_conf", type=int, default=50)
    ap.add_argument("--probe_every", type=int, default=1)
    ap.add_argument("--probe_conf_thresh", type=int, default=65)
    ap.add_argument("--probe_scan_degrees", type=int, default=45)
    ap.add_argument("--probe_scan_steps", type=int, default=2)
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
    ap.add_argument("--log_every", type=int, default=1, help="Print step summary every N steps (0=off).")
    ap.add_argument("--spawn_yaw", type=float, default=90, help="Optional initial yaw rotation (degrees).")
    ap.add_argument(
        "--oracle_approach",
        action="store_true",
        default=True,
        help="Use oracle APPROACH logic (metadata + instance masks) like file 13.",
    )
    ap.add_argument(
        "--no_oracle_approach",
        action="store_false",
        dest="oracle_approach",
        help="Disable oracle APPROACH logic.",
    )
    return ap
