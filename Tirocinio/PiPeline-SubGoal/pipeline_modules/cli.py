# Scopo: definizione CLI (argparse) con default centralizzati.
import argparse


def build_arg_parser(default_vlm_model: str, default_yolo_model: str) -> argparse.ArgumentParser:
    """
    Costruisce il parser CLI della pipeline.
    Imposta default per modelli, soglie e limiti di step.
    Ritorna un argparse.ArgumentParser pronto all'uso.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="", help="Natural language goal, e.g. 'Cercami la mela'.")  # goal testuale
    ap.add_argument("--scene", default="FloorPlan1")  # scena AI2-THOR
    ap.add_argument("--target_type", default="", help="Optional override for the target label.")  # target forzato
    ap.add_argument("--vlm_model", default=default_vlm_model)  # modello VLM
    ap.add_argument("--yolo_model", default=default_yolo_model)  # modello YOLO
    ap.add_argument("--yolo_conf", type=float, default=0.25)  # soglia confidenza YOLO
    ap.add_argument("--imgsz", type=int, default=640)  # dimensione input YOLO
    ap.add_argument("--max_steps", type=int, default=300)  # max step episodio
    ap.add_argument("--max_steps_per_subgoal", type=int, default=40)  # max step per subgoal
    ap.add_argument("--max_explore_steps", type=int, default=None)  # cap step EXPLORE
    ap.add_argument("--max_search_steps", type=int, default=None)  # cap step SEARCH
    ap.add_argument("--max_localize_steps", type=int, default=None)  # cap step LOCALIZE
    ap.add_argument("--near_max_steps", type=int, default=None)  # cap step SEARCH_NEAR
    ap.add_argument("--yolo_every", type=int, default=0, help="Force periodic YOLO every N steps (0=disabled).")  # YOLO periodico
    ap.add_argument("--yolo_cooldown", type=int, default=0)  # cooldown YOLO
    ap.add_argument("--target_conf_thresh", type=float, default=0.6)  # soglia conf target VLM
    ap.add_argument("--yolo_low_conf", type=float, default=0.2)  # soglia bassa YOLO
    ap.add_argument("--search_confirm_k", type=int, default=1)  # k successi SEARCH
    ap.add_argument("--search_confirm_n", type=int, default=1)  # finestra SEARCH
    ap.add_argument("--localize_confirm_k", type=int, default=1)  # k successi LOCALIZE
    ap.add_argument("--localize_confirm_n", type=int, default=1)  # finestra LOCALIZE
    ap.add_argument("--lost_target_frames", type=int, default=10)  # frame perdita target
    ap.add_argument("--near_radius", type=float, default=1.5)  # raggio near mode
    ap.add_argument("--approach_center_tol_px", type=int, default=40)  # tolleranza centraggio px
    ap.add_argument("--approach_dist_thresh_m", type=float, default=0.7)  # soglia distanza approach
    ap.add_argument("--approach_bbox_area_thresh", type=float, default=4000)  # soglia area bbox
    ap.add_argument("--approach_confirm_k", type=int, default=2)  # conferme approach
    ap.add_argument("--approach_rotate_degrees", type=int, default=5)  # gradi rotazione approach
    ap.add_argument("--navigate_dist_thresh_m", type=float, default=0.8)  # soglia NAVIGATE
    ap.add_argument("--nav_plan_max_steps", type=int, default=6)  # max step piano nav
    ap.add_argument("--nav_subgoals_max", type=int, default=5)  # max subgoal nav
    ap.add_argument("--nav_scan_degrees", type=int, default=15)  # gradi scan nav
    ap.add_argument("--nav_stuck_steps", type=int, default=200)  # max step NAVIGATE
    ap.add_argument("--nav_min_progress_m", type=float, default=0.05)  # progresso minimo nav
    ap.add_argument("--nav_no_progress_steps", type=int, default=2)  # step senza progresso
    ap.add_argument("--hint_strong_conf", type=int, default=50)  # soglia hint forte
    ap.add_argument("--probe_every", type=int, default=1)  # probe ogni N step
    ap.add_argument("--probe_conf_thresh", type=int, default=65)  # soglia probe
    ap.add_argument("--probe_scan_degrees", type=int, default=45)  # gradi scan probe
    ap.add_argument("--probe_scan_steps", type=int, default=2)  # step scan probe
    ap.add_argument("--cell_size", type=float, default=0.5)  # cella mappa esplorazione
    ap.add_argument("--scan_degrees", type=int, default=60)  # gradi scan macro
    ap.add_argument("--scan_trigger", type=float, default=0.5)  # soglia scan macro
    ap.add_argument("--scan_cooldown", type=int, default=15)  # cooldown scan macro
    ap.add_argument("--safe_front_m", type=float, default=0.6)  # distanza sicurezza
    ap.add_argument("--advance_steps", type=int, default=2)  # step advance macro
    ap.add_argument("--advance_min_front", type=float, default=1.2)  # spazio minimo advance
    ap.add_argument("--depth_radius", type=int, default=12)  # raggio depth per stima
    ap.add_argument("--save_depth", action="store_true")  # salva frame depth
    ap.add_argument("--iou_threshold", type=float, default=0.4)  # soglia IoU
    ap.add_argument("--no_vlm_bbox", action="store_true")  # disabilita bbox VLM
    ap.add_argument("--save_frames", action="store_true", default=True, help="Save frames (default).")  # salva frame
    ap.add_argument("--no_save_frames", action="store_false", dest="save_frames", help="Disable saving frames.")  # no frame
    ap.add_argument("--save_debug_every", type=int, default=0)  # salva debug ogni N step
    ap.add_argument("--no_vlm", action="store_true")  # disabilita VLM
    ap.add_argument("--topdown_map_size", type=int, default=512)  # size mappa topdown
    ap.add_argument("--topdown_avoid_stride", type=int, default=1)  # stride avoid
    ap.add_argument("--topdown_grid_size", type=float, default=0.15)  # densita punti mappa
    ap.add_argument("--topdown_path_step_m", type=float, default=0.3)  # connessioni path
    ap.add_argument("--topdown_checkpoint_step_m", type=float, default=0.8)  # distanza checkpoint
    ap.add_argument("--log_every", type=int, default=1, help="Print step summary every N steps (0=off).")  # log step
    ap.add_argument(
        "--step_delay",
        type=float,
        default=1.0,
        help="Seconds to sleep after each step (debug/slowdown).",
    )  # pausa tra step
    ap.add_argument("--spawn_yaw", type=float, default=90, help="Optional initial yaw rotation (degrees).")  # yaw iniziale
    ap.add_argument(
        "--oracle_approach",
        action="store_true",
        default=True,
        help="Use oracle APPROACH logic (metadata + instance masks) like file 13.",
    )  # approach oracle
    ap.add_argument(
        "--no_oracle_approach",
        action="store_false",
        dest="oracle_approach",
        help="Disable oracle APPROACH logic.",
    )  # disabilita approach oracle
    return ap
