import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "state"


def run_step(title: str, args_list):
    cmd = [sys.executable] + list(args_list)
    print(f"\n[STEP] {title}")
    print("[CMD] ", " ".join(str(x) for x in cmd))
    p = subprocess.run(cmd, cwd=str(ROOT))
    if p.returncode != 0:
        raise SystemExit(f"[FAIL] Step '{title}' exit code={p.returncode}")


def ensure_teleport_state_from_arm_pose():
    tele_path = STATE_DIR / "teleport_state.json"
    if tele_path.exists():
        return

    arm_path = STATE_DIR / "arm_best_pose_state.json"
    if not arm_path.exists():
        raise SystemExit("[FAIL] Manca sia teleport_state.json sia arm_best_pose_state.json.")

    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    scene = arm.get("scene") or arm.get("sceneName") or "FloorPlan1"
    target = arm.get("target") or {}
    agent = arm.get("agent") or arm

    pos = agent.get("position")
    rot = agent.get("rotation")
    horizon = agent.get("horizon", agent.get("cameraHorizon", 20.0))
    if not (isinstance(pos, dict) and isinstance(rot, dict)):
        raise SystemExit("[FAIL] arm_best_pose_state.json non contiene agent.position/agent.rotation validi.")

    tele = {
        "scene": scene,
        "target": target,
        "agent": {"position": pos, "rotation": rot, "horizon": float(horizon)},
        "source": "arm_best_pose_state",
    }
    tele_path.write_text(json.dumps(tele, indent=2), encoding="utf-8")

def ensure_arm_best_pose(scene: str, target_type: str):
    arm_path = STATE_DIR / "arm_best_pose_state.json"
    if arm_path.exists():
        return
    # se non c'è, prova a calcolarla da teleport_state senza esplorazione
    tele_path = STATE_DIR / "teleport_state.json"
    if not tele_path.exists():
        raise SystemExit("[FAIL] Manca arm_best_pose_state.json e non esiste teleport_state.json per ricostruirla.")
    run_step(
        "02 Best arm pose (no explore)",
        [
            "src/02_exploration_and_best_arm_pose.py",
            "--no_explore",
            "--scene",
            scene,
            "--target_type",
            target_type,
            "--teleport_out",
            "teleport_state.json",
            "--arm_pose_out",
            "arm_best_pose_state.json",
        ],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="FloorPlan1")
    ap.add_argument("--target_type", default="Apple")
    ap.add_argument("--yolo_model", default="yolov8n-seg.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--vlm_model", default="google/gemma-3-4b-it")
    ap.add_argument(
        "--nav",
        choices=["file13", "step02"],
        default="file13",
        help="Metodo navigazione: 'file13' usa src/13_autonomous_exploration.py; 'step02' usa src/02_exploration_and_best_arm_pose.py.",
    )
    ap.add_argument("--skip_nav", action="store_true", help="Salta la navigazione e usa gli state json già presenti.")
    ap.add_argument("--skip_vlm", action="store_true")
    ap.add_argument("--skip_pick", action="store_true")
    args = ap.parse_args()

    # 1) Navigazione/esplorazione: o file 13 oppure lo step 02 derivato
    if not args.skip_nav:
        if args.nav == "file13":
            run_step("13 Autonomous exploration", ["src/13_autonomous_exploration.py"])
        else:
            run_step(
                "02 Exploration and best arm pose",
                [
                    "src/02_exploration_and_best_arm_pose.py",
                    "--scene",
                    args.scene,
                    "--target_type",
                    args.target_type,
                    "--teleport_out",
                    "teleport_state.json",
                    "--arm_pose_out",
                    "arm_best_pose_state.json",
                ],
            )

    # se manca teleport_state, lo ricostruiamo da arm_best_pose_state (tipico quando nav=file13)
    ensure_teleport_state_from_arm_pose()
    # se manca arm_best_pose_state, la calcoliamo da teleport_state (senza esplorare)
    ensure_arm_best_pose(args.scene, args.target_type)

    # 2) YOLO-seg: bbox/mask -> centroide pixel (salva centroid_state.json)
    run_step(
        "03 YOLO segmentation centroid",
        [
            "src/03_yolo_seg_centroid.py",
            "--model",
            args.yolo_model,
            "--conf",
            str(args.yolo_conf),
            "--target",
            args.target_type,
            "--oracle_assist",
            "--out_state",
            "centroid_state.json",
            "--debug_name",
            "03_yolo_centroid.png",
        ],
    )

    # 3) Depth + centroide -> punto 3D world (pregrasp_point.json)
    run_step(
        "04 Pregrasp point from depth",
        [
            "src/04_pregrasp_point.py",
        ],
    )

    # 4) (Opzionale) VLM grasp hints (usa immagine + bbox da centroid_state)
    if not args.skip_vlm:
        run_step(
            "11 VLM grasp hints",
            [
                "src/11_vlm_grasp_hints.py",
                "--model",
                args.vlm_model,
                "--image",
                "data/frames/02_teleport_rgb.png",
                "--bbox_state",
                "data/state/centroid_state.json",
                "--out",
                "data/vlm/grasp_hints.json",
                "--raw_out",
                "data/vlm/raw_output.txt",
            ],
        )

    # 5) Grasp point da pregrasp (sensor-based) -> aggiorna grasp_point_bbox.json
    run_step(
        "04b Grasp point from pregrasp",
        [
            "src/04b_grasp_point_bbox.py",
            "--from_pregrasp",
            "pregrasp_point.json",
            "--out",
            "grasp_point_bbox.json",
        ],
    )

    # 7) Pick con braccio (usa arm_best_pose_state + grasp_point_bbox + grasp_hints)
    if not args.skip_pick:
        run_step(
            "12 Arm VLM pick",
            [
                "src/12_arm_vlm_pick.py",
            ],
        )

    print("\n[OK] Pipeline completata.")
    print("- Stato: data/state/teleport_state.json, data/state/centroid_state.json, data/state/pregrasp_point.json")
    print("- Debug: data/outputs/03_yolo_centroid.png, data/frames/12_arm_vlm/")


if __name__ == "__main__":
    main()
