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


def load_state(rel_state_path: str):
    p = (ROOT / rel_state_path).resolve()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_state(rel_state_path: str, obj):
    p = (ROOT / rel_state_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="FloorPlan1")
    ap.add_argument("--target_type", default="Apple", help="Obiettivo (stringa): es. Apple, Mug, Bottle")
    ap.add_argument("--yolo_model", default="yolov8n-seg.pt")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--vlm_model", default="google/gemma-3-4b-it")
    ap.add_argument("--use_vlm", action="store_true", help="Se attivo, genera data/vlm/grasp_hints.json con VLM.")
    ap.add_argument(
        "--nav",
        choices=["file13", "step02"],
        default="step02",
        help="Metodo navigazione per ogni ciclo: 'file13' usa src/13_autonomous_exploration.py, 'step02' usa src/02_exploration_and_best_arm_pose.py.",
    )
    ap.add_argument("--skip_nav", action="store_true", help="Salta navigazione e usa gli state json già presenti.")
    ap.add_argument("--max_cycles", type=int, default=4, help="Quante iterazioni Percezione->Piano->Azione.")
    args = ap.parse_args()

    # Loop: se fallisce, rifai percezione (oggetto potrebbe essere spostato) e ricalcola posa migliore.
    for cycle in range(1, max(1, int(args.max_cycles)) + 1):
        print(f"\n===== CYCLE {cycle}/{args.max_cycles} =====")

        # 1) Navigazione/esplorazione + posa arm
        if not args.skip_nav:
            if args.nav == "file13":
                run_step("13 Autonomous exploration", ["src/13_autonomous_exploration.py"])
                ensure_teleport_state_from_arm_pose()
                # se non esiste arm_best_pose_state, calcolala da teleport_state (senza esplorazione)
                if not (STATE_DIR / "arm_best_pose_state.json").exists():
                    run_step(
                        "02 Best arm pose (no explore)",
                        [
                            "src/02_exploration_and_best_arm_pose.py",
                            "--no_explore",
                            "--scene",
                            args.scene,
                            "--teleport_out",
                            "teleport_state.json",
                            "--arm_pose_out",
                            "arm_best_pose_state.json",
                        ],
                    )
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
        ensure_teleport_state_from_arm_pose()

        # 2) YOLO-seg: bbox/mask -> centroide pixel (+ oracle assist per robustezza in demo)
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
                f"03_yolo_centroid_cycle{cycle:02d}.png",
            ],
        )

        # 3) Depth + centroide -> punto 3D world (sensor-based)
        run_step("04 Pregrasp point from depth", ["src/04_pregrasp_point.py"])

        # 4) VLM hints: passa anche contesto (pregrasp/depth) per non andare "alla cieca"
        if args.use_vlm:
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
                    "--context_state",
                    "data/state/pregrasp_point.json",
                    "--out",
                    "data/vlm/grasp_hints.json",
                    "--raw_out",
                    f"data/vlm/raw_output_cycle{cycle:02d}.txt",
                ],
            )

        # 5) Costruisci grasp_point_bbox.json dal pregrasp (sensor-based)
        run_step(
            "04b Build grasp_point_bbox from pregrasp",
            [
                "src/04b_grasp_point_bbox.py",
                "--from_pregrasp",
                "pregrasp_point.json",
                "--out",
                "grasp_point_bbox.json",
            ],
        )

        # 6) (Opzionale) raffinamento posa arm con centro sensor-based (pregrasp): evita mismatch dopo contatti
        run_step(
            "02 Refinement best arm pose (sensor-based)",
            [
                "src/02_exploration_and_best_arm_pose.py",
                "--no_explore",
                "--center_state",
                "pregrasp_point.json",
                "--k",
                "50",
                "--scene",
                args.scene,
                "--teleport_out",
                "teleport_state.json",
                "--arm_pose_out",
                "arm_best_pose_state.json",
            ],
        )

        # 7) Pick con braccio (usa arm_best_pose_state + grasp_point_bbox + grasp_hints)
        run_step("12 Arm pick", ["src/12_arm_vlm_pick.py"])

        result = load_state("data/state/12_pick_result.json")
        if isinstance(result, dict) and result.get("success") is True:
            print("\n[OK] Pickup riuscito.")
            return

        # Se fallisce, continua: l'oggetto potrebbe essersi spostato dopo il contatto.
        print("\n[WARN] Pickup non riuscito in questo ciclo; rifaccio percezione e riprovo.")
        if isinstance(result, dict) and result.get("last_error"):
            print("[WARN] last_error:", result["last_error"])

    print("\n[FAIL] Tutti i cicli falliti.")
    res = load_state("data/state/12_pick_result.json") or {}
    res["success"] = False
    res["cycles"] = int(args.max_cycles)
    write_state("data/state/14_integrated_result.json", res)


if __name__ == "__main__":
    main()
