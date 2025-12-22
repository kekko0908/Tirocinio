import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from PIL import Image
from ai2thor.controller import Controller


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE_DIR = DATA / "state"
VLM_DIR = DATA / "vlm"
FRAMES_DIR = DATA / "frames" / "12_arm_vlm"

POSE_PATH = STATE_DIR / "arm_best_pose_state.json"
GRASP_PATH = STATE_DIR / "grasp_point_bbox.json"
HINTS_PATH = VLM_DIR / "grasp_hints.json"
RESULT_PATH = STATE_DIR / "12_pick_result.json"

W, H = 640, 480
AGENT_MODE = "arm"
MAX_HOVER_Y = 1.95


def info(*a): print("[INFO]", *a)
def warn(*a): print("[WARN]", *a)
def ok(*a): print("[OK]", *a)


def load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        warn(f"JSON non valido: {p} -> {e}")
        return None


def vec(x: float, y: float, z: float) -> Dict[str, float]:
    return {"x": float(x), "y": float(y), "z": float(z)}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def save_frame(controller: Controller, name: str):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(controller.last_event.frame).save(FRAMES_DIR / name)


def unit2(x: float, z: float) -> Tuple[float, float]:
    n = math.hypot(x, z)
    if n < 1e-8:
        return 0.0, 0.0
    return x / n, z / n


def basis_from_yaw(yaw_deg: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    rad = math.radians(float(yaw_deg))
    fwd = (math.sin(rad), math.cos(rad))          # +Z when yaw=0
    right = (math.cos(rad), -math.sin(rad))       # +X when yaw=0
    return fwd, right


def extract_agent(pose_state: Dict[str, Any]) -> Dict[str, Any]:
    agent = pose_state.get("agent") if isinstance(pose_state.get("agent"), dict) else pose_state
    pos = agent.get("position")
    rot = agent.get("rotation")
    if not (isinstance(pos, dict) and isinstance(rot, dict)):
        raise RuntimeError("arm_best_pose_state.json: manca agent.position/agent.rotation")
    horizon = agent.get("horizon", agent.get("cameraHorizon", 0.0))
    return {"position": pos, "rotation": rot, "horizon": float(horizon)}


def restore_pose(controller: Controller, pose_state: Dict[str, Any]):
    agent = extract_agent(pose_state)
    pos = agent["position"]
    rot = agent["rotation"]
    controller.step(
        action="TeleportFull",
        x=float(pos["x"]),
        y=float(pos["y"]),
        z=float(pos["z"]),
        rotation={
            "x": float(rot.get("x", 0.0)),
            "y": float(rot.get("y", 0.0)),
            "z": float(rot.get("z", 0.0)),
        },
        horizon=float(agent["horizon"]),
        standing=True,
    )
    controller.step(action="Pass")


def get_hand_pos(controller: Controller) -> Dict[str, float]:
    arm = controller.last_event.metadata.get("arm", {})
    hc = arm.get("handSphereCenter")
    if not (isinstance(hc, dict) and all(k in hc for k in ("x", "y", "z"))):
        raise RuntimeError("handSphereCenter non presente in metadata['arm']")
    return vec(hc["x"], hc["y"], hc["z"])


def move_arm_world(controller: Controller, goal: Dict[str, float], speed: float = 1.0) -> Tuple[bool, str]:
    ev = controller.step(
        action="MoveArm",
        position=goal,
        coordinateSpace="world",
        speed=float(speed),
        returnToStart=False,
    )
    ok_ = ev.metadata.get("lastActionSuccess", False)
    return ok_, ev.metadata.get("errorMessage", "")


def pickup_object(controller: Controller, object_id: str) -> Tuple[bool, str]:
    try:
        ev = controller.step(
            action="PickupObject",
            objectId=str(object_id),
            forceAction=True,
            manualInteract=True,
        )
    except ValueError as e:
        return False, str(e)
    ok_ = ev.metadata.get("lastActionSuccess", False)
    return ok_, ev.metadata.get("errorMessage", "")

def open_gripper(controller: Controller):
    try:
        controller.step(action="OpenGripper")
    except Exception:
        # alcuni build possono non supportarlo o lanciare ValueError: ignoriamo
        pass


def is_target_in_range(controller: Controller, target_id: str) -> Tuple[bool, bool]:
    arm = controller.last_event.metadata.get("arm", {})
    pickupable = arm.get("pickupableObjects", []) or []
    touched = arm.get("touchedNotHeldObjects", []) or []
    in_pick = any(o.get("objectId") == target_id for o in pickupable if isinstance(o, dict))
    in_touch = any(o.get("objectId") == target_id for o in touched if isinstance(o, dict))
    return in_pick, in_touch


def compute_approach_dir(
    approach: str,
    agent_pos: Dict[str, Any],
    target: Dict[str, float],
    agent_forward: Tuple[float, float],
    agent_right: Tuple[float, float],
) -> Tuple[float, float]:
    a = (approach or "top_down").strip().lower()
    if a == "top_down":
        return 0.0, 0.0
    if a == "from_left":
        return -agent_right[0], -agent_right[1]
    if a == "from_right":
        return agent_right[0], agent_right[1]
    if a == "from_back":
        return agent_forward[0], agent_forward[1]
    if a in ("from_front", "from_agent_side", "from_edge"):
        dx = float(agent_pos["x"]) - float(target["x"])
        dz = float(agent_pos["z"]) - float(target["z"])
        ux, uz = unit2(dx, dz)
        if (ux, uz) != (0.0, 0.0):
            return ux, uz
        return -agent_forward[0], -agent_forward[1]
    dx = float(agent_pos["x"]) - float(target["x"])
    dz = float(agent_pos["z"]) - float(target["z"])
    return unit2(dx, dz)


def make_nudges() -> List[Tuple[float, float]]:
    # (side_m, approach_m)
    return [
        (0.00, 0.00),
        (0.04, 0.00),
        (-0.04, 0.00),
        (0.08, 0.00),
        (-0.08, 0.00),
        (0.00, 0.04),
        (0.00, -0.04),
        (0.04, 0.04),
        (-0.04, 0.04),
        (0.04, -0.04),
        (-0.04, -0.04),
    ]


def move_arm_with_clearance(
    controller: Controller,
    goal: Dict[str, float],
    speed: float,
    attempt_tag: str,
) -> Tuple[bool, str]:
    """
    Tenta MoveArm e, in caso di collisione con island/counter, riprova alzando Y.
    Questo evita di fare MoveBack (che spesso rende la mano troppo lontana).
    """
    ok_m, err_m = move_arm_world(controller, goal, speed=speed)
    if ok_m:
        return True, ""
    if "StandardIslandHeight" not in (err_m or ""):
        return False, err_m

    # Retry: alza solo Y di poco (senza cambiare XZ)
    for k, dy in enumerate((0.05, 0.10, 0.15, 0.25, 0.35, 0.45), start=1):
        g2 = vec(goal["x"], min(goal["y"] + dy, MAX_HOVER_Y), goal["z"])
        ok2, err2 = move_arm_world(controller, g2, speed=speed)
        save_frame(controller, f"{attempt_tag}_CLEARANCE_{k:02d}.png")
        if ok2:
            return True, ""
        err_m = err2

    # Ultimo fallback: micro-rotazione del base (senza arretrare)
    controller.step(action="RotateRight", degrees=6)
    controller.step(action="Pass")
    ok3, err3 = move_arm_world(controller, goal, speed=speed)
    return ok3, err3

def reach_over_counter(
    controller: Controller,
    x: float,
    z: float,
    y: float,
    attempt_tag: str,
    label: str,
) -> Tuple[bool, float, str]:
    """
    Se il movimento a quota y collide con il bancone, prova a salire ancora e ripetere.
    Ritorna (ok, y_usata, err).
    """
    ok_h, err_h = move_arm_with_clearance(controller, vec(x, y, z), speed=1.0, attempt_tag=f"{attempt_tag}_{label}")
    save_frame(controller, f"{attempt_tag}_{label}.png")
    if ok_h:
        return True, y, ""
    if "StandardIslandHeight" not in (err_h or ""):
        return False, y, err_h

    # due segmenti: prima sali in verticale sul punto attuale, poi vai sopra il target a quota maggiore
    for i, y2 in enumerate((min(y + 0.20, MAX_HOVER_Y), min(y + 0.35, MAX_HOVER_Y), MAX_HOVER_Y), start=1):
        hand = get_hand_pos(controller)
        ok_up, err_up = move_arm_with_clearance(
            controller, vec(hand["x"], y2, hand["z"]), speed=1.0, attempt_tag=f"{attempt_tag}_{label}_UP{i}"
        )
        save_frame(controller, f"{attempt_tag}_{label}_UP_{i:02d}.png")
        if not ok_up:
            err_h = err_up
            continue
        ok2, err2 = move_arm_with_clearance(
            controller, vec(x, y2, z), speed=1.0, attempt_tag=f"{attempt_tag}_{label}_AT{i}"
        )
        save_frame(controller, f"{attempt_tag}_{label}_AT_{i:02d}.png")
        if ok2:
            return True, y2, ""
        err_h = err2

    return False, y, err_h


def descend_with_checks(
    controller: Controller,
    start: Dict[str, float],
    end: Dict[str, float],
    steps: int,
    target_id: str,
    attempt_tag: str,
) -> Tuple[bool, str]:
    for s in range(1, steps + 1):
        f = s / steps
        goal = vec(
            start["x"] + (end["x"] - start["x"]) * f,
            start["y"] + (end["y"] - start["y"]) * f,
            start["z"] + (end["z"] - start["z"]) * f,
        )
        ok_m, err_m = move_arm_world(controller, goal, speed=0.9)
        save_frame(controller, f"{attempt_tag}_DESC_{s:02d}.png")
        if not ok_m:
            return False, err_m

        in_pick, in_touch = is_target_in_range(controller, target_id)
        if in_pick or in_touch:
            open_gripper(controller)
            ok_p, err_p = pickup_object(controller, target_id)
            save_frame(controller, f"{attempt_tag}_PICK_INRANGE_{s:02d}.png")
            if ok_p:
                return True, ""
            warn(f"[{attempt_tag}] Pickup in-range fallito: {err_p}")

    return False, ""

def micro_search_and_pick(
    controller: Controller,
    center_x: float,
    center_y: float,
    center_z: float,
    target_id: str,
    attempt_tag: str,
) -> Tuple[bool, str]:
    """
    Piccola ricerca locale attorno a un punto (spirale su XZ + micro jitter su Y)
    per compensare errori di centroide/offset e piccoli mismatch IK (hand != goal).
    """
    deltas = [0.00, 0.015, 0.030, 0.045, 0.060]
    y_jitter = [0.0, 0.01, -0.01, 0.02, -0.02]
    offsets = []
    for d in deltas:
        offsets.extend(
            [
                (d, 0.0),
                (-d, 0.0),
                (0.0, d),
                (0.0, -d),
                (d, d),
                (d, -d),
                (-d, d),
                (-d, -d),
            ]
        )

    j = 0
    last_err = ""
    for dx, dz in offsets:
        for dy in y_jitter:
            j += 1
            goal = vec(center_x + dx, center_y + dy, center_z + dz)
            ok_m, err_m = move_arm_with_clearance(
                controller, goal, speed=0.8, attempt_tag=f"{attempt_tag}_MICRO"
            )
            save_frame(controller, f"{attempt_tag}_MICRO_{j:02d}.png")
            if not ok_m:
                last_err = err_m or last_err
                continue
            open_gripper(controller)
            ok_p, err_p = pickup_object(controller, target_id)
            save_frame(controller, f"{attempt_tag}_MICROPICK_{j:02d}.png")
            if ok_p:
                return True, ""
            last_err = err_p or last_err
    return False, last_err if 'last_err' in locals() else ""


def main():
    pose_state = load_json(POSE_PATH)
    grasp_state = load_json(GRASP_PATH)
    hints = load_json(HINTS_PATH) or {}

    if pose_state is None:
        raise RuntimeError(f"Manca {POSE_PATH}")
    if grasp_state is None:
        raise RuntimeError(f"Manca {GRASP_PATH}")

    scene = str(pose_state.get("scene") or pose_state.get("sceneName") or "FloorPlan1")
    agent = extract_agent(pose_state)
    agent_pos = agent["position"]
    agent_yaw = float(agent["rotation"].get("y", 0.0))
    agent_forward, agent_right = basis_from_yaw(agent_yaw)

    target_meta = grasp_state.get("target", {}) if isinstance(grasp_state.get("target"), dict) else {}
    target_id = str(target_meta.get("objectId", ""))
    target_type = str(target_meta.get("objectType", ""))
    if not target_id:
        raise RuntimeError("grasp_point_bbox.json: manca target.objectId")

    tw = grasp_state.get("bbox_center_world")
    if not (isinstance(tw, dict) and all(k in tw for k in ("x", "y", "z"))):
        raise RuntimeError("grasp_point_bbox.json: manca bbox_center_world")
    target_center = vec(tw["x"], tw["y"], tw["z"])

    bs = grasp_state.get("bbox_size") if isinstance(grasp_state.get("bbox_size"), dict) else {}
    obj_h = float(bs.get("y", 0.12))
    top_y = float(target_center["y"]) + obj_h * 0.5

    approach = str(hints.get("approach", "top_down"))
    prefer_vertical = bool(hints.get("prefer_vertical_descend", True))
    hover_h = float(hints.get("hover_height_hint_m", 0.30))
    hover_h = clamp(hover_h, 0.15, 0.55)

    xz_off = hints.get("xz_offset_hint_m", [0.0, 0.0])
    if not (isinstance(xz_off, list) and len(xz_off) == 2):
        xz_off = [0.0, 0.0]
    off_dx = clamp(float(xz_off[0]), -0.20, 0.20)  # right
    off_dz = clamp(float(xz_off[1]), -0.20, 0.20)  # forward

    ok("Scene:", scene)
    ok("Target:", {"type": target_type, "id": target_id})
    ok("Target center:", target_center)
    ok("VLM hints:", {"approach": approach, "prefer_vertical": prefer_vertical, "hover_h": hover_h, "xz_off": [off_dx, off_dz]})

    # Apply offset in AGENT frame -> world
    target_seed = vec(
        float(target_center["x"]) + agent_right[0] * off_dx + agent_forward[0] * off_dz,
        float(target_center["y"]),
        float(target_center["z"]) + agent_right[1] * off_dx + agent_forward[1] * off_dz,
    )

    hover_y = clamp(top_y + hover_h, 1.25, 1.60)
    final_y = float(target_center["y"]) + min(0.02, obj_h * 0.15)

    approach_dir = compute_approach_dir(approach, agent_pos, target_seed, agent_forward, agent_right)
    side_dir = (-approach_dir[1], approach_dir[0])
    if approach_dir == (0.0, 0.0):
        # top-down: usa frame agente per nudges
        approach_dir = agent_forward
        side_dir = agent_right

    pre_dist = clamp(0.22 + max(float(bs.get("x", 0.10)), float(bs.get("z", 0.10))) * 0.5, 0.20, 0.38)

    controller = Controller(
        scene=scene,
        agentMode=AGENT_MODE,
        width=W,
        height=H,
        renderDepthImage=True,
        snapToGrid=False,
    )

    restore_pose(controller, pose_state)
    open_gripper(controller)
    save_frame(controller, "000_START.png")

    # Strategy: se prefer_vertical -> top-down; altrimenti prova side/edge ma con hover alto e discesa diagonale
    do_vertical = prefer_vertical or approach.strip().lower() == "top_down"

    success = False
    last_err = ""

    for attempt_i, (n_side, n_app) in enumerate(make_nudges(), start=1):
        tag = f"{attempt_i:02d}"
        restore_pose(controller, pose_state)
        # Non arretrare di default: se la posa 02c è buona, il backoff spesso rovina la reach.
        controller.step(action="Pass")

        # raise on the spot
        hand = get_hand_pos(controller)
        ok_r, err_r = move_arm_with_clearance(controller, vec(hand["x"], hover_y, hand["z"]), speed=1.0, attempt_tag=tag)
        save_frame(controller, f"{tag}_RAISE.png")
        if not ok_r:
            last_err = err_r
            continue

        ax, az = approach_dir
        sx, sz = side_dir
        cand_x = float(target_seed["x"]) + sx * n_side + ax * n_app
        cand_z = float(target_seed["z"]) + sz * n_side + az * n_app

        if do_vertical:
            ok_h, used_y, err_h = reach_over_counter(controller, cand_x, cand_z, hover_y, tag, "HOVER")
            if not ok_h:
                last_err = err_h
                continue

            ok_d, err_d = descend_with_checks(
                controller,
                start=vec(cand_x, used_y, cand_z),
                end=vec(cand_x, final_y, cand_z),
                steps=14,
                target_id=target_id,
                attempt_tag=tag,
            )
            if ok_d:
                success = True
                break
            if err_d:
                last_err = err_d
                continue
            # se non siamo mai "in range", prova micro-search:
            # 1) intorno al punto target; 2) intorno alla posizione reale della mano (IK mismatch)
            ok_ms, err_ms = micro_search_and_pick(controller, cand_x, final_y, cand_z, target_id, tag)
            if ok_ms:
                success = True
                break
            if err_ms:
                last_err = err_ms
            hand_now = get_hand_pos(controller)
            ok_ms_h, err_ms_h = micro_search_and_pick(
                controller,
                float(hand_now["x"]),
                float(hand_now["y"]),
                float(hand_now["z"]),
                target_id,
                f"{tag}_HAND",
            )
            if ok_ms_h:
                success = True
                break
            if err_ms_h:
                last_err = err_ms_h
            # se abbiamo toccato l'oggetto, prova una micro-search centrata sulla mano (spesso la mela si sposta)
            in_pick, in_touch = is_target_in_range(controller, target_id)
            if in_touch:
                hand_now = get_hand_pos(controller)
                ok_ms2, err_ms2 = micro_search_and_pick(
                    controller, float(hand_now["x"]), float(hand_now["y"]), float(hand_now["z"]), target_id, f"{tag}_HAND"
                )
                if ok_ms2:
                    success = True
                    break
                if err_ms2:
                    last_err = err_ms2

        else:
            # side: pre-hover a distanza pre_dist lungo approach_dir, poi path diagonale
            pre_x = cand_x + ax * pre_dist
            pre_z = cand_z + az * pre_dist
            ok_p, used_y, err_p = reach_over_counter(controller, pre_x, pre_z, hover_y, tag, "PRE")
            if not ok_p:
                last_err = err_p
                continue

            ok_d, err_d = descend_with_checks(
                controller,
                start=vec(pre_x, used_y, pre_z),
                end=vec(cand_x, final_y, cand_z),
                steps=16,
                target_id=target_id,
                attempt_tag=tag,
            )
            if ok_d:
                success = True
                break
            if err_d:
                last_err = err_d
                continue
            ok_ms, err_ms = micro_search_and_pick(controller, cand_x, final_y, cand_z, target_id, tag)
            if ok_ms:
                success = True
                break
            if err_ms:
                last_err = err_ms
            hand_now = get_hand_pos(controller)
            ok_ms_h, err_ms_h = micro_search_and_pick(
                controller,
                float(hand_now["x"]),
                float(hand_now["y"]),
                float(hand_now["z"]),
                target_id,
                f"{tag}_HAND",
            )
            if ok_ms_h:
                success = True
                break
            if err_ms_h:
                last_err = err_ms_h
            in_pick, in_touch = is_target_in_range(controller, target_id)
            if in_touch:
                hand_now = get_hand_pos(controller)
                ok_ms2, err_ms2 = micro_search_and_pick(
                    controller, float(hand_now["x"]), float(hand_now["y"]), float(hand_now["z"]), target_id, f"{tag}_HAND"
                )
                if ok_ms2:
                    success = True
                    break
                if err_ms2:
                    last_err = err_ms2

        # ultimo tentativo pickup
        for t in range(1, 4):
            open_gripper(controller)
            ok_p, err_p = pickup_object(controller, target_id)
            save_frame(controller, f"{tag}_PICK_{t:02d}.png")
            if ok_p:
                success = True
                break
            last_err = err_p
        if success:
            break

    save_frame(controller, "999_END.png")
    if success:
        ok(f"[ARM] Pickup riuscito: {target_type} ({target_id})")
    else:
        warn("[ARM] Pickup fallito.")
        if last_err:
            warn("Last error:", last_err)
        warn("Frames:", FRAMES_DIR.resolve())

    # stato finale per runner/loop
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "success": bool(success),
                "target": {"objectId": target_id, "objectType": target_type},
                "last_error": last_err,
                "frames_dir": str(FRAMES_DIR.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    controller.stop()


if __name__ == "__main__":
    main()
