from common import EnvConfig, make_controller, load_json, print_ok, print_warn
import math
from pathlib import Path
from PIL import Image


# root del progetto (TirocinioV2)
ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = ROOT / "data" / "frames" / "05_pickup"


def save_all_frames(event, stem: str):
    """
    Salva frame ego + third-party (se presenti) da un EVENT (ritornato da controller.step).
    """
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Ego
    try:
        Image.fromarray(event.frame).save(FRAMES_DIR / f"{stem}_ego.png")
    except Exception as e:
        print_warn(f"Impossibile salvare ego frame: {e}")

    # Third party
    tpf = getattr(event, "third_party_camera_frames", None)
    if tpf is None or not isinstance(tpf, (list, tuple)) or len(tpf) == 0:
        print_warn("third_party_camera_frames non disponibile/vuoto: salvo solo ego.")
        return

    # salva fino a 2 camere
    for i in range(min(2, len(tpf))):
        try:
            Image.fromarray(tpf[i]).save(FRAMES_DIR / f"{stem}_tp{i}.png")
        except Exception as e:
            print_warn(f"Impossibile salvare tp{i}: {e}")


def yaw_to_face(agent_pos, obj_pos):
    dx = obj_pos["x"] - agent_pos["x"]
    dz = obj_pos["z"] - agent_pos["z"]
    return math.degrees(math.atan2(dx, dz))


def main():
    tele = load_json("teleport_state.json")

    scene = tele["scene"]
    target_id = tele["target"]["objectId"]
    target_type = tele["target"]["objectType"]
    agent_pos = tele["agent"]["position"]
    agent_rot = tele["agent"]["rotation"]
    horizon = tele["agent"]["horizon"]

    controller = make_controller(EnvConfig(scene=scene, render_depth=False))

    # ripristina posa come Step 02 (meglio TeleportFull per coerenza con arm mode)
    ev = controller.step(
        action="TeleportFull",
        x=float(agent_pos["x"]),
        y=float(agent_pos["y"]),
        z=float(agent_pos["z"]),
        rotation={
            "x": float(agent_rot.get("x", 0.0)),
            "y": float(agent_rot.get("y", 0.0)),
            "z": float(agent_rot.get("z", 0.0)),
        },
        horizon=float(horizon),
        standing=True
    )

    # ✅ aggiungi due camere esterne (fixed) - come nel tuo esempio
    # Camera 0: laterale
    ev = controller.step(
    action="AddThirdPartyCamera",
    position={"x": 0.25, "y": 1.7, "z": -2.2},   # dietro e un po' in alto
    rotation={"x": 15, "y": 0, "z": 0},          # guarda leggermente in basso
    fieldOfView=90
)

    save_all_frames(ev, "00_restored_pose")

    # recupera info target dalla metadata corrente
    objs = ev.metadata.get("objects", [])
    target = next((o for o in objs if o["objectId"] == target_id), None)

    if not target:
        print_warn(f"Target non trovato nella metadata: {target_type} ({target_id})")
        save_all_frames(ev, "99_target_not_found")
        controller.stop()
        return

    # se non visibile, ruota verso target usando posizione (no LookAtObject)
    if not target.get("visible", False):
        agent_pos_now = ev.metadata["agent"]["position"]
        yaw = yaw_to_face(agent_pos_now, target["position"])

        ev = controller.step(
            action="TeleportFull",
            x=float(agent_pos_now["x"]),
            y=float(agent_pos_now["y"]),
            z=float(agent_pos_now["z"]),
            rotation={"x": 0, "y": float(yaw), "z": 0},
            horizon=float(horizon),
            standing=True
        )
        save_all_frames(ev, "01_after_yaw_to_face")

    # frame appena prima dei tentativi di pickup
    save_all_frames(ev, "02_before_pickup")

    max_tries = 8
    for i in range(max_tries):
        save_all_frames(ev, f"10_try_{i+1:02d}_before")

        try:
            ev = controller.step(action="PickupObject", objectId=target_id, forceAction=True)
        except ValueError as e:
            print_warn(f"PickupObject non valido/errore: {e}")
            save_all_frames(controller.last_event, f"98_exception_try_{i+1:02d}")
            controller.stop()
            return

        save_all_frames(ev, f"11_try_{i+1:02d}_after")

        if ev.metadata.get("lastActionSuccess", False):
            print_ok(f"Pickup riuscito ✅  {target_type} ({target_id})")
            save_all_frames(ev, "20_pickup_success")
            controller.stop()
            return

        err = ev.metadata.get("errorMessage", "unknown error")
        print_warn(f"Tentativo {i+1}/{max_tries} fallito: {err}")

        ev = controller.step(action="MoveAhead", moveMagnitude=0.15)
        save_all_frames(ev, f"12_after_moveahead_{i+1:02d}")

    print_warn("Pickup fallito dopo i retry. Probabile: distanza/occlusione/angolo non buono.")
    save_all_frames(ev, "30_pickup_failed")
    controller.stop()


if __name__ == "__main__":
    main()
