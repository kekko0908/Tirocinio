import argparse
import math

from common import EnvConfig, make_controller, get_rgb_bgr, save_rgb, print_ok, print_warn, save_json

def dist_xz(a, b):
    return math.sqrt((a["x"] - b["x"])**2 + (a["z"] - b["z"])**2)

def find_best_reachable_near_object(reachable_positions, obj_pos):
    return min(reachable_positions, key=lambda p: dist_xz(p, obj_pos))

def yaw_to_face(agent_pos, obj_pos):
    dx = obj_pos["x"] - agent_pos["x"]
    dz = obj_pos["z"] - agent_pos["z"]
    return math.degrees(math.atan2(dx, dz))

def teleport_face_object(controller, target_pos, obj_pos, horizon=20):
    controller.step(
        action="Teleport",
        position=dict(x=target_pos["x"], y=target_pos["y"], z=target_pos["z"]),
        rotation=dict(x=0, y=0, z=0),
        horizon=horizon
    )
    agent_pos = controller.last_event.metadata["agent"]["position"]
    yaw = yaw_to_face(agent_pos, obj_pos)
    controller.step(
        action="Teleport",
        position=dict(x=target_pos["x"], y=target_pos["y"], z=target_pos["z"]),
        rotation=dict(x=0, y=yaw, z=0),
        horizon=horizon
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="FloorPlan1")
    ap.add_argument("--target_type", default="", help="Se vuoto sceglie il primo pickable visibile.")
    ap.add_argument("--max_pickables", type=int, default=50)
    ap.add_argument("--horizon", type=float, default=20.0)
    ap.add_argument("--out_state", default="teleport_state.json", help="Output in data/state/")
    ap.add_argument("--out_frame", default="02_teleport_rgb.png", help="Output in data/frames/")
    args = ap.parse_args()

    scene = args.scene
    controller = make_controller(EnvConfig(scene=scene))

    meta = controller.last_event.metadata
    objects = meta.get("objects", [])
    pickable = [o for o in objects if o.get("pickupable", False)]
    if not pickable:
        print_warn("Nessun oggetto pickupable trovato.")
        controller.stop()
        return

    controller.step("GetReachablePositions")
    reachable = controller.last_event.metadata.get("actionReturn", [])
    if not reachable:
        print_warn("GetReachablePositions vuoto.")
        controller.stop()
        return

    chosen_pose = None
    want_type = args.target_type.strip().lower()

    for obj in pickable[: max(1, int(args.max_pickables))]:
        obj_id = obj["objectId"]
        obj_type = obj["objectType"]
        obj_pos = obj["position"]

        if want_type and str(obj_type).lower() != want_type:
            continue

        best_pos = find_best_reachable_near_object(reachable, obj_pos)
        teleport_face_object(controller, best_pos, obj_pos, horizon=float(args.horizon))

        objs_after = controller.last_event.metadata.get("objects", [])
        target_after = next((x for x in objs_after if x["objectId"] == obj_id), None)

        if target_after and target_after.get("visible", False):
            agent_pos = controller.last_event.metadata["agent"]["position"]
            agent_rot = controller.last_event.metadata["agent"]["rotation"]
            horizon = controller.last_event.metadata["agent"]["cameraHorizon"]
            chosen_pose = {
                "scene": scene,
                "target": {"objectId": obj_id, "objectType": obj_type},
                "agent": {"position": agent_pos, "rotation": agent_rot, "horizon": horizon},
            }
            break

    frame = get_rgb_bgr(controller.last_event)
    save_rgb(frame, str(args.out_frame))

    if chosen_pose:
        path = save_json(chosen_pose, str(args.out_state))
        print_ok(
            f"Target visibile: {chosen_pose['target']['objectType']} ({chosen_pose['target']['objectId']})"
        )
        print_ok(f"Stato salvato in {path}")
    else:
        if want_type:
            print_warn(f"Non ho trovato un pickable visibile di tipo: {args.target_type}")
        else:
            print_warn(f"Non ho trovato un pickable visibile nei primi {args.max_pickables}.")

    controller.stop()

if __name__ == "__main__":
    main()
