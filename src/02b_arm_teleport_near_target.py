from common import EnvConfig, make_controller, load_json, save_json, print_ok, print_warn
import math

def dist_xz(p, q):
    return math.sqrt((p["x"]-q["x"])**2 + (p["z"]-q["z"])**2)

def yaw_to_face(agent_pos, obj_pos):
    dx = obj_pos["x"] - agent_pos["x"]
    dz = obj_pos["z"] - agent_pos["z"]
    return math.degrees(math.atan2(dx, dz))

def main():
    tele = load_json("teleport_state.json")
    scene = tele["scene"]
    target_id = tele["target"]["objectId"]
    target_type = tele["target"]["objectType"]

    # 1) In default leggiamo bbox center (ground truth)
    ctrl = make_controller(EnvConfig(scene=scene, agent_mode="default"))
    objs = ctrl.last_event.metadata.get("objects", [])
    obj = next((o for o in objs if o.get("objectId") == target_id), None)
    if not obj:
        print_warn("Target non trovato in metadata.")
        ctrl.stop()
        return

    aabb = obj.get("axisAlignedBoundingBox") or obj.get("boundingBox")
    if not aabb or "center" not in aabb:
        print_warn("BBox center non disponibile.")
        ctrl.stop()
        return

    center = aabb["center"]  # world
    ctrl.stop()

    # 2) In arm mode scegliamo una reachable position VICINA al center
    controller = make_controller(EnvConfig(scene=scene, agent_mode="arm"))
    controller.step("GetReachablePositions")
    reachable = controller.last_event.metadata.get("actionReturn", [])
    if not reachable:
        print_warn("GetReachablePositions vuoto.")
        controller.stop()
        return

    # pick: la reachable più vicina in XZ al bbox center
    best = min(reachable, key=lambda p: dist_xz(p, center))
    d = dist_xz(best, center)

    # ruota per guardare l’oggetto
    yaw = yaw_to_face(best, center)

    controller.step(
        action="Teleport",
        position=best,
        rotation={"x": 0, "y": yaw, "z": 0},
        horizon=tele["agent"]["horizon"]
    )

    arm_pose = {
        "scene": scene,
        "target": {"objectId": target_id, "objectType": target_type},
        "agent": {
            "position": controller.last_event.metadata["agent"]["position"],
            "rotation": controller.last_event.metadata["agent"]["rotation"],
            "horizon": controller.last_event.metadata["agent"]["cameraHorizon"],
        },
        "bbox_center_world": center,
        "xz_distance_to_target": d
    }

    save_json(arm_pose, "arm_teleport_state.json")
    print_ok(f"Salvato arm_teleport_state.json | dist_xz={d:.3f} m")
    controller.stop()

if __name__ == "__main__":
    main()
