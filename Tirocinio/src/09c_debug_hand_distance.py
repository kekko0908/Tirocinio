from common import EnvConfig, make_controller, load_json, print_ok, print_warn
import json

def main():
    tele = load_json("teleport_state.json")
    bbox = load_json("grasp_point_bbox.json")

    scene = tele["scene"]
    agent_pos = tele["agent"]["position"]
    agent_rot = tele["agent"]["rotation"]
    horizon = tele["agent"]["horizon"]

    center = bbox["bbox_center_world"]

    controller = make_controller(EnvConfig(scene=scene, agent_mode="arm"))
    controller.step(action="Teleport", position=agent_pos, rotation=agent_rot, horizon=horizon)

    md = controller.last_event.metadata
    arm = md.get("arm", {})

    print_ok(f"bbox center: {center}")
    print_ok(f"arm keys: {list(arm.keys())}")

    # prova alcune path comuni
    candidates = [
        ("arm['hand']", arm.get("hand")),
        ("arm['handState']", arm.get("handState")),
        ("arm['handPosition']", arm.get("handPosition")),
        ("arm['heldObjectPose']", md.get("heldObjectPose")),
        ("arm (raw excerpt)", arm),
    ]

    for name, val in candidates:
        print("\n---", name, "---")
        if isinstance(val, (dict, list)):
            print(json.dumps(val, indent=2)[:2000])  # limita output
        else:
            print(val)

    controller.stop()

if __name__ == "__main__":
    main()
