from common import EnvConfig, make_controller, load_json, print_ok, print_warn
import math

def main():
    tele = load_json("teleport_state.json")
    bbox = load_json("grasp_point_bbox.json")

    scene = tele["scene"]
    agent_pos = tele["agent"]["position"]
    agent_rot = tele["agent"]["rotation"]
    horizon = tele["agent"]["horizon"]

    c = bbox["bbox_center_world"]
    cx, cy, cz = float(c["x"]), float(c["y"]), float(c["z"])

    controller = make_controller(EnvConfig(scene=scene, agent_mode="arm"))
    controller.step(action="Teleport", position=agent_pos, rotation=agent_rot, horizon=horizon)

    # approccio: prima sopra, poi verso il centro (così eviti collisioni)
    approach = [
        {"x": cx, "y": cy + 0.10, "z": cz},
        {"x": cx, "y": cy + 0.05, "z": cz},
        {"x": cx, "y": cy + 0.02, "z": cz},
        {"x": cx, "y": cy,        "z": cz},
    ]

    # piccoli offset laterali nel caso il centro non coincida col punto afferrabile
    offsets = [
        (0.00, 0.00, 0.00),
        (0.02, 0.00, 0.00),
        (-0.02, 0.00, 0.00),
        (0.00, 0.00, 0.02),
        (0.00, 0.00, -0.02),
    ]

    for oi, (dx, dy, dz) in enumerate(offsets, start=1):
        print_ok(f"=== Offset set {oi}/{len(offsets)}: dx={dx},dy={dy},dz={dz} ===")

        for si, base in enumerate(approach, start=1):
            goal = {"x": base["x"] + dx, "y": base["y"] + dy, "z": base["z"] + dz}
            print_ok(f"[{si}/{len(approach)}] MoveArm WORLD: {goal}")

            ev = controller.step(
                action="MoveArm",
                position=goal,
                coordinateSpace="world",
                restrictMovement=False,
                speed=1,
                returnToStart=False,
                fixedDeltaTime=0.02
            )

            if not ev.metadata.get("lastActionSuccess", False):
                print_warn("MoveArm fallito: " + ev.metadata.get("errorMessage", ""))
                break

        # tenta pickup dopo la sequenza di avvicinamento
        try:
            pick = controller.step(action="PickupObject")  # arm signature: no objectId
        except ValueError as e:
            print_warn(f"PickupObject ValueError: {e}")
            continue

        ok = pick.metadata.get("lastActionSuccess", False)
        err = pick.metadata.get("errorMessage", "")

        if ok:
            print_ok("[ARM] Pickup riuscito ✅")
            inv = pick.metadata.get("inventoryObjects", [])
            held = pick.metadata.get("heldObjectPose", None)
            print_ok(f"inventoryObjects: {len(inv)} | heldObjectPose: {'yes' if held else 'no'}")
            controller.stop()
            return
        else:
            print_warn("[ARM] Pickup fallito ❌: " + err)

    print_warn("Pickup ARM fallito anche con bbox center + offsets.")
    controller.stop()

if __name__ == "__main__":
    main()
