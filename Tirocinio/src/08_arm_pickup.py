from common import EnvConfig, make_controller, load_json, print_ok, print_warn

def main():
    tele = load_json("teleport_state.json")

    scene = tele["scene"]
    target_id = tele["target"]["objectId"]
    target_type = tele["target"]["objectType"]
    agent_pos = tele["agent"]["position"]
    agent_rot = tele["agent"]["rotation"]
    horizon = tele["agent"]["horizon"]

    # ARM MODE
    controller = make_controller(EnvConfig(scene=scene, agent_mode="arm"))

    # Ripristina posa (come Step 02)
    controller.step(action="Teleport", position=agent_pos, rotation=agent_rot, horizon=horizon)

    # Prova pickup in arm mode
    try:
        ev = controller.step(action="PickupObject", objectId=target_id, forceAction=True)
    except ValueError as e:
        print_warn(f"PickupObject ha lanciato errore: {e}")
        controller.stop()
        return

    ok = ev.metadata.get("lastActionSuccess", False)
    err = ev.metadata.get("errorMessage", "")

    if ok:
        print_ok(f"[ARM] Pickup riuscito ✅  {target_type} ({target_id})")
        # debug utile
        inv = ev.metadata.get("inventoryObjects", [])
        held = ev.metadata.get("heldObjectPose", None)
        print_ok(f"inventoryObjects: {len(inv)} | heldObjectPose: {'yes' if held else 'no'}")
    else:
        print_warn(f"[ARM] Pickup fallito ❌  {target_type} ({target_id})")
        print_warn(f"errorMessage: {err}")

    controller.stop()

if __name__ == "__main__":
    main()
