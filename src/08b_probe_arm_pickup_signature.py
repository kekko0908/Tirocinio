from common import EnvConfig, make_controller, load_json, print_ok, print_warn

def try_call(controller, label, **kwargs):
    try:
        ev = controller.step(action="PickupObject", **kwargs)
        ok = ev.metadata.get("lastActionSuccess", False)
        err = ev.metadata.get("errorMessage", "")
        print(f"[TRY] {label} -> success={ok} | err={err}")
        return True
    except ValueError as e:
        print(f"[TRY] {label} -> ValueError: {e}")
        return False

def main():
    tele = load_json("teleport_state.json")
    scene = tele["scene"]
    agent_pos = tele["agent"]["position"]
    agent_rot = tele["agent"]["rotation"]
    horizon = tele["agent"]["horizon"]

    controller = make_controller(EnvConfig(scene=scene, agent_mode="arm"))
    controller.step(action="Teleport", position=agent_pos, rotation=agent_rot, horizon=horizon)

    # PROBE: varie firme possibili
    try_call(controller, "no-args")
    try_call(controller, "forceAction", forceAction=True)
    try_call(controller, "manualInteract", manualInteract=True)
    try_call(controller, "forceAction+manualInteract", forceAction=True, manualInteract=True)

    controller.stop()

if __name__ == "__main__":
    main()
