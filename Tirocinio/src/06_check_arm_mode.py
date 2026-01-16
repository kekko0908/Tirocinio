from common import EnvConfig, make_controller

def safe_print_dict(d, title, keys=None):
    print(f"\n--- {title} ---")
    if d is None:
        print("None")
        return
    if keys is None:
        keys = list(d.keys())
    for k in keys:
        if k in d:
            print(f"{k}: {d[k]}")

def main():
    cfg = EnvConfig(scene="FloorPlan1", agent_mode="arm")
    controller = make_controller(cfg)

    md = controller.last_event.metadata
    print("Top-level metadata keys:", list(md.keys()))

    agent = md.get("agent", {})
    safe_print_dict(agent, "agent metadata (sample)", keys=["position", "rotation", "cameraHorizon", "cameraPosition"])

    # 1) Prova un'azione tipica del braccio: se è valida -> arm mode (o comunque arm actions abilitate)
    test_action = controller.step(
        action="MoveArm",
        position={"x": 0.02, "y": 0.0, "z": 0.02},
        speed=1
    )

    ok = test_action.metadata.get("lastActionSuccess", False)
    err = test_action.metadata.get("errorMessage", "")

    print("\n--- MoveArm test ---")
    print("lastActionSuccess:", ok)
    if not ok:
        print("errorMessage:", err)

    # 2) Prova gripper (se esiste)
    grip = controller.step(action="OpenGripper")
    ok2 = grip.metadata.get("lastActionSuccess", False)
    err2 = grip.metadata.get("errorMessage", "")
    print("\n--- OpenGripper test ---")
    print("lastActionSuccess:", ok2)
    if not ok2:
        print("errorMessage:", err2)

    controller.stop()

if __name__ == "__main__":
    main()
