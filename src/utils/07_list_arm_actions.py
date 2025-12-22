from common import EnvConfig, make_controller

# Per ogni action, mettiamo un payload minimo (quando serve).
CANDIDATES = [
    ("OpenGripper", {}),
    ("CloseGripper", {}),
    ("OpenHand", {}),
    ("CloseHand", {}),
    ("ReleaseObject", {}),
    ("DropHeldObject", {}),
    ("ThrowHeldObject", {}),
    ("PickupObject", {"objectId": "DUMMY"}),   # dummy -> ci serve solo per capire se l'action esiste
    ("PutObject", {"objectId": "DUMMY"}),

    # già sappiamo che MoveArm esiste, ma lasciamo per completezza
    ("MoveArm", {"position": {"x": 0.02, "y": 0.0, "z": 0.02}, "speed": 1}),
    ("MoveArmBase", {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}),
    ("MoveHand", {"position": {"x": 0.02, "y": 0.0, "z": 0.02}}),
    ("RotateWrist", {"rotation": {"x": 0.0, "y": 10.0, "z": 0.0}}),
    ("RotateWristRelative", {"rotation": {"x": 0.0, "y": 10.0, "z": 0.0}}),
]

def try_action(controller, name, payload):
    try:
        ev = controller.step(action=name, **payload)
        return ("SUPPORTED", ev.metadata.get("lastActionSuccess", False), ev.metadata.get("errorMessage", ""))
    except ValueError as e:
        msg = str(e)
        if "Invalid action" in msg:
            return ("INVALID", False, msg)
        return ("ERROR", False, msg)

def main():
    controller = make_controller(EnvConfig(scene="FloorPlan1", agent_mode="arm"))
    print("Probe actions (arm mode)\n")

    supported = []
    for name, payload in CANDIDATES:
        status, ok, err = try_action(controller, name, payload)
        if status == "SUPPORTED":
            supported.append((name, ok, err))
            print(f"[SUPPORTED] {name} | success={ok} | err={err}")
        else:
            print(f"[{status}] {name} | {err}")

    print("\n---- SUMMARY (SUPPORTED) ----")
    for name, ok, err in supported:
        print(f"- {name}")

    controller.stop()

if __name__ == "__main__":
    main()
