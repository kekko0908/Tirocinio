from common import EnvConfig, make_controller, print_ok

def main():
    controller = make_controller(EnvConfig(scene="FloorPlan1"))
    event = controller.step("GetReachablePositions")  # non obbligatorio, ma utile

    meta = controller.last_event.metadata
    objects = meta.get("objects", [])

    pickable = [o for o in objects if o.get("pickupable", False)]
    pickable_sorted = sorted(pickable, key=lambda x: x.get("objectType", ""))

    print_ok(f"Pickable trovati: {len(pickable_sorted)}\n")
    for o in pickable_sorted[:50]:
        print(f"- {o.get('objectType')}  | id={o.get('objectId')}")

    if len(pickable_sorted) > 50:
        print("\n... (mostrati solo i primi 50)")

    controller.stop()

if __name__ == "__main__":
    main()
