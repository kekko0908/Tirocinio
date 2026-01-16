import argparse

from common import EnvConfig, make_controller, load_json, save_json, print_ok, print_warn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from_pregrasp",
        default="",
        help="Se impostato, usa data/state/pregrasp_point.json (o path relativo data/state/) come bbox_center_world (sensor-based).",
    )
    ap.add_argument("--out", default="grasp_point_bbox.json", help="Output in data/state/")
    args = ap.parse_args()

    tele = load_json("teleport_state.json")
    scene = tele["scene"]

    # --- sensor-based center from pregrasp_point.json ---
    if args.from_pregrasp:
        pre = load_json(args.from_pregrasp)
        pw = pre.get("pregrasp_world") if isinstance(pre, dict) else None
        if not (isinstance(pw, dict) and all(k in pw for k in ("x", "y", "z"))):
            print_warn("pregrasp_point.json non contiene pregrasp_world valido.")
            return

        # bbox_size non disponibile con sensor-only: usa un default (mela ~ 12cm) o lascia None
        out = {
            "scene": scene,
            "target": tele["target"],
            "bbox_center_world": {"x": float(pw["x"]), "y": float(pw["y"]), "z": float(pw["z"])},
            "bbox_size": {"x": 0.12, "y": 0.12, "z": 0.12},
            "source": "pregrasp_world",
        }
        save_json(out, args.out)
        print_ok(f"[SENSOR] BBox center WORLD (from pregrasp): {out['bbox_center_world']}")
        print_ok(f"Salvato in data/state/{args.out}")
        return

    # --- GT center from metadata ---
    target_id = tele["target"]["objectId"]
    agent_pos = tele["agent"]["position"]
    agent_rot = tele["agent"]["rotation"]
    horizon = tele["agent"]["horizon"]

    controller = make_controller(EnvConfig(scene=scene, agent_mode="default"))
    controller.step(action="Teleport", position=agent_pos, rotation=agent_rot, horizon=horizon)

    objs = controller.last_event.metadata.get("objects", [])
    obj = next((o for o in objs if o.get("objectId") == target_id), None)

    if not obj:
        print_warn(f"Target {target_id} non trovato in metadata.")
        controller.stop()
        return

    aabb = obj.get("axisAlignedBoundingBox") or obj.get("boundingBox")
    if not aabb or "center" not in aabb:
        print_warn("Nessuna axisAlignedBoundingBox/boundingBox.center disponibile.")
        controller.stop()
        return

    center = aabb["center"]  # world coords
    size = aabb.get("size", None)

    out = {
        "scene": scene,
        "target": tele["target"],
        "bbox_center_world": center,
        "bbox_size": size,
        "source": "metadata_aabb",
    }
    save_json(out, args.out)

    print_ok(f"[GT] BBox center WORLD: {center}")
    if size:
        print_ok(f"BBox size: {size}")
    print_ok(f"Salvato in data/state/{args.out}")

    controller.stop()

if __name__ == "__main__":
    main()
