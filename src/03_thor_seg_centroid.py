from common import EnvConfig, make_controller, get_rgb_bgr, data_dir, print_ok, print_warn, load_json,save_json
import numpy as np
import cv2

def centroid_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())

def main():
    state = load_json("teleport_state.json")

    scene = state["scene"]
    target_id = state["target"]["objectId"]
    target_type = state["target"]["objectType"]
    agent_pos = state["agent"]["position"]
    agent_rot = state["agent"]["rotation"]
    horizon = state["agent"]["horizon"]

    cfg = EnvConfig(scene=scene, render_instance_segmentation=True)
    controller = make_controller(cfg)

    # ripristina posa agente esattamente come in 02
    controller.step(
        action="Teleport",
        position=agent_pos,
        rotation=agent_rot,
        horizon=horizon
    )

    event = controller.last_event
    masks = event.instance_masks

    if masks is None or target_id not in masks:
        print_warn(f"Instance mask non disponibile per {target_type} ({target_id}).")
        controller.stop()
        return

    mask = masks[target_id].astype(np.uint8) * 255
    c = centroid_from_mask(mask)
    if c is None:
        print_warn("Maschera vuota.")
        controller.stop()
        return

    cx, cy = c

    rgb_bgr = get_rgb_bgr(event)
    overlay = rgb_bgr.copy()
    colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(overlay, 1.0, colored, 0.3, 0)
    cv2.circle(overlay, (cx, cy), 6, (0, 255, 0), -1)

    out = data_dir("outputs") / "03_thor_centroid.png"
    cv2.imwrite(str(out), overlay)

    print_ok(f"Target: {target_type} ({target_id})")
    print_ok(f"Centroide (pixel): x={cx}, y={cy}")
    save_json(
    {
        "centroid": {"x": cx, "y": cy},
        "target": {"objectId": target_id, "objectType": target_type}
    },
    "centroid_state.json"
)
    print_ok(f"Debug salvato in {out}")

    controller.stop()

if __name__ == "__main__":
    main()
