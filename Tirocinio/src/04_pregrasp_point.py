from common import EnvConfig, make_controller, load_json, save_json, print_ok, print_warn
import numpy as np
import math

def intrinsics_from_fov(width, height, fov_deg):
    fov = math.radians(fov_deg)
    fx = (width / 2.0) / math.tan(fov / 2.0)
    fy = fx  # pixel square in THOR
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

def rot_yaw_deg(yaw_deg):
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    # right-handed, Y up, Z forward
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)

def rot_pitch_deg(pitch_deg):
    # pitch around X (right axis). pitch>0 looks DOWN in our convention
    a = math.radians(pitch_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ], dtype=np.float32)

def main():
    tele = load_json("teleport_state.json")
    cen = load_json("centroid_state.json")

    scene = tele["scene"]
    agent_pos = tele["agent"]["position"]
    agent_rot = tele["agent"]["rotation"]
    horizon = tele["agent"]["horizon"]

    u = int(cen["centroid"]["x"])
    v = int(cen["centroid"]["y"])

    # Controller con DEPTH abilitata
    cfg = EnvConfig(scene=scene, render_depth=True, render_instance_segmentation=False)
    controller = make_controller(cfg)

    # ripristina posa identica a 02
    controller.step(
        action="Teleport",
        position=agent_pos,
        rotation=agent_rot,
        horizon=horizon
    )

    event = controller.last_event

    # depth_frame: (H,W) in metri
    if event.depth_frame is None:
        print_warn("depth_frame è None: assicurati renderDepthImage=True nel controller.")
        controller.stop()
        return

    H, W = event.depth_frame.shape[:2]
    if not (0 <= u < W and 0 <= v < H):
        print_warn(f"Centroide fuori immagine: (u={u}, v={v}) con W={W}, H={H}")
        controller.stop()
        return

    d = float(event.depth_frame[v, u])
    if not np.isfinite(d) or d <= 0:
        print_warn(f"Depth non valida nel pixel (u={u}, v={v}): d={d}")
        controller.stop()
        return

    # intrinsics da FOV/risoluzione
    fx, fy, cx, cy = intrinsics_from_fov(W, H, cfg.fov)

    # backproject (camera coords)
    # camera axes: X right, Y up, Z forward
    x_cam = (u - cx) / fx * d
    y_cam = -(v - cy) / fy * d  # v cresce verso il basso => Y up è negativo
    z_cam = d
    p_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float32)

    # camera pose in world
    meta_agent = event.metadata.get("agent", {})
    cam_pos = meta_agent.get("cameraPosition", None)
    if cam_pos is None:
        # fallback (di solito cameraPosition esiste, ma mettiamo un ripiego)
        cam_pos = meta_agent.get("position", agent_pos)

    cam_pos_vec = np.array([cam_pos["x"], cam_pos["y"], cam_pos["z"]], dtype=np.float32)

    yaw = float(meta_agent.get("rotation", agent_rot)["y"])
    pitch = float(meta_agent.get("cameraHorizon", horizon))  # >0 guarda verso il basso

    R = rot_yaw_deg(yaw) @ rot_pitch_deg(pitch)  # camera->world
    p_world = cam_pos_vec + (R @ p_cam)

    out = {
        "scene": scene,
        "target": tele["target"],
        "centroid_px": {"x": u, "y": v},
        "depth_m": d,
        "camera": {
            "position": cam_pos,
            "yaw_deg": yaw,
            "horizon_deg": pitch
        },
        "pregrasp_world": {"x": float(p_world[0]), "y": float(p_world[1]), "z": float(p_world[2])}
    }

    save_json(out, "pregrasp_point.json")

    print_ok(f"Centroide: (u={u}, v={v}) | depth={d:.3f} m")
    print_ok(f"Pregrasp WORLD: x={p_world[0]:.3f}, y={p_world[1]:.3f}, z={p_world[2]:.3f}")
    print_ok("Salvato in data/state/pregrasp_point.json")

    controller.stop()

if __name__ == "__main__":
    main()
