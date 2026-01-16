import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ai2thor.controller import Controller

# Debug: overlay dei punti raggiungibili direttamente sulla vista ego.


def project_point_manual(point_3d, camera_pos, camera_rot, fov_deg, image_width, image_height):
    """
    Proietta un punto 3D su immagine usando modello pinhole semplificato.
    Usa yaw/pitch della camera per la trasformazione.
    """
    dx = point_3d[0] - camera_pos[0]
    dy = point_3d[1] - camera_pos[1]
    dz = point_3d[2] - camera_pos[2]

    yaw = np.deg2rad(float(camera_rot[1]))
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    x1 = cos_y * dx - sin_y * dz
    z1 = sin_y * dx + cos_y * dz
    y1 = dy

    pitch = np.deg2rad(float(camera_rot[0]))
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)
    y2 = cos_p * y1 - sin_p * z1
    z2 = sin_p * y1 + cos_p * z1
    x2 = x1

    if z2 <= 0.1:
        return None

    fov = np.deg2rad(float(fov_deg))
    focal_length = (image_width / 2.0) / np.tan(fov / 2.0)
    u = (x2 / z2) * focal_length + (image_width / 2.0)
    v = -(y2 / z2) * focal_length + (image_height / 2.0)

    if 0 <= u < image_width and 0 <= v < image_height:
        return (int(u), int(v))
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Definisce parametri CLI per l'overlay ego-view.
    Include scena, densita' e output di debug.
    """
    ap = argparse.ArgumentParser(description="Overlay reachable points on agent view.")
    ap.add_argument("--scene", default="FloorPlan10")
    ap.add_argument("--grid_size", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--visibility_distance", type=float, default=3.0)
    ap.add_argument("--rotate_degrees", type=int, default=45)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--save_path", default="overlay_debug.png", help="Optional path to save PNG.")
    ap.add_argument("--alpha", type=float, default=0.5)
    return ap


def main() -> None:
    """
    Entry point: calcola reachable, proietta e salva/mostra.
    Genera overlay direttamente sulla vista ego del robot.
    """
    args = build_arg_parser().parse_args()
    controller = Controller(
        agentMode="default",
        visibilityDistance=float(args.visibility_distance),
        scene=args.scene,
        gridSize=float(args.grid_size),
        renderDepthImage=True,
        fieldOfView=float(args.fov),
        width=int(args.width),
        height=int(args.height),
    )

    event_map = controller.step(action="GetReachablePositions")
    reachable_positions = event_map.metadata.get("actionReturn", []) or []

    controller.step(action="RotateLeft", degrees=int(args.rotate_degrees))
    controller.step(action="MoveBack")
    event = controller.step(action="LookDown", degrees=30)
    base_frame = event.frame.copy()
    overlay = base_frame.copy()

    color_reachable = (255, 100, 0)  # BGR
    color_agent = (0, 0, 255)
    color_agent_center = (255, 255, 255)

    agent_meta = event.metadata.get("agent") or {}
    agent_pos = agent_meta.get("position") or {}
    agent_rot = agent_meta.get("rotation") or {}
    horizon_deg = agent_meta.get("cameraHorizon", 0.0)
    cam_pos = event.metadata.get("cameraPosition") or agent_pos

    def project_with_fallbacks(point_3d):
        """
        Prova piu' combinazioni camera/pose per robustezza.
        Ritorna il primo pixel valido oppure None.
        """
        cam_tuple = (cam_pos.get("x", 0.0), cam_pos.get("y", 0.0), cam_pos.get("z", 0.0))
        agent_tuple = (agent_pos.get("x", 0.0), agent_pos.get("y", 0.0), agent_pos.get("z", 0.0))
        for pitch in (horizon_deg, -horizon_deg):
            pixel = project_point_manual(
                point_3d,
                cam_tuple,
                (pitch, agent_rot.get("y", 0.0), 0.0),
                args.fov,
                args.width,
                args.height,
            )
            if pixel:
                return pixel
        for pitch in (horizon_deg, -horizon_deg):
            pixel = project_point_manual(
                point_3d,
                agent_tuple,
                (pitch, agent_rot.get("y", 0.0), 0.0),
                args.fov,
                args.width,
                args.height,
            )
            if pixel:
                return pixel
        return None

    projected_count = 0
    for point in reachable_positions:
        point_y = point.get("y")
        if point_y is None:
            point_y = float(agent_pos.get("y", 0.0)) - 0.9
        point_3d = np.array([point["x"], point_y, point["z"]])
        pixel_coords = project_with_fallbacks(point_3d)
        if pixel_coords:
            projected_count += 1
            cv2.circle(overlay, pixel_coords, radius=2, color=color_reachable, thickness=-1)

    agent_pos_3d = np.array(
        [
            float(agent_pos.get("x", 0.0)),
            float(agent_pos.get("y", 0.0)) - 1.0,
            float(agent_pos.get("z", 0.0)),
        ]
    )
    agent_pixel = project_with_fallbacks(agent_pos_3d)
    if agent_pixel:
        cv2.circle(overlay, agent_pixel, radius=10, color=color_agent, thickness=-1)
        cv2.circle(overlay, agent_pixel, radius=5, color=color_agent_center, thickness=-1)

    alpha = max(0.0, min(1.0, float(args.alpha)))
    cv2.addWeighted(overlay, alpha, base_frame, 1 - alpha, 0, base_frame)

    plt.figure(figsize=(10, 8))
    plt.imshow(base_frame)
    plt.title("Agent View with Reachable Overlay")
    plt.axis("off")
    if args.save_path:
        print(f"[DEBUG] Projected points: {projected_count}/{len(reachable_positions)}")
        plt.savefig(args.save_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
