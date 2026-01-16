import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ai2thor.controller import Controller

# Debug: genera una mappa top-down con overlay dei punti raggiungibili.


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Definisce gli argomenti CLI dello script di debug top-down.
    Centralizza scene, dimensioni e parametri di overlay.
    """
    ap = argparse.ArgumentParser(description="Top-down map view with reachable overlay.")
    ap.add_argument("--scene", default="FloorPlan10")
    ap.add_argument("--grid_size", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--visibility_distance", type=float, default=3.0)
    ap.add_argument("--save_path", default="topdown_reachable3.png")
    ap.add_argument("--point_radius", type=int, default=3)
    ap.add_argument("--point_color", default="orange")
    ap.add_argument("--avoid_stride", type=int, default=1)
    ap.add_argument("--dump_metadata", default="", help="Optional path to save top-down metadata keys.")
    return ap


def extract_topdown_event(controller) -> object:
    """
    Richiede al simulatore il frame top-down tramite ToggleMapView.
    Lancia errore se il frame non e' disponibile.
    """
    event = controller.step(action="ToggleMapView")
    if event is None or event.frame is None:
        raise RuntimeError("ToggleMapView failed or returned no frame.")
    return event


def project_world_to_image(point_xyz, camera_matrix, image_width, image_height):
    """
    Proietta un punto world usando la cameraMatrix fornita dai metadata.
    Ritorna coordinate pixel o None se fuori dal frame.
    """
    point_4d = np.append(point_xyz, 1)
    projected = camera_matrix @ point_4d
    if projected[2] <= 0:
        return None
    u = projected[0] / projected[2]
    v = projected[1] / projected[2]
    pixel_x = int((u + 0.5) * image_width)
    pixel_y = int((0.5 - v) * image_height)
    if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
        return (pixel_x, pixel_y)
    return None


def project_world_to_topdown(point_xyz, map_props, image_width, image_height):
    """
    Proiezione world->pixel usando i parametri ortografici di MapView.
    Ritorna coordinate pixel o None se fuori dalla mappa.
    """
    pos = map_props.get("position") or {}
    ortho_size = map_props.get("orthographicSize")
    if ortho_size is None:
        ortho_size = map_props.get("orthographic_size")
    if pos is None or ortho_size is None:
        return None
    try:
        cam_x = float(pos.get("x", 0.0))
        cam_z = float(pos.get("z", 0.0))
        half_h = float(ortho_size)
    except Exception:
        return None
    half_w = half_h * (float(image_width) / float(image_height))
    x_min = cam_x - half_w
    x_max = cam_x + half_w
    z_min = cam_z - half_h
    z_max = cam_z + half_h
    x = float(point_xyz[0])
    z = float(point_xyz[2])
    if x_max <= x_min or z_max <= z_min:
        return None
    u = (x - x_min) / (x_max - x_min)
    v = 1.0 - (z - z_min) / (z_max - z_min)
    pixel_x = int(round(u * (image_width - 1)))
    pixel_y = int(round(v * (image_height - 1)))
    if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
        return (pixel_x, pixel_y)
    return None


def main() -> None:
    """
    Entry point dello script: carica scena, overlay e salva/mostra.
    Genera mappa top-down con punti raggiungibili.
    """
    args = build_arg_parser().parse_args()
    controller = Controller(
        agentMode="default",
        visibilityDistance=float(args.visibility_distance),
        scene=args.scene,
        gridSize=float(args.grid_size),
        width=int(args.width),
        height=int(args.height),
    )

    event_map = controller.step(action="GetReachablePositions")
    reachable_positions = event_map.metadata.get("actionReturn", []) or []
    agent_meta = event_map.metadata.get("agent") or {}
    agent_pos = agent_meta.get("position") or {}
    agent_rot = agent_meta.get("rotation") or {}

    topdown_event = extract_topdown_event(controller)
    topdown_frame = topdown_event.frame.copy()
    camera_raw = topdown_event.metadata.get("cameraMatrix") or topdown_event.metadata.get("cameraProjectionMatrix")
    camera_matrix = None
    if camera_raw:
        camera_matrix = np.array(camera_raw).reshape(4, 4).T

    map_props = None
    if camera_matrix is None:
        props_event = controller.step(action="GetMapViewCameraProperties")
        map_props = props_event.metadata.get("actionReturn") or {}

    if args.dump_metadata:
        dump = {
            "topdown_metadata_keys": list(topdown_event.metadata.keys()),
            "topdown_action_return_keys": list((topdown_event.metadata.get("actionReturn") or {}).keys())
            if isinstance(topdown_event.metadata.get("actionReturn"), dict)
            else None,
            "map_view_props": map_props,
        }
        with open(args.dump_metadata, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)

    overlay = topdown_frame.copy()
    reachable_cells = set()
    for point in reachable_positions:
        point_xyz = np.array([point["x"], point.get("y", 0.0), point["z"]])
        if camera_matrix is not None:
            pixel = project_world_to_image(point_xyz, camera_matrix, args.width, args.height)
        else:
            pixel = project_world_to_topdown(point_xyz, map_props or {}, args.width, args.height)
        if pixel:
            cv2.circle(overlay, pixel, args.point_radius, (0, 165, 255), -1)
        key = (int(round(point["x"] / args.grid_size)), int(round(point["z"] / args.grid_size)))
        reachable_cells.add(key)

    if reachable_positions:
        xs = [p["x"] for p in reachable_positions]
        zs = [p["z"] for p in reachable_positions]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        step = max(1, int(args.avoid_stride)) * float(args.grid_size)
        x = min_x
        while x <= max_x:
            z = min_z
            while z <= max_z:
                key = (int(round(x / args.grid_size)), int(round(z / args.grid_size)))
                if key not in reachable_cells:
                    point_xyz = np.array([x, 0.0, z])
                    if camera_matrix is not None:
                        pixel = project_world_to_image(point_xyz, camera_matrix, args.width, args.height)
                    else:
                        pixel = project_world_to_topdown(point_xyz, map_props or {}, args.width, args.height)
                    if pixel:
                        cv2.circle(overlay, pixel, 2, (0, 0, 255), -1)
                z += step
            x += step

    agent_xyz = np.array(
        [float(agent_pos.get("x", 0.0)), float(agent_pos.get("y", 0.0)), float(agent_pos.get("z", 0.0))]
    )
    front_xyz = np.array(
        [
            float(agent_pos.get("x", 0.0)) + 0.5 * np.sin(np.deg2rad(float(agent_rot.get("y", 0.0)))),
            float(agent_pos.get("y", 0.0)),
            float(agent_pos.get("z", 0.0)) + 0.5 * np.cos(np.deg2rad(float(agent_rot.get("y", 0.0)))),
        ]
    )
    if camera_matrix is not None:
        agent_px = project_world_to_image(agent_xyz, camera_matrix, args.width, args.height)
        front_px = project_world_to_image(front_xyz, camera_matrix, args.width, args.height)
    else:
        agent_px = project_world_to_topdown(agent_xyz, map_props or {}, args.width, args.height)
        front_px = project_world_to_topdown(front_xyz, map_props or {}, args.width, args.height)
    if agent_px and front_px:
        cv2.arrowedLine(overlay, agent_px, front_px, (0, 0, 0), 3, tipLength=0.3)
        cv2.circle(overlay, agent_px, 6, (255, 255, 255), -1)
        cv2.circle(overlay, agent_px, 4, (0, 0, 0), -1)

    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(args.width / 100.0, args.height / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb)
    ax.axis("off")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Reachable", markerfacecolor="#00A5FF", markersize=6),
        plt.Line2D([0], [0], marker="o", color="w", label="Avoid", markerfacecolor="#FF0000", markersize=6),
        plt.Line2D([0], [0], marker=">", color="#000000", label="Robot", markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.8)
    if args.save_path:
        plt.savefig(args.save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


if __name__ == "__main__":
    main()
