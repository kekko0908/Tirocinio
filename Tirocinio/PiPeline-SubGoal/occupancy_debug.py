import argparse

import matplotlib.pyplot as plt
from ai2thor.controller import Controller

# Debug: mappa di occupazione top-down basata su GetReachablePositions.


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Definisce argomenti CLI per la mappa di occupazione.
    Centralizza parametri scena e salvataggio.
    """
    ap = argparse.ArgumentParser(description="AI2-THOR occupancy grid debug map.")
    ap.add_argument("--scene", default="FloorPlan10")
    ap.add_argument("--grid_size", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--visibility_distance", type=float, default=1.5)
    ap.add_argument("--save_path", default="", help="Optional path to save PNG instead of show.")
    return ap


def main() -> None:
    """
    Entry point: genera la mappa top-down e la salva/mostra.
    Usa GetReachablePositions come base navigabile.
    """
    args = build_arg_parser().parse_args()
    controller = Controller(
        agentMode="default",
        visibilityDistance=float(args.visibility_distance),
        scene=args.scene,
        gridSize=float(args.grid_size),
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=int(args.width),
        height=int(args.height),
    )

    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata.get("actionReturn", []) or []

    xs = [p["x"] for p in reachable_positions]
    zs = [p["z"] for p in reachable_positions]

    plt.figure(figsize=(8, 8))
    plt.scatter(xs, zs, c="blue", s=10, label="Reachable")

    agent_pos = (event.metadata.get("agent") or {}).get("position") or {}
    plt.scatter(
        [agent_pos.get("x", 0.0)],
        [agent_pos.get("z", 0.0)],
        c="red",
        s=100,
        marker="*",
        label="Agent",
    )

    scene_name = event.metadata.get("sceneName", args.scene)
    plt.title(f"Occupancy Grid Debug - {scene_name}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()

    if args.save_path:
        plt.savefig(args.save_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
