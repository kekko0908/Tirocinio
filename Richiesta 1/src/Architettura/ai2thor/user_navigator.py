"""Navigator manuale da CLI per esplorare una scena AI2-THOR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from Architettura.ai2thor.init_controller import SimConfig, create_controller


@dataclass
class NavConfig:
    scene: str = "FloorPlan7"
    move_mag: float = 0.10
    rotate_deg: float = 15.0
    look_deg: float = 15.0

# TOP 5 SCENE (YOLO match count + oggetti):
# - FloorPlan7: 15
#   oggetti: apple, book, bowl, chair, cup, fork, knife, microwave, sink, spoon, toaster, vase
# - FloorPlan10: 14
#   oggetti: apple, bottle, bowl, chair, cup, fork, knife, microwave, sink, spoon, toaster, vase
# - FloorPlan1: 13
#   oggetti: apple, book, bottle, bowl, cup, fork, knife, microwave, sink, spoon, toaster, vase
# - FloorPlan18: 13
#   oggetti: apple, bowl, chair, cup, fork, knife, microwave, sink, spoon, toaster, vase
# - FloorPlan20: 13
#   oggetti: apple, bowl, cup, fork, knife, microwave, sink, spoon, toaster, vase


_HELP = """
Comandi:
  w  -> MoveAhead
  s  -> MoveBack
  a  -> MoveLeft
  d  -> MoveRight
  q  -> RotateLeft
  e  -> RotateRight
  r  -> LookUp
  f  -> LookDown
  h  -> help
  x  -> exit
"""


def _step(controller, action: str, params: dict) -> None:
    event = controller.step(action=action, **params)
    success = event.metadata.get("lastActionSuccess", True)
    error = event.metadata.get("errorMessage")
    if not success:
        print(f"[NAV] fail action={action} error={error}")


def run_user_navigator(config: NavConfig | None = None) -> None:
    cfg = config or NavConfig()
    sim_cfg = SimConfig(scene=cfg.scene)
    controller = create_controller(sim_cfg)
    print(_HELP.strip())
    try:
        while True:
            cmd = input("nav> ").strip().lower()
            if not cmd:
                continue
            if cmd == "x":
                break
            if cmd == "h":
                print(_HELP.strip())
                continue
            if cmd == "w":
                _step(controller, "MoveAhead", {"moveMagnitude": cfg.move_mag})
            elif cmd == "s":
                _step(controller, "MoveBack", {"moveMagnitude": cfg.move_mag})
            elif cmd == "a":
                _step(controller, "MoveLeft", {"moveMagnitude": cfg.move_mag})
            elif cmd == "d":
                _step(controller, "MoveRight", {"moveMagnitude": cfg.move_mag})
            elif cmd == "q":
                _step(controller, "RotateLeft", {"degrees": cfg.rotate_deg})
            elif cmd == "e":
                _step(controller, "RotateRight", {"degrees": cfg.rotate_deg})
            elif cmd == "r":
                _step(controller, "LookUp", {"degrees": cfg.look_deg})
            elif cmd == "f":
                _step(controller, "LookDown", {"degrees": cfg.look_deg})
            else:
                print("Comando non valido. Premi 'h' per help.")
    finally:
        controller.stop()


if __name__ == "__main__":
    run_user_navigator()
