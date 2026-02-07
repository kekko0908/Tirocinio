"""Trova le scene con piu' oggetti compatibili con le classi YOLO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from ultralytics import YOLO

from Architettura.ai2thor.init_controller import SimConfig, create_controller


@dataclass
class ScanConfig:
    model_path: str = "MODELLI YOLO/yolo26x-seg.pt"
    floorplan_start: int = 1
    floorplan_end: int = 29
    top_k: int = 5


def _load_yolo_classes(model_path: str) -> set[str]:
    model = YOLO(model_path)
    names = model.names or {}
    return {str(name).strip().lower() for name in names.values() if name}


def _count_scene_matches(scene: str, yolo_classes: set[str]) -> tuple[int, list[str]]:
    cfg = SimConfig(scene=scene)
    controller = create_controller(cfg)
    try:
        event = controller.step(action="Pass")
        objects = event.metadata.get("objects") or []
        count = 0
        matched: set[str] = set()
        for obj in objects:
            obj_type = str(obj.get("objectType", "")).strip().lower()
            if obj_type and obj_type in yolo_classes:
                count += 1
                matched.add(obj_type)
        return count, sorted(matched)
    finally:
        controller.stop()


def run_scan(config: ScanConfig | None = None) -> list[tuple[str, int]]:
    cfg = config or ScanConfig()
    yolo_classes = _load_yolo_classes(cfg.model_path)
    results: list[tuple[str, int]] = []
    for idx in range(cfg.floorplan_start, cfg.floorplan_end + 1):
        scene = f"FloorPlan{idx}"
        try:
            count, matched = _count_scene_matches(scene, yolo_classes)
            preview = ", ".join(matched[:10])
            more = " ..." if len(matched) > 10 else ""
            print(f"[SCAN] {scene}: {count} | {preview}{more}")
            results.append((scene, count, matched))
        except Exception as exc:
            print(f"[SCAN] {scene}: errore {exc}")
            results.append((scene, 0, []))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[: cfg.top_k]


def main() -> None:
    top = run_scan()
    print("\nTOP 5 SCENE (YOLO match count + oggetti):")
    for scene, count, matched in top:
        print(f"- {scene}: {count}")
        if matched:
            print(f"  oggetti: {', '.join(matched)}")


if __name__ == "__main__":
    main()
