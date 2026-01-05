# Scopo: exploration memory e summary per la navigazione.
import math
from typing import Dict, Tuple


class ExplorationMemory:
    def __init__(self, cell_size: float = 0.5, dir_bins: int = 8):
        self.cell_size = float(cell_size)
        self.dir_bins = int(dir_bins)
        self.cells: Dict[Tuple[int, int], Dict] = {}

    def cell_key(self, pos: Dict) -> Tuple[int, int]:
        x = float(pos.get("x", 0.0))
        z = float(pos.get("z", 0.0))
        return (int(round(x / self.cell_size)), int(round(z / self.cell_size)))

    def dir_idx(self, yaw: float) -> int:
        step = 360.0 / float(self.dir_bins)
        return int(round(yaw / step)) % self.dir_bins

    def dir_vector(self, dir_idx: int) -> Tuple[int, int]:
        angle = math.radians(dir_idx * (360.0 / self.dir_bins))
        dx = int(round(math.sin(angle)))
        dz = int(round(math.cos(angle)))
        return dx, dz

    def neighbor_key(self, key: Tuple[int, int], dir_idx: int) -> Tuple[int, int]:
        dx, dz = self.dir_vector(dir_idx)
        return (key[0] + dx, key[1] + dz)

    def update(self, pos: Dict, yaw: float) -> None:
        key = self.cell_key(pos)
        cell = self.cells.setdefault(key, {"visited": 0, "dirs": set()})
        cell["visited"] += 1
        cell["dirs"].add(self.dir_idx(yaw))

    def coverage_ratio(self, key: Tuple[int, int]) -> float:
        cell = self.cells.get(key)
        if not cell:
            return 0.0
        return float(len(cell["dirs"])) / float(self.dir_bins)

    def estimate_free_m(self, sensor: Dict, dir_idx: int, current_dir: int) -> float:
        front = float(sensor.get("dist_front_m", 1.0))
        left = float(sensor.get("dist_left_m", 1.0))
        right = float(sensor.get("dist_right_m", 1.0))
        back = max(left, right)
        diff = (dir_idx - current_dir) % self.dir_bins
        step = 360.0 / float(self.dir_bins)
        angle = diff * step
        if angle <= 30 or angle >= 330:
            return front
        if 150 <= angle <= 210:
            return back
        if 60 <= angle <= 120:
            return right
        if 240 <= angle <= 300:
            return left
        if 0 < angle < 60:
            return min(front, right)
        if 300 < angle < 360:
            return min(front, left)
        if 120 < angle < 180:
            return min(back, right)
        if 180 < angle < 240:
            return min(back, left)
        return front

    def summarize(self, pos: Dict, yaw: float, sensor: Dict) -> Dict:
        key = self.cell_key(pos)
        visited = key in self.cells
        coverage_ratio = self.coverage_ratio(key)
        current_dir = self.dir_idx(yaw)
        view_dirs = sorted(self.cells.get(key, {}).get("dirs", set()))

        ranked = []
        for d in range(self.dir_bins):
            neighbor = self.neighbor_key(key, d)
            unvisited = neighbor not in self.cells
            free_m = self.estimate_free_m(sensor, d, current_dir)
            novelty = (1.0 if unvisited else 0.0) + min(free_m / 2.0, 1.0) + (1.0 - coverage_ratio)
            ranked.append(
                {
                    "dir_idx": int(d),
                    "free_m": round(float(free_m), 2),
                    "unvisited": bool(unvisited),
                    "novelty": round(float(novelty), 2),
                }
            )

        ranked.sort(key=lambda r: (r["unvisited"], r["free_m"], r["novelty"]), reverse=True)
        return {
            "pose_discrete": {"cell": {"x": key[0], "z": key[1]}, "dir_idx": int(current_dir)},
            "visited": bool(visited),
            "coverage_pct": round(float(coverage_ratio) * 100.0, 1),
            "coverage_ratio": round(float(coverage_ratio), 3),
            "view_dirs": view_dirs,
            "ranked_directions": ranked[:3],
            "novelty_score": ranked[0]["novelty"] if ranked else 0.0,
        }
