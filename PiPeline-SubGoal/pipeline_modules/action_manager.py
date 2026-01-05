# Scopo: regole azioni (safety, anti-loop, macro) e selezione finale.
from typing import Dict, List, Optional, Tuple


def action_fallback(step_idx: int) -> str:
    if step_idx % 6 == 0:
        return "RotateRight"
    if step_idx % 6 == 3:
        return "RotateLeft"
    return "MoveAhead"


class ActionManager:
    def __init__(
        self,
        safe_front_m: float = 0.6,
        scan_degrees: int = 60,
        scan_trigger: float = 0.5,
        scan_cooldown: int = 15,
        advance_steps: int = 2,
        advance_min_front: float = 1.2,
    ):
        self.safe_front_m = float(safe_front_m)
        self.scan_degrees = int(scan_degrees)
        self.scan_trigger = float(scan_trigger)
        self.scan_cooldown = int(scan_cooldown)
        self.advance_steps = int(advance_steps)
        self.advance_min_front = float(advance_min_front)
        self.queue: List[Dict] = []
        self.history: List[str] = []
        self.scan_remaining = 0
        self.last_scan_step = -999
        self.scan_completed_step = -999

    def maybe_enqueue_scan(self, coverage_ratio: float, step: int) -> None:
        if self.scan_remaining > 0:
            return
        if coverage_ratio >= self.scan_trigger:
            return
        if step - self.last_scan_step < self.scan_cooldown:
            return
        steps = max(1, int(round(360 / float(self.scan_degrees))))
        self.queue.extend([{"action": "RotateRight", "degrees": self.scan_degrees} for _ in range(steps)])
        self.scan_remaining = steps
        self.last_scan_step = step

    def maybe_enqueue_advance(self, front_dist: float, coverage_ratio: float) -> None:
        if self.scan_remaining > 0 or self.queue:
            return
        if coverage_ratio < 0.5:
            return
        if front_dist < self.advance_min_front:
            return
        self.queue.extend([{"action": "MoveAhead"} for _ in range(max(1, self.advance_steps))])

    def _choose_turn(self, sensor: Dict) -> Dict:
        left = float(sensor.get("dist_left_m", 0.0))
        right = float(sensor.get("dist_right_m", 0.0))
        if left >= right:
            return {"action": "RotateLeft", "degrees": self.scan_degrees}
        return {"action": "RotateRight", "degrees": self.scan_degrees}

    def _is_inverse(self, action: str) -> bool:
        if not self.history:
            return False
        last = self.history[-1]
        return (last, action) in {
            ("RotateLeft", "RotateRight"),
            ("RotateRight", "RotateLeft"),
            ("LookUp", "LookDown"),
            ("LookDown", "LookUp"),
        }

    def _is_oscillation(self) -> bool:
        if len(self.history) < 4:
            return False
        a, b, c, d = self.history[-4:]
        return a == c and b == d and a != b

    def select_action(
        self, candidate_action: Optional[str], sensor: Dict, step: int
    ) -> Tuple[Dict, Dict, bool]:
        overrides = []
        source = "vlm" if candidate_action else "rule"

        if self.queue:
            spec = self.queue.pop(0)
            source = "macro"
            if self.scan_remaining > 0:
                self.scan_remaining -= 1
                if self.scan_remaining == 0:
                    self.scan_completed_step = step
        else:
            action = candidate_action or action_fallback(step)
            spec = {"action": action}

        action = spec.get("action", "")
        if sensor.get("collision", False) or not sensor.get("last_action_success", True):
            spec = self._choose_turn(sensor)
            overrides.append("recovery")
            action = spec["action"]

        if action == "MoveAhead" and float(sensor.get("dist_front_m", 0.0)) < self.safe_front_m:
            spec = self._choose_turn(sensor)
            overrides.append("blocked_front")
            action = spec["action"]

        if self._is_inverse(action):
            spec = self._choose_turn(sensor)
            overrides.append("no_inverse")
            action = spec["action"]

        if self._is_oscillation():
            spec = self._choose_turn(sensor)
            overrides.append("anti_loop")
            action = spec["action"]

        if spec.get("action") in {"RotateLeft", "RotateRight"} and "degrees" not in spec:
            spec["degrees"] = self.scan_degrees

        self.history.append(action)
        return spec, {"source": source, "overrides": overrides}, self.scan_completed_step == step
