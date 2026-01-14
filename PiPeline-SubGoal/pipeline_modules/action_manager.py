# Scopo: regole azioni (safety, anti-loop, macro) e selezione finale.
from typing import Dict, List, Optional, Tuple


def action_fallback(step_idx: int) -> str:
    """
    Seleziona un'azione di fallback deterministica.
    Alterna rotazioni e avanzamento in base allo step.
    Restituisce una stringa azione valida.
    """
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
        """
        Inizializza parametri di sicurezza e macro-scan.
        Prepara queue e history per la selezione azioni.
        Imposta contatori e cooldown di scan.
        """
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
        """
        Accoda una macro di scan quando coverage e bassa.
        Rispetta cooldown e scansioni gia in corso.
        Aggiorna queue, scan_remaining e last_scan_step.
        """
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

    def enqueue_probe_scan(self, steps: int, degrees: int, step: int) -> bool:
        """
        Accoda una breve rotazione di probe se libero.
        Evita conflitti con altre macro attive.
        Ritorna True se la coda viene popolata.
        """
        if self.scan_remaining > 0 or self.queue:
            return False
        steps = max(1, int(steps))
        degrees = int(degrees)
        self.queue.extend([{"action": "RotateRight", "degrees": degrees} for _ in range(steps)])
        self.scan_remaining = steps
        self.last_scan_step = step
        return True

    def maybe_enqueue_advance(self, front_dist: float, coverage_ratio: float) -> None:
        """
        Accoda MoveAhead quando coverage e buona e frontale libero.
        Non interviene se sono presenti macro in corso.
        Usa soglie su coverage e distanza.
        """
        if self.scan_remaining > 0 or self.queue:
            return
        if coverage_ratio < 0.5:
            return
        if front_dist < self.advance_min_front:
            return
        self.queue.extend([{"action": "MoveAhead"} for _ in range(max(1, self.advance_steps))])

    def _choose_turn(self, sensor: Dict) -> Dict:
        """
        Sceglie una rotazione verso il lato piu libero.
        Confronta distanze sinistra e destra.
        Ritorna uno spec con action e degrees.
        """
        left = float(sensor.get("dist_left_m", 0.0))
        right = float(sensor.get("dist_right_m", 0.0))
        if left >= right:
            return {"action": "RotateLeft", "degrees": self.scan_degrees}
        return {"action": "RotateRight", "degrees": self.scan_degrees}

    def _is_inverse(self, action: str) -> bool:
        """
        Verifica se l'azione e l'inversa dell'ultima.
        Usa la history per evitare oscillazioni inutili.
        Ritorna True/False.
        """
        if not self.history:
            return False
        last = self.history[-1]
        return (last, action) in {
            ("MoveAhead", "MoveBack"),
            ("MoveBack", "MoveAhead"),
            ("MoveLeft", "MoveRight"),
            ("MoveRight", "MoveLeft"),
            ("RotateLeft", "RotateRight"),
            ("RotateRight", "RotateLeft"),
        }

    def _is_oscillation(self) -> bool:
        """
        Rileva un pattern oscillatorio nelle ultime azioni.
        Controlla sequenza a-b-a-b con valori diversi.
        Ritorna True se rileva un loop.
        """
        if len(self.history) < 4:
            return False
        a, b, c, d = self.history[-4:]
        return a == c and b == d and a != b

    def select_action(
        self, candidate_action: Optional[object], sensor: Dict, step: int
    ) -> Tuple[Dict, Dict, bool]:
        """
        Seleziona l'azione finale con regole di safety.
        Integra macro in coda, candidate action e recovery.
        Ritorna spec, meta decisione e flag scan completato.
        """
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
            if isinstance(candidate_action, dict):
                spec = dict(candidate_action)
                source = "forced"
            else:
                action = candidate_action or action_fallback(step)
                spec = {"action": action}

        action = spec.get("action", "")
        if sensor.get("collision", False) or not sensor.get("last_action_success", True):
            # Recovery deterministico: prova laterale/back prima di ruotare.
            left = float(sensor.get("dist_left_m", 0.0))
            right = float(sensor.get("dist_right_m", 0.0))
            back_est = max(left, right)
            safe = float(self.safe_front_m)
            if action in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
                if left >= right and left >= safe:
                    spec = {"action": "MoveLeft"}
                    overrides.append("recovery_side")
                elif right >= safe:
                    spec = {"action": "MoveRight"}
                    overrides.append("recovery_side")
                elif back_est >= safe:
                    spec = {"action": "MoveBack"}
                    overrides.append("recovery_back")
                else:
                    spec = self._choose_turn(sensor)
                    overrides.append("recovery_turn")
            else:
                spec = self._choose_turn(sensor)
                overrides.append("recovery_turn")
            action = spec["action"]

        if action == "MoveAhead" and float(sensor.get("dist_front_m", 0.0)) < self.safe_front_m:
            spec = self._choose_turn(sensor)
            overrides.append("blocked_front")
            action = spec["action"]
        if action == "MoveLeft" and float(sensor.get("dist_left_m", 0.0)) < self.safe_front_m:
            spec = self._choose_turn(sensor)
            overrides.append("blocked_left")
            action = spec["action"]
        if action == "MoveRight" and float(sensor.get("dist_right_m", 0.0)) < self.safe_front_m:
            spec = self._choose_turn(sensor)
            overrides.append("blocked_right")
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

    def get_state(self) -> Dict:
        """
        Restituisce uno stato sintetico dell'action manager.
        Include lunghezza coda, contatori e history recente.
        Utile per logging e debug.
        """
        return {
            "queue_len": len(self.queue),
            "scan_remaining": int(self.scan_remaining),
            "last_scan_step": int(self.last_scan_step),
            "history_tail": self.history[-6:],
        }
        if action == "MoveBack":
            left = float(sensor.get("dist_left_m", 0.0))
            right = float(sensor.get("dist_right_m", 0.0))
            back_est = max(left, right)
            if back_est < self.safe_front_m:
                spec = self._choose_turn(sensor)
                overrides.append("blocked_back")
                action = spec["action"]
