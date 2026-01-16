# Scopo: memoria persistente dei fallimenti/strategie di navigazione.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from .utils import safe_write_json


@dataclass
class _TargetStats:
    runs: int = 0
    success: int = 0
    no_progress: int = 0
    action_failures: Dict[str, int] = field(default_factory=dict)
    recent_failures: List[Dict] = field(default_factory=list)
    avoid_zones: Dict[str, int] = field(default_factory=dict)


class BehaviorMemory:
    def __init__(self, path: Path, scene_id: str):
        """
        Inizializza memoria persistente per statistiche di navigazione.
        Carica stato da disco e imposta parametri di quantizzazione.
        """
        self.path = Path(path)
        self.scene_id = str(scene_id)
        self.data = {"version": 1, "scenes": {}}
        self._load()
        self.grid_size_m = 0.5
        self.yaw_bin_deg = 30

    def _load(self) -> None:
        """
        Carica il file JSON se esiste e valida la struttura base.
        In caso di errore, ripristina lo schema vuoto.
        """
        if self.path.exists():
            try:
                self.data = self.path.read_text(encoding="utf-8")
            except Exception:
                return
            try:
                import json

                self.data = json.loads(self.data)
            except Exception:
                self.data = {"version": 1, "scenes": {}}

    def _bucket(self, target_type: str) -> Dict:
        """
        Restituisce il bucket di scena/target, creando campi mancanti.
        Centralizza la struttura dati per statistiche.
        """
        scenes = self.data.setdefault("scenes", {})
        scene = scenes.setdefault(self.scene_id, {})
        bucket = scene.setdefault(target_type, {})
        bucket.setdefault("stats", {})
        bucket.setdefault("recent_failures", [])
        return bucket

    def _stats(self, target_type: str) -> _TargetStats:
        """
        Converte il bucket in _TargetStats con valori tipizzati.
        Isola la lettura da dati mancanti o corrotti.
        """
        bucket = self._bucket(target_type)
        stats = bucket.get("stats", {}) or {}
        return _TargetStats(
            runs=int(stats.get("runs", 0)),
            success=int(stats.get("success", 0)),
            no_progress=int(stats.get("no_progress", 0)),
            action_failures=dict(stats.get("action_failures", {}) or {}),
            recent_failures=list(bucket.get("recent_failures", []) or []),
            avoid_zones=dict(stats.get("avoid_zones", {}) or {}),
        )

    def _write(self, target_type: str, stats: _TargetStats) -> None:
        """
        Scrive su disco le statistiche aggiornate del target.
        Applica trimming alle failure recenti.
        """
        bucket = self._bucket(target_type)
        bucket["stats"] = {
            "runs": stats.runs,
            "success": stats.success,
            "no_progress": stats.no_progress,
            "action_failures": stats.action_failures,
            "avoid_zones": stats.avoid_zones,
        }
        bucket["recent_failures"] = stats.recent_failures[-20:]
        safe_write_json(self.path, self.data)

    def _quantize_pose(self, position: Optional[Dict], yaw: Optional[float]) -> Optional[Tuple[float, float, int]]:
        """
        Quantizza posizione e yaw su griglia per aggregare zone.
        Ritorna None se input incompleto o non valido.
        """
        if not position or yaw is None:
            return None
        try:
            x = float(position.get("x", 0.0))
            z = float(position.get("z", 0.0))
            yaw_f = float(yaw)
        except Exception:
            return None
        qx = round(x / self.grid_size_m) * self.grid_size_m
        qz = round(z / self.grid_size_m) * self.grid_size_m
        qyaw = int(round(yaw_f / self.yaw_bin_deg) * self.yaw_bin_deg) % 360
        return qx, qz, qyaw

    def _zone_key(self, qpose: Tuple[float, float, int]) -> str:
        """
        Serializza una posa quantizzata in chiave stringa.
        Usata per aggregare zone da evitare.
        """
        return f"{qpose[0]:.2f},{qpose[1]:.2f},{qpose[2]}"

    def start_target(self, target_type: str) -> None:
        """
        Registra l'inizio di un nuovo target per le statistiche.
        Incrementa il contatore di run.
        """
        stats = self._stats(target_type)
        stats.runs += 1
        self._write(target_type, stats)

    def record_success(self, target_type: str, step: int, position: Optional[Dict] = None, yaw: Optional[float] = None) -> None:
        """
        Registra un successo e lo aggiunge alle failure recenti.
        Conserva posizione/yaw per analisi post-run.
        """
        stats = self._stats(target_type)
        stats.success += 1
        stats.recent_failures.append(
            {"event": "success", "step": int(step), "ts": time.time(), "position": position, "yaw": yaw}
        )
        self._write(target_type, stats)

    def record_no_progress(self, target_type: str, step: int, position: Optional[Dict] = None, yaw: Optional[float] = None) -> None:
        """
        Registra un evento di no-progress e aggiorna avoid_zones.
        Salva il contesto per future decisioni.
        """
        stats = self._stats(target_type)
        stats.no_progress += 1
        qpose = self._quantize_pose(position, yaw)
        if qpose:
            key = self._zone_key(qpose)
            stats.avoid_zones[key] = int(stats.avoid_zones.get(key, 0)) + 1
        stats.recent_failures.append(
            {"event": "no_progress", "step": int(step), "ts": time.time(), "position": position, "yaw": yaw}
        )
        self._write(target_type, stats)

    def record_action_failure(
        self,
        target_type: str,
        action: str,
        reason: str,
        step: int,
        position: Optional[Dict] = None,
        yaw: Optional[float] = None,
    ) -> None:
        """
        Registra un fallimento di azione e aggiorna statistiche.
        Marca la zona come da evitare se la posa e' disponibile.
        """
        stats = self._stats(target_type)
        stats.action_failures[action] = int(stats.action_failures.get(action, 0)) + 1
        qpose = self._quantize_pose(position, yaw)
        if qpose:
            key = self._zone_key(qpose)
            stats.avoid_zones[key] = int(stats.avoid_zones.get(key, 0)) + 1
        stats.recent_failures.append(
            {
                "event": "action_fail",
                "action": action,
                "reason": reason,
                "step": int(step),
                "ts": time.time(),
                "position": position,
                "yaw": yaw,
            }
        )
        self._write(target_type, stats)

    def get_summary(self, target_type: str) -> Dict:
        """
        Restituisce un riassunto compatto per prompt VLM e log.
        Seleziona top-k azioni/zone da evitare.
        """
        stats = self._stats(target_type)
        avoid_actions = sorted(
            stats.action_failures.items(), key=lambda x: x[1], reverse=True
        )[:3]
        avoid_zones = sorted(
            stats.avoid_zones.items(), key=lambda x: x[1], reverse=True
        )[:5]
        zone_list = []
        for key, count in avoid_zones:
            try:
                x_str, z_str, yaw_str = key.split(",")
                zone_list.append(
                    {
                        "x": float(x_str),
                        "z": float(z_str),
                        "yaw": int(float(yaw_str)),
                        "count": int(count),
                    }
                )
            except Exception:
                continue
        return {
            "runs": stats.runs,
            "success": stats.success,
            "no_progress": stats.no_progress,
            "avoid_actions": [a for a, _ in avoid_actions],
            "avoid_zones": zone_list,
            "recent_failures": stats.recent_failures[-5:],
        }
