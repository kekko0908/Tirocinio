# Scopo: memoria spaziale degli oggetti (short-term + long-term).
import json
from pathlib import Path
from typing import Dict, List, Optional


class ObjectMemory:
    def __init__(self, storage_path: Path, scene_id: str, episode_id: str):
        """
        Inizializza memoria oggetti e storage long-term.
        Imposta scena ed episodio per il tracciamento.
        Prepara short_term e carica long_term se presente.
        """
        self.storage_path = Path(storage_path)
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.short_term: Dict[str, Dict] = {"detections": [], "last_seen": {}}
        self.long_term: Dict = {}
        self._load_long_term()

    def _load_long_term(self) -> None:
        """
        Carica la memoria long-term dal file JSON.
        Gestisce errori di lettura o parsing in sicurezza.
        Lascia long_term vuoto se fallisce.
        """
        if self.storage_path.exists():
            try:
                self.long_term = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception:
                self.long_term = {}

    def _save_long_term(self) -> None:
        """
        Salva la memoria long-term su disco.
        Crea la directory di destinazione se serve.
        Serializza in JSON con indent per leggibilita.
        """
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(self.long_term, indent=2), encoding="utf-8")

    def _update_long_term(self, target_type: str, position: Dict, step: int, confidence: float) -> None:
        """
        Aggiorna statistiche long-term per un target.
        Aggiorna count, posizione media e ultimo avvistamento.
        Persistente su file dopo l'update.
        """
        scene = self.long_term.setdefault(self.scene_id, {})
        entry = scene.setdefault(target_type, {"count": 0, "mean_position": None, "last_position": None})
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last_position"] = position
        entry["last_seen_step"] = int(step)
        entry["last_seen_episode"] = self.episode_id
        entry["last_confidence"] = float(confidence)

        mean = entry.get("mean_position")
        if mean is None:
            entry["mean_position"] = position
        else:
            c = entry["count"]
            entry["mean_position"] = {
                "x": (mean.get("x", 0.0) * (c - 1) + position.get("x", 0.0)) / c,
                "y": (mean.get("y", 0.0) * (c - 1) + position.get("y", 0.0)) / c,
                "z": (mean.get("z", 0.0) * (c - 1) + position.get("z", 0.0)) / c,
            }
        self._save_long_term()

    def record_detection(
        self,
        target_type: str,
        position: Optional[Dict],
        step: int,
        confidence: float,
        centroid_px: Optional[List[int]] = None,
        source: str = "unknown",
    ) -> None:
        """
        Registra una detection nella memoria short-term.
        Aggiorna last_seen e limita la lista recente.
        Propaga su long_term se posizione disponibile.
        """
        record = {
            "object_class": target_type,
            "position": position,
            "confidence": float(confidence),
            "step": int(step),
            "centroid_px": centroid_px,
            "source": source,
            "episode_id": self.episode_id,
        }
        self.short_term["detections"].append(record)
        self.short_term["detections"] = self.short_term["detections"][-20:]

        self.short_term["last_seen"][target_type] = record
        if position:
            self._update_long_term(target_type, position, step, confidence)

    def get_summary(self, target_type: str) -> Dict:
        """
        Restituisce un riepilogo delle detection recenti.
        Include last_seen e topk per confidenza.
        Ritorna un dict pronto per logging.
        """
        last_seen = self.short_term["last_seen"].get(target_type)
        recent = [d for d in self.short_term["detections"] if d["object_class"] == target_type]
        recent = sorted(recent, key=lambda d: d["confidence"], reverse=True)[:3]
        return {
            "target_last_seen": last_seen,
            "recent_detections_topk": recent,
        }

    def get_long_term_priors(self, target_type: str) -> Dict:
        """
        Recupera prior spaziali dalla memoria long-term.
        Usa la mean_position come indizio principale.
        Ritorna una lista topk anche se vuota.
        """
        scene = self.long_term.get(self.scene_id, {})
        entry = scene.get(target_type)
        if not entry:
            return {"likely_target_regions_topk": []}
        mean = entry.get("mean_position")
        return {
            "likely_target_regions_topk": [
                {"position": mean, "count": entry.get("count", 0), "last_position": entry.get("last_position")}
            ]
        }

    def record_action_trace(
        self,
        target_type: str,
        actions: List[Dict],
        start_step: int,
        end_step: int,
        success: bool,
        mode: str = "global",
        reference: Optional[str] = None,
    ) -> None:
        """
        Salva una traccia azioni associata al target.
        Memorizza parametri di esecuzione e successo.
        Mantiene solo le ultime tracce e salva.
        """
        # Salva la sequenza di movimenti che ha portato al target.
        scene = self.long_term.setdefault(self.scene_id, {})
        entry = scene.setdefault(target_type, {"count": 0, "mean_position": None, "last_position": None})
        traces = entry.setdefault("action_traces", [])
        traces.append(
            {
                "episode_id": self.episode_id,
                "start_step": int(start_step),
                "end_step": int(end_step),
                "success": bool(success),
                "mode": mode,
                "reference": reference,
                "actions": actions,
            }
        )
        entry["action_traces"] = traces[-5:]
        self._save_long_term()
