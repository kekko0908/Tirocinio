# Scopo: memoria leggera per guidare la VLM tra chiamate (hint, rotte, note).
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import safe_write_json


class VlmMemory:
    def __init__(self, path: Path, scene_id: str, episode_id: str):
        """
        Inizializza memoria VLM su file JSON.
        Carica dati esistenti se presenti e validi.
        Imposta scene_id ed episode_id.
        """
        self.path = Path(path)
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.data: Dict[str, Any] = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "targets": {},
        }
        if self.path.exists():
            try:
                existing = self.path.read_text(encoding="utf-8")
                if existing.strip():
                    loaded = __import__("json").loads(existing)
                    if isinstance(loaded, dict):
                        self.data.update(loaded)
            except Exception:
                # Se il file e' corrotto/non leggibile, riparto pulito.
                pass

    def _target_entry(self, target_label: str) -> Dict[str, Any]:
        """
        Restituisce o crea la voce per un target.
        Usa data["targets"] come contenitore.
        Ritorna il dict dell'entry.
        """
        targets = self.data.setdefault("targets", {})
        return targets.setdefault(target_label, {})

    def update_last_seen(
        self,
        target_label: str,
        hint: str,
        step: int,
        source: str,
        centroid_px: Optional[list] = None,
    ) -> None:
        """
        Aggiorna l'ultimo hint e step di un target.
        Registra fonte e centroid se disponibile.
        Esegue flush su disco.
        """
        entry = self._target_entry(target_label)
        entry["last_seen_hint"] = hint
        entry["last_seen_step"] = int(step)
        entry["last_seen_source"] = source
        if centroid_px is not None:
            entry["last_seen_centroid_px"] = centroid_px
        self.flush()

    def update_nav_summary(self, target_label: str, route_side: str, rationale: str) -> None:
        """
        Aggiorna info sulla rotta di navigazione.
        Registra lato scelto e razionale.
        Esegue flush su disco.
        """
        entry = self._target_entry(target_label)
        if route_side:
            entry["last_nav_route"] = route_side
        if rationale:
            entry["last_nav_rationale"] = rationale
        self.flush()

    def get_summary(self, target_label: str) -> Dict[str, Any]:
        """
        Ritorna una copia della entry del target.
        Evita side effects sul dict interno.
        Utile per logging o prompt.
        """
        return dict(self._target_entry(target_label))

    def flush(self) -> None:
        """
        Scrive i dati correnti su disco in JSON.
        Usa safe_write_json per garantire directory.
        Ritorna None.
        """
        safe_write_json(self.path, self.data)
