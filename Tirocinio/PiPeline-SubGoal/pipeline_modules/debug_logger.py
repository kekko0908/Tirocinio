# Scopo: logger diagnostico su file (jsonl + testo).
import json
from pathlib import Path
from typing import Dict, Optional


class DebugLogger:
    def __init__(self, jsonl_path: Path, text_path: Optional[Path] = None):
        """
        Inizializza i file di debug jsonl e testo.
        Crea le cartelle di output se mancanti.
        Memorizza i path da usare per i log.
        """
        self.jsonl_path = Path(jsonl_path)
        self.text_path = Path(text_path) if text_path else None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.text_path:
            self.text_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict) -> None:
        """
        Scrive un record JSON in append su file jsonl.
        Serializza il dict con ensure_ascii per robustezza.
        Non fa buffering e registra riga per riga.
        """
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def log_text(self, line: str) -> None:
        """
        Scrive una riga di testo su file se abilitato.
        Normalizza newline e usa append.
        No-op se text_path non e presente.
        """
        if not self.text_path:
            return
        with self.text_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
