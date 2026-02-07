"""Mapping tra italiano e label YOLO."""

from __future__ import annotations

from typing import Set

ARTICLES: Set[str] = {
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "uno",
    "una",
    "un'",
}

def normalize_target(text: str) -> str:
    """Normalizza una parola/target in modo che corrisponda alle label YOLO."""
    # 1) pulisce spazi/apostrofi e rimuove articoli ("il", "la", "uno"...),
    # 2) ricompone la frase,
    # 3) restituisce la forma normalizzata (senza sinonimi/mapping).
    raw = (text or "").strip().lower()
    if not raw:
        return ""
    parts = [p for p in raw.replace("'", " ").split() if p not in ARTICLES]
    if not parts:
        return raw
    base = " ".join(parts)
    return base
