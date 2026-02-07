"""Utility per estrarre e tradurre il target."""

from __future__ import annotations

import logging
import re

try:
    import argostranslate.translate as argos_translate
except Exception:
    argos_translate = None

# Silenzia i log verbosi di Argos Translate.
logging.getLogger("argostranslate").setLevel(logging.WARNING)

from Architettura.yolo.labels import ARTICLES, normalize_target


def extract_target_phrase(text: str) -> str:
    """Estrae la parte di frase che descrive il target (in lingua originale)."""
    # Estrae la parte "oggetto" da una frase tipo "cerco la mela vicino al tavolo":
    # - prima identifica la porzione dopo verbi come "cerco/trova/mostra/guarda/vai a",
    # - poi taglia quando incontra stop-word (vicino, sopra, a destra, ecc.).
    if not text:
        return ""
    lowered = text.lower().strip()
    match = re.search(
        r"(cerco|cerca|cerchi|cerchiamo|cercate|cercano|cercami|cercatemi|"
        r"trovo|trova|trovi|troviamo|trovate|trovano|trovami|trovatemi|"
        r"mostra|mostrami|mostrate|mostratemi|mostramelo|mostramela|"
        r"guarda|vedi|individua|identifica|localizza|ricerca|scopri|"
        r"prendi|prendiamo"
        r"fammi vedere|fai vedere|vai a|vai|andare a|andare)\s+(.+)",
        lowered,
    )
    candidate = match.group(2) if match else lowered
    candidate = candidate.strip(" .")
    words = [w for w in re.split(r"\s+", candidate) if w]

    stop_words = {
        "che",
        "sopra",
        "sotto",
        "su",
        "in",
        "al",
        "allo",
        "alla",
        "ai",
        "agli",
        "alle",
        "dal",
        "dallo",
        "dalla",
        "dai",
        "dagli",
        "dalle",
        "del",
        "dello",
        "della",
        "dei",
        "degli",
        "delle",
        "nel",
        "nello",
        "nella",
        "nei",
        "negli",
        "nelle",
        "sul",
        "sullo",
        "sulla",
        "sui",
        "sugli",
        "sulle",
        "vicino",
        "accanto",
        "davanti",
        "dietro",
    }
    stop_idx = None
    for i, w in enumerate(words):
        if w in stop_words:
            stop_idx = i
            break
        if w == "fronte" and i > 0 and words[i - 1] == "di":
            stop_idx = i - 1
            break
        if w in {"sinistra", "destra"} and i > 0 and words[i - 1] == "a":
            stop_idx = i - 1
            break

    if stop_idx is not None:
        words = words[:stop_idx]

    while words and words[0] in ARTICLES:
        words = words[1:]
    return " ".join(words).strip()


def translate_target(text: str, source_lang: str = "it", target_lang: str = "en") -> str:
    """Traduce solo l'oggetto target (es. 'trova la mela' -> 'apple')."""
    # 1) Estrae la porzione target in italiano (solo la "parola/oggetto").
    # 2) Prova a tradurla in inglese con Argos.
    # 3) Normalizza spazi/articoli per uso YOLO.
    target_phrase = extract_target_phrase(text)
    if not target_phrase:
        return ""
    if argos_translate is None:
        translated = target_phrase
    else:
        try:
            translated = argos_translate.translate(target_phrase, source_lang, target_lang)
        except Exception:
            translated = target_phrase
    normalized = normalize_target(translated)
    return normalized or translated
