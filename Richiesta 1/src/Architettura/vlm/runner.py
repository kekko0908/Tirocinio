"""Gestione chiamate VLM e salvataggio risposte."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from Architettura.ai2thor.context import build_context
from Architettura.paths import ARTIFACTS_ROOT, PROJECT_ROOT
try:
    from PIL import Image
except Exception:
    Image = None

VLM_RESPONSES_DIR = ARTIFACTS_ROOT / "vlm_outputs" / "responses"
DIAGNOSTIC_DIR = ARTIFACTS_ROOT / "diagnostic"
_FALLBACK_LAST_SEEN: Dict[str, Any] | None = None


def _to_pil_image(img: Any) -> "Image.Image" | None:
    # Converte un'immagine generica (numpy o gia' PIL) in PIL.Image.
    # PIL = Pillow (libreria Python per gestire immagini).
    # Usiamo PIL perche' e' lo standard piu' comodo per leggere/salvare/convertire.
    # Se PIL non e' disponibile o la conversione fallisce, ritorna None.
    if Image is None:
        return None
    if isinstance(img, Image.Image):
        return img
    try:
        return Image.fromarray(img)
    except Exception:
        return None


def _save_diagnostic(images: list[Any], frame_idx: int, max_width: int = 640) -> None:
    # Crea un'immagine di debug affiancando le prime due immagini fornite.
    # Serve per verificare rapidamente cosa sta vedendo la VLM (es. semantic + overlay).
    if Image is None or len(images) < 2:
        return
    left = _to_pil_image(images[0])
    right = _to_pil_image(images[1])
    if left is None or right is None:
        return
    target_h = max(left.height, right.height)
    if left.height != target_h:
        new_w = int(left.width * (target_h / left.height))
        left = left.resize((new_w, target_h), Image.BILINEAR)
    if right.height != target_h:
        new_w = int(right.width * (target_h / right.height))
        right = right.resize((new_w, target_h), Image.BILINEAR)
    combined = Image.new("RGB", (left.width + right.width, target_h))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    if max_width and combined.width > max_width:
        scale = float(max_width) / float(combined.width)
        new_size = (max_width, int(combined.height * scale))
        combined = combined.resize(new_size, Image.BILINEAR)
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAGNOSTIC_DIR / f"diagnostic_{frame_idx:05d}.png"
    combined.save(out_path.as_posix())


def _extract_current_state(answer: str) -> str | None:
    """Estrae lo stato corrente (state/STATO/stato) dalla risposta VLM."""
    if not answer:
        return None
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*", "", answer).replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return None
    for key in ("state", "STATO", "stato"):
        value = payload.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return None


def _side_from_norm_x(x_norm: float) -> str:
    if x_norm < 0.4:
        return "left"
    if x_norm > 0.6:
        return "right"
    return "center"


def run_vlm_on_step(
    vlm_client: Any,
    event: Any,
    image: Any,
    action_name: str,
    frame_idx: int,
    paths: Dict[str, str | None],
    target_label: str | None = None,
    user_prompt: str | None = None,
    scan_count: int | None = None,
    last_state: str | None = None,
) -> str:
    """Esegue la VLM per uno step e salva la risposta in JSON.

    Flusso sintetico:
    - costruisce il contesto (telemetria + target + stato),
    - prepara prompt e immagini,
    - chiama la VLM,
    - salva debug e risposta su file.
    """
    # Parametri chiave in ingresso:
    # - event: stato AI2-THOR corrente (frame + metadata).
    # - image: frame RGB grezzo (fallback se non troviamo semantic).
    # - paths: dizionario con path degli artefatti salvati (semantic, overlay, yolo, ecc.).
    # - target_label: target normalizzato.
    # - user_prompt: testo che guida la VLM.
    # - scan_count: contatore rotazioni iniziali.
    #
    # 1) Costruiamo il contesto dal metadata (telemetria, target, stato collisioni).
    print(
        f"\nVLM: analizzo frame dopo '{action_name}'...",
        flush=True,
    )
    # Se presente, prendiamo il JSON YOLO salvato per stimare "last_seen" e posizione.
    yolo_json_path = paths.get("yolo_json") if paths else None
    # Costruisce il contesto "legibile" per la VLM (con informazioni utili alla decisione).
    context = build_context(
        event,
        scan_count=scan_count,
        target_label=target_label,
        yolo_json_path=yolo_json_path,
        frame_idx=frame_idx,
    )
    # Pulizia/arricchimento del contesto:
    # - rimuoviamo scan_count (gia' presente nel prompt),
    # - manteniamo last_seen per dare memoria alla VLM,
    # - aggiungiamo yolo_target_match: True se il target risulta trovato.
    if isinstance(context, dict):
        context.pop("scan_count", None)
        yolo_match = paths.get("yolo_target_match") if paths else None
        context["yolo_target_match"] = bool(yolo_match)
        context["last_state"] = last_state
        target_center_x = paths.get("yolo_target_center_x") if paths else None

        # Fallback memoria target: se last_seen da context e' assente, usa il centro YOLO.
        global _FALLBACK_LAST_SEEN
        if bool(yolo_match) and target_center_x is not None:
            try:
                x_norm = float(target_center_x)
            except Exception:
                x_norm = None
            if x_norm is not None:
                _FALLBACK_LAST_SEEN = {
                    "label": target_label,
                    "side": _side_from_norm_x(x_norm),
                    "confidence": None,
                    "frame_index": frame_idx,
                    "stale": False,
                }
        last_seen = context.get("last_seen")
        if not isinstance(last_seen, dict) and _FALLBACK_LAST_SEEN is not None:
            last_seen = dict(_FALLBACK_LAST_SEEN)
            if frame_idx is not None:
                last_seen["stale"] = last_seen.get("frame_index") != frame_idx
            context["last_seen"] = last_seen
        if isinstance(last_seen, dict):
            context["target_last_seen_side"] = last_seen.get("side")
            context["target_last_seen_stale"] = bool(last_seen.get("stale"))
        else:
            context["target_last_seen_side"] = None
            context["target_last_seen_stale"] = None
        if target_center_x is not None:
            context["target_center_x"] = float(target_center_x)
    # 2) Compone il prompt finale: system prompt + richiesta + contesto JSON.
    if user_prompt is not None:
        prompt = vlm_client.compose_prompt(user_prompt, context=context)
    else:
        prompt = vlm_client.compose_prompt(context=context)
    # Lista immagini da fornire alla VLM (puo' essere 1 o 2 immagini).
    images = []
    # 3) Prepara le immagini: preferisce semantic se disponibile.
    if paths and Image is not None:
        semantic_path = paths.get("semantic")
        if semantic_path:
            candidate = Path(semantic_path)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            try:
                images.append(Image.open(candidate.as_posix()).convert("RGB"))
            except Exception:
                # Se la lettura fallisce, resettiamo e useremo il frame RGB.
                images = []
    # Se non abbiamo semantic, usa il frame RGB grezzo.
    if not images:
        images = [image]
    # Aggiunge overlay di oracolo o YOLO-World come seconda immagine (se disponibile).
    if paths and Image is not None:
        overlay_path = paths.get("oracle_overlay") or paths.get("yolo")
        if overlay_path:
            candidate = Path(overlay_path)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            try:
                overlay_img = Image.open(candidate.as_posix()).convert("RGB")
                # Seconda immagine: evidenzia target (oracle/YOLO) per aiutare la VLM.
                images.append(overlay_img)
            except Exception:
                # Se l'overlay non e' leggibile, continuiamo con una sola immagine.
                pass
    # 4) Salva un PNG diagnostico affiancato per debugging rapido.
    _save_diagnostic(images, frame_idx)
    # 5) Chiamata alla VLM: produce testo (JSON) a partire da immagini + prompt.
    # Qui otteniamo la risposta effettiva della VLM, generata usando il prompt
    # costruito da compose_prompt (system + richiesta + contesto).
    answer = vlm_client.generate(image=images, prompt=prompt)
    current_state = _extract_current_state(answer)
    print(f"VLM risposta: {answer}")

    answer_text = str(answer or "").strip()
    if answer_text.startswith("```"):
        debug_answer = f"VLM risposta: {answer_text}"
    else:
        debug_answer = f"VLM risposta: ```json\n{answer_text}\n```"

    debug_path = Path(__file__).resolve().parent / "vlm_output.json"
    # Salva un dump completo del prompt/contesto per debug locale.
    # Questo file serve a ispezionare esattamente cosa e' stato inviato alla VLM.
    debug_payload = {
        "frame_index": frame_idx,
        "action": action_name,
        "system_prompt": getattr(vlm_client, "system_prompt", None),
        "user_prompt": user_prompt,
        "prompt": prompt,
        "context": context,
        "current_state": current_state,
        "scan_count": None,
        "vlm_response": answer,
        "vlm_response_debug": debug_answer,
    }
    debug_path.write_text(
        json.dumps(debug_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    response_path = VLM_RESPONSES_DIR / f"response_{frame_idx:05d}.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    # Salva la risposta per ogni frame in Artefatti/vlm_outputs/responses.
    payload = {
        "frame_index": frame_idx,
        "action": action_name,
        "prompt": prompt,
        "response": answer,
        "context": context,
        "current_state": current_state,
    }
    response_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return answer
