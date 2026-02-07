"""Esegue comandi VLM (JSON) sul controller AI2-THOR."""
# Questo modulo prende l'output testuale della VLM e lo trasforma in azioni.
# Include parsing robusto del JSON e validazione delle azioni consentite.
# Serve a proteggere il simulatore da comandi non previsti o malformati.

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple


# Azioni consentite: evita che la VLM inventi comandi non previsti.
ALLOWED_ACTIONS = {
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateRight",
    "RotateLeft",
    "LookUp",
    "LookDown",
    "Crouch",
    "Stand",
    "Stop",
}

FINAL_STATE_NAMES = {"FINE"}
FINAL_PHASE_ACTIONS = {"Stop"}


def _cm_to_meters(value_cm: float) -> float:
    """Converte centimetri in metri."""
    # Alcune VLM restituiscono distanze in cm: qui normalizziamo a metri.
    # Esempio: 15 cm -> 0.15 m.
    # Usato durante la normalizzazione dei parametri d'azione.
    return value_cm / 100.0


def parse_vlm_json(text: str) -> Dict[str, Any]:
    """Estrae un JSON dall'output VLM e lo parse in dict."""
    # Logica principale:
    # 1) prova a trovare un JSON ben formato nell'output della VLM,
    # 2) se non trova, tenta riparazioni/fallback (regex + brace balancing),
    # 3) se trova, restituisce un dict pronto per l'esecuzione.
    # Rimuove eventuali code fences ```json / ```JSON / ``` prima del parsing.
    cleaned = re.sub(r"```[a-zA-Z0-9_-]*", "", text or "")
    cleaned = cleaned.replace("```", "").strip()
    if "{" not in cleaned:
        fallback = _fallback_from_text(cleaned)
        if fallback is not None:
            return fallback
        raise ValueError("JSON non trovato nell'output VLM.")
    # Prova a estrarre il primo JSON bilanciato; se manca la "}" finale, chiudila.
    candidate = _extract_json_candidate(cleaned)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Estrae tutti i blocchi JSON bilanciati e prova a parsare dall'ultimo.
    blocks: list[str] = []
    depth = 0
    start = None
    for idx, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blocks.append(cleaned[start : idx + 1])
                    start = None

    if not blocks:
        # Ultimo tentativo: bilancia le parentesi e prova a parsare.
        repaired = _repair_json_candidate(cleaned)
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        fallback = _fallback_from_text(cleaned)
        if fallback is not None:
            return fallback
        raise ValueError("JSON non trovato nell'output VLM.")

    last_error = None
    for payload in reversed(blocks):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        fallback = _fallback_from_text(cleaned)
        if fallback is not None:
            return fallback
        raise last_error
    fallback = _fallback_from_text(cleaned)
    if fallback is not None:
        return fallback
    raise ValueError("JSON non valido.")


def _extract_json_candidate(text: str) -> str | None:
    """Estrae il primo oggetto JSON bilanciato, ignorando testo extra."""
    # Scorre il testo e cerca la prima apertura "{", poi bilancia le parentesi.
    # Restituisce il primo blocco JSON completo trovato (se esiste).
    # Serve per isolare JSON anche se c'e' testo extra prima/dopo.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    collected = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        collected.append(ch)
        if depth == 0 and collected:
            return "".join(collected)
    return None


def _repair_json_candidate(text: str) -> str | None:
    """Prova a riparare un JSON non chiuso bilanciando le parentesi."""
    # Se mancano parentesi di chiusura, le aggiunge in coda.
    # Poi tenta di estrarre un JSON bilanciato.
    # Utile quando la VLM tronca la risposta prima di "}".
    start = text.find("{")
    if start == -1:
        return None
    payload = text[start:]
    missing = payload.count("{") - payload.count("}")
    if missing > 0:
        payload = payload + ("}" * missing)
    # Taglia eventuale testo prima del JSON già incluso e prova a isolare l'oggetto.
    candidate = _extract_json_candidate(payload)
    return candidate or payload


def _fallback_from_text(text: str) -> Dict[str, Any] | None:
    """Fallback: se manca il JSON, prova a estrarre l'azione dal testo."""
    # Cerca parole chiave di azioni consentite nel testo libero.
    # Se trova, prova a estrarre un numero (gradi o distanza).
    # Ultima spiaggia per non bloccare il sistema.
    action = None
    for cand in ALLOWED_ACTIONS:
        if re.search(rf"\\b{re.escape(cand)}\\b", text):
            action = cand
            break
    if action is None:
        return None

    nums = re.findall(r"[-+]?\\d*\\.?\\d+", text)
    params: Dict[str, Any] = {}
    value = float(nums[0]) if nums else None

    if action.startswith("Rotate") or action.startswith("Look"):
        params["degrees"] = value if value is not None else 10.0
    elif action.startswith("Move"):
        params["moveMagnitude"] = value if value is not None else 0.10
    else:
        params = {}
    return {"action": action, "parameters": params}


def normalize_action(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Valida azione e normalizza i parametri (es. cm -> metri)."""
    # Controlla che l'azione sia consentita (whitelist).
    # Converte parametri alternativi (moveMagnitudeCm/magnitude) nel formato standard.
    # Ritorna (action, params) pronti per controller.step().
    action = payload.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Azione non valida: {action}")

    params = payload.get("parameters") or payload.get("params") or {}

    # Se la VLM fornisce la distanza in cm, trasformala in metri.
    if "moveMagnitudeCm" in params and "moveMagnitude" not in params:
        params["moveMagnitude"] = _cm_to_meters(float(params["moveMagnitudeCm"]))
        params.pop("moveMagnitudeCm", None)
    if "magnitude" in params and "moveMagnitude" not in params:
        params["moveMagnitude"] = float(params["magnitude"])
        params.pop("magnitude", None)

    return action, params


def _extract_state(payload: Dict[str, Any]) -> str | None:
    state = payload.get("state")
    if state is None:
        state = payload.get("STATO")
    if state is None:
        state = payload.get("stato")
    if state is None:
        return None
    return str(state).strip()


def _extract_actions(payload: Dict[str, Any]) -> list[str]:
    sequence = payload.get("action_sequence")
    actions: list[str] = []
    if isinstance(sequence, list) and sequence:
        for step in sequence:
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            if isinstance(action, str) and action.strip():
                actions.append(action.strip())
    if actions:
        return actions
    action = payload.get("action")
    if isinstance(action, str) and action.strip():
        return [action.strip()]
    return []


def validate_vlm_semantics(
    payload: Dict[str, Any],
    *,
    target_distance: float | None = None,
    final_distance_threshold: float = 0.8,
    distance_tolerance: float = 0.0,
) -> list[str]:
    """Valida coerenza logica minima del piano VLM rispetto al contesto."""
    errors: list[str] = []
    if target_distance is None:
        return errors
    try:
        dist = float(target_distance)
    except Exception:
        return errors
    tol = max(0.0, float(distance_tolerance))
    effective_threshold = float(final_distance_threshold) + tol
    if dist <= effective_threshold:
        return errors

    state = _extract_state(payload)
    actions = _extract_actions(payload)
    has_final_state = bool(state) and state.upper() in FINAL_STATE_NAMES
    has_final_action = any(action in FINAL_PHASE_ACTIONS for action in actions)

    if not (has_final_state or has_final_action):
        return errors

    errors.append(
        f"target_distance={dist:.2f} > {effective_threshold:.2f} "
        f"(soglia={float(final_distance_threshold):.2f} + tolleranza={tol:.2f})"
    )
    errors.append("In queste condizioni non puoi usare azioni di FINE (es. LookDown finale).")
    if has_final_state and state is not None:
        errors.append(f"STATO proposto non valido: {state}")
    if has_final_action:
        invalid_actions = [a for a in actions if a in FINAL_PHASE_ACTIONS]
        errors.append(f"Azioni non valide in questa condizione: {invalid_actions}")
    return errors


def execute_vlm_payload(controller, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Esegue un payload VLM gia' parsato."""
    state = _extract_state(payload)
    sequence = payload.get("action_sequence")
    if isinstance(sequence, list) and sequence:
        # Esegue una sequenza di azioni in ordine.
        executed = []
        event = None
        for step in sequence:
            action, params = normalize_action(step)
            if action == "Stop":
                # "Stop" e' un comando logico del planner, non un'azione AI2-THOR.
                executed.append(
                    {
                        "action": action,
                        "parameters": params,
                        "success": True,
                        "error": None,
                    }
                )
                break
            event = controller.step(action=action, **params)
            success = event.metadata.get("lastActionSuccess", True)
            error = event.metadata.get("errorMessage")
            executed.append(
                {
                    "action": action,
                    "parameters": params,
                    "success": success,
                    "error": error,
                }
            )
            # Interrompe se Stop o azione fallita (evita catene inutili).
            if action == "Stop" or not success:
                break
        last = executed[-1] if executed else {"action": "None", "parameters": {}}
        return {
            "action": last.get("action"),
            "parameters": last.get("parameters"),
            "success": last.get("success", True),
            "error": last.get("error"),
            "event": event if event is not None else getattr(controller, "last_event", None),
            "state": state,
            "actions_executed": executed,
            "planned_count": len(sequence),
        }

    # Caso base: singola azione.
    action, params = normalize_action(payload)
    if action == "Stop":
        return {
            "action": action,
            "parameters": params,
            "success": True,
            "error": None,
            "event": getattr(controller, "last_event", None),
            "state": state,
            "actions_executed": [
                {
                    "action": action,
                    "parameters": params,
                    "success": True,
                    "error": None,
                }
            ],
            "planned_count": 1,
        }
    event = controller.step(action=action, **params)
    return {
        "action": action,
        "parameters": params,
        "success": event.metadata.get("lastActionSuccess", True),
        "error": event.metadata.get("errorMessage"),
        "event": event,
        "state": state,
        "actions_executed": [
            {
                "action": action,
                "parameters": params,
                "success": event.metadata.get("lastActionSuccess", True),
                "error": event.metadata.get("errorMessage"),
            }
        ],
        "planned_count": 1,
    }


def execute_vlm_action(controller, vlm_output: str) -> Dict[str, Any]:
    """Parsa output VLM e chiama controller.step()."""
    # Flusso:
    # 1) parse del JSON VLM,
    # 2) se c'e' una sequenza, esegue ogni step finche' fallisce o Stop,
    # 3) ritorna dettagli di esecuzione (azioni, successi, errori, event).
    payload = parse_vlm_json(vlm_output)
    return execute_vlm_payload(controller, payload)
