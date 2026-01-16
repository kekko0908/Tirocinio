# Scopo: logiche specifiche dello stato LOCALIZE.
from typing import Optional, Tuple

from .state_names import STATE_SEARCH


def next_state_localize(state_steps: int, max_localize_steps: int) -> Optional[Tuple[str, str]]:
    """
    Gestisce la transizione dallo stato LOCALIZE.
    Ritorna SEARCH se supera il massimo di step consentiti.
    Mantiene lo stato se non ci sono condizioni di uscita.
    """
    if state_steps >= int(max_localize_steps):
        return STATE_SEARCH, "localize_timeout"
    return None
