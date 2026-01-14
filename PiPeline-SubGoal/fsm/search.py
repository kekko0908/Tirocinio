# Scopo: logiche specifiche dello stato SEARCH.
from typing import Optional, Tuple

from .state_names import STATE_APPROACH, STATE_EXPLORE


def next_state_search(
    search_confirmed: bool, state_steps: int, max_search_steps: int
) -> Optional[Tuple[str, str]]:
    """
    Decide la transizione dallo stato SEARCH.
    Passa ad APPROACH se la ricerca e confermata.
    Passa a EXPLORE se supera max_search_steps.
    """
    if search_confirmed:
        return STATE_APPROACH, "search_confirmed"
    if state_steps >= int(max_search_steps):
        return STATE_EXPLORE, "search_timeout"
    return None
