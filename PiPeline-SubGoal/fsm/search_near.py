# Scopo: logiche specifiche dello stato SEARCH_NEAR.
from typing import Optional, Tuple

from .state_names import STATE_APPROACH, STATE_SEARCH


def next_state_search_near(
    anchor_missing: bool,
    anchor_dist: float,
    near_radius: float,
    search_confirmed: bool,
    state_steps: int,
    max_steps: int,
) -> Optional[Tuple[str, str]]:
    """
    Decide la transizione per SEARCH_NEAR vicino all'anchor.
    Ritorna SEARCH se anchor manca o e troppo lontana.
    Passa ad APPROACH se confermato o in caso di timeout.
    """
    if anchor_missing:
        return STATE_SEARCH, "near_anchor_missing"
    if anchor_dist is not None and anchor_dist > float(near_radius):
        return STATE_SEARCH, "near_anchor_far"
    if search_confirmed:
        return STATE_APPROACH, "search_confirmed"
    if state_steps >= int(max_steps):
        return STATE_SEARCH, "near_timeout"
    return None
