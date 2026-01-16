# Scopo: logiche specifiche dello stato EXPLORE.
from typing import Optional, Tuple

from .state_names import STATE_SEARCH


def apply_explore_macros(
    action_mgr,
    mem_summary,
    sensor,
    total_steps: int,
    probe_positive: bool,
    probe_conf: int,
    hint_strong_conf: int,
) -> None:
    """
    Applica macro di esplorazione in base a coverage e hint.
    Enqueue rotazioni se l'hint non e forte o stabile.
    Aggiunge avanzamenti quando il frontale e sicuro.
    """
    # Se non abbiamo un hint forte, continuiamo con le macro-scan di coverage.
    if not (probe_positive and probe_conf >= int(hint_strong_conf)):
        action_mgr.maybe_enqueue_scan(mem_summary["coverage_ratio"], total_steps)
    action_mgr.maybe_enqueue_advance(sensor["dist_front_m"], mem_summary["coverage_ratio"])


def next_state_explore(
    candidate_seen: bool, state_steps: int, max_explore_steps: int
) -> Optional[Tuple[str, str]]:
    """
    Decide la transizione dallo stato EXPLORE.
    Passa a SEARCH se il candidato e visto.
    Va in timeout verso SEARCH se supera il limite step.
    """
    if candidate_seen:
        return STATE_SEARCH, "candidate_seen"
    if state_steps >= int(max_explore_steps):
        return STATE_SEARCH, "explore_timeout"
    return None
