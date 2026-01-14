# Scopo: nomi stato e descrizioni FSM centralizzati.

STATE_SELECT_TARGET = "SELECT_TARGET"
STATE_EXPLORE = "EXPLORE"
STATE_SEARCH = "SEARCH"
STATE_SEARCH_NEAR = "SEARCH_NEAR"
STATE_NAVIGATE = "NAVIGATE"
STATE_APPROACH = "APPROACH"
STATE_LOCALIZE = "LOCALIZE"
STATE_DONE = "DONE"
STATE_FAIL = "FAIL"

STATE_DESCRIPTIONS = {
    STATE_EXPLORE: "Explore the scene and scan to increase coverage.",
    STATE_SEARCH: "Search the target with frequent scans and careful moves.",
    STATE_SEARCH_NEAR: "Search near the reference object with local scans.",
    STATE_NAVIGATE: "Navigate around obstacles to reach a far target.",
    STATE_APPROACH: "Center the target and reduce distance.",
    STATE_LOCALIZE: "Stabilize detection and save evidence.",
}
