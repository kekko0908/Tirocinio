# Scopo: utilita condivise tra stati.
from collections import deque


def confirmed_from_hits(hits: deque, k: int, n: int) -> bool:
    """
    Verifica se la finestra di hit raggiunge la soglia.
    Controlla che ci siano almeno n elementi utili.
    Ritorna True se la somma degli hit risulta >= k.
    """
    if len(hits) < int(n):
        return False
    return sum(hits) >= int(k)


def reduce_hint_deg(current_deg: int, base_deg: int) -> int:
    """
    Riduce gradualmente l'angolo suggerito per lo scan.
    Costruisce una sequenza di gradi decrescenti.
    Ritorna il prossimo grado piu piccolo disponibile.
    """
    # Riduce gradualmente l'angolo per evitare overshoot.
    seq = [int(base_deg)]
    for d in [30, 20, 15, 10, 5]:
        if d < int(base_deg) and d not in seq:
            seq.append(d)
    cur = int(current_deg) if current_deg else seq[0]
    for d in seq:
        if d < cur:
            return d
    return seq[-1]
