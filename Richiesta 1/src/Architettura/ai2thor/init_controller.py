"""Inizializzazione del sistema."""

from dataclasses import dataclass, field

from ai2thor.controller import Controller


@dataclass
class SimConfig:
    scene: str = "FloorPlan7"  # Mappa/stanza da caricare (es. FloorPlan1).
    width: int = 1920  # Larghezza del frame RGB in pixel.
    height: int = 1080  # Altezza del frame RGB in pixel.
    headless: bool = False  # True = nessuna finestra grafica; utile su server.
    server_timeout: float = 120.0  # Timeout per singole azioni step (secondi).
    server_start_timeout: float = 300.0  # Timeout per avvio server (secondi).
    grid_size: float = 0.10  # Passo base della griglia per movimenti discreti.Più piccolo = movimento più fine.
    visibility_distance: float = 1.5  # Distanza max per oggetti "visibili",Più piccolo = movimento più fine.
    snapToGrid: bool = True  # Forza movimenti/rotazioni su griglia e angoli discreti. Rende il movimento più “pulito” e ripetibile.
    agent_position: dict | None = field(
        default_factory=lambda: {"x": -0.15, "y": 0.9, "z": 1.70}
    )  # Posizione iniziale agente (x,y,z).
    agent_rotation: dict | None = field(
        default_factory=lambda: {"x": -0.0, "y": 180, "z": 0.0}
    )  # Rotazione iniziale agente (x,y,z).
    agent_horizon: float | None = 0  # Inclinazione iniziale camera (gradi).


def create_controller(config: SimConfig) -> Controller:
    """Crea e ritorna il controller AI2-THOR."""
    controller = Controller(
        scene=config.scene,  # Scena iTHOR da caricare.
        width=config.width,  # Larghezza rendering camera principale.
        height=config.height,  # Altezza rendering camera principale.
        headless=config.headless,  # Esegue senza finestra grafica.
        server_timeout=config.server_timeout,  # Timeout step/azioni.
        server_start_timeout=config.server_start_timeout,  # Timeout avvio server.
        gridSize=config.grid_size,  # Risoluzione griglia di movimento.
        visibilityDistance=config.visibility_distance,  # Raggio visibilita' oggetti.
        renderDepthImage=True,  # Abilita depth_frame.
        renderInstanceSegmentation=True,  # Abilita instance_segmentation_frame.
        renderSemanticSegmentation=True,  # Abilita semantic_segmentation_frame.
        snapToGrid=False  # Forza snapping a griglia (movimenti/rotazioni).
    )
    has_teleport_pose = (
        config.agent_position is not None
        or config.agent_rotation is not None
        or config.agent_horizon is not None
    )
    if has_teleport_pose:
        print(
            f"[INIT] Teleport request: pos={config.agent_position} "
            f"rot={config.agent_rotation} horizon={config.agent_horizon}",
            flush=True,
        )
        event = controller.step(
            action="Teleport",
            position=config.agent_position,
            rotation=config.agent_rotation,
            horizon=config.agent_horizon,
        )
        success = event.metadata.get("lastActionSuccess", True)
        error = event.metadata.get("errorMessage")
        print(
            f"[INIT] Teleport success={success} error={error}",
            flush=True,
        )
    else:
        print("[INIT] Teleport skipped: pose iniziale non configurata.", flush=True)
    return controller
