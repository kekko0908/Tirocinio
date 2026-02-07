"""Percorsi base condivisi per l'architettura."""

from __future__ import annotations

from pathlib import Path


def _find_project_root() -> Path:
    """Risalgo fino a 'src' e uso 'src' come root progetto."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "src":
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
ARTIFACTS_ROOT = PROJECT_ROOT / "Artefatti"
YOLO_MODELS_DIR = PROJECT_ROOT.parent.parent / "MODELLI YOLO"


def rel_to_project(path: Path) -> str:
    """Ritorna un path relativo al project root, se possibile."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
