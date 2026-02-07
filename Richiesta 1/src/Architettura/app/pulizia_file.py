"""Pulizia dei file dentro la cartella Artefatti."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Architettura.paths import ARTIFACTS_ROOT


def clear_artifacts(root: Path) -> int:
    """Elimina file e cartelle sotto root, preservando Documentazione."""
    if not root.exists():
        return 0
    removed = 0
    doc_root = root / "Documentazione"
    for path in root.rglob("*"):
        if path.is_file():
            if doc_root in path.parents:
                continue
            path.unlink()
            removed += 1
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            if path == doc_root or doc_root in path.parents:
                continue
            try:
                path.rmdir()
            except OSError:
                pass
    return removed


def main() -> None:
    removed = clear_artifacts(ARTIFACTS_ROOT)
    print(f"Pulizia completata. File rimossi: {removed}")


if __name__ == "__main__":
    main()
