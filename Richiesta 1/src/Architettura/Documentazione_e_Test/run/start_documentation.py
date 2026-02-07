"""Esegue overlay e video per documentare una simulazione."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Architettura.paths import ARTIFACTS_ROOT
from Architettura.app.pulizia_file import clear_artifacts


def _next_sim_name(root: Path) -> str:
    existing = []
    if root.exists():
        for entry in root.glob("SIM_*"):
            if not entry.is_dir():
                continue
            suffix = entry.name.replace("SIM_", "")
            if suffix.isdigit():
                existing.append(int(suffix))
    next_idx = max(existing, default=0) + 1
    return f"SIM_{next_idx}"


def _log_prefix() -> str:
    idx = os.environ.get("TEST_INDEX", "").strip()
    return f"[TEST {idx}]" if idx else "[TEST]"


def run_documentation(sim_name: str | None = None) -> int:
    overlay_script = Path(__file__).resolve().parents[1] / "overlay" / "generate_action_overlay.py"
    video_script = Path(__file__).resolve().parents[1] / "video" / "generate_video.py"

    doc_root = ARTIFACTS_ROOT / "Documentazione"

    if not sim_name:
        sim_name = _next_sim_name(doc_root)

    src_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{existing_pp}" if existing_pp else str(src_root)
    )

    overlay_cmd = [
        sys.executable,
        overlay_script.as_posix(),
        "--output-root",
        doc_root.as_posix(),
        "--sim-name",
        sim_name,
    ]
    video_cmd = [
        sys.executable,
        video_script.as_posix(),
        "--output-root",
        doc_root.as_posix(),
        "--sim-name",
        sim_name,
    ]

    prefix = _log_prefix()
    overlay_dir = doc_root / sim_name / "overlays"
    print(f"{prefix} [SAVE]: creo overlay in {overlay_dir}")
    overlay_rc = subprocess.run(overlay_cmd, check=False, env=env).returncode
    if overlay_rc != 0:
        print(f"{prefix} [SAVE]: overlay fallito (codice {overlay_rc}).")

    video_dir = doc_root / sim_name / "videos"
    print(f"{prefix} [SAVE]: creo video in {video_dir}")
    video_rc = subprocess.run(video_cmd, check=False, env=env).returncode
    if video_rc != 0:
        print(f"{prefix} [SAVE]: video fallito (codice {video_rc}).")

    try:
        print(f"{prefix} [REMOVE]: rimuovo file con pulizia")
        removed = clear_artifacts(ARTIFACTS_ROOT)
        print(f"{prefix} [REMOVE]: pulizia completata. File rimossi: {removed}")
    except Exception as exc:
        print(f"{prefix} [REMOVE]: pulizia fallita: {exc}")

    return 0 if overlay_rc == 0 and video_rc == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Esegue overlay e video di documentazione."
    )
    parser.add_argument(
        "--sim-name",
        default="",
        help="Nome cartella simulazione (es. SIM_1). Se vuoto, crea la prossima.",
    )
    args = parser.parse_args()
    sim_name = args.sim_name.strip() or None
    return run_documentation(sim_name=sim_name)


if __name__ == "__main__":
    raise SystemExit(main())
