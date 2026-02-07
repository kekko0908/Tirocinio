"""Genera un video dai frame YOLO-World salvati negli artefatti."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from Architettura.paths import ARTIFACTS_ROOT


def _pattern_to_glob(pattern: str) -> str:
    return re.sub(r"%0?\d*d", "*", pattern)


def _pattern_regex(pattern: str) -> re.Pattern:
    parts = []
    last = 0
    for match in re.finditer(r"%0?(\d*)d", pattern):
        parts.append(re.escape(pattern[last : match.start()]))
        width = match.group(1)
        if width:
            parts.append(rf"(\d{{{int(width)}}})")
        else:
            parts.append(r"(\d+)")
        last = match.end()
    parts.append(re.escape(pattern[last:]))
    return re.compile(rf"^{''.join(parts)}$")


def _find_frames(input_dir: Path, pattern: str) -> list[Path]:
    glob_pattern = _pattern_to_glob(pattern)
    frames = sorted(input_dir.glob(glob_pattern))
    return [p for p in frames if p.is_file()]


def _min_index(frames: list[Path], pattern: str) -> int:
    regex = _pattern_regex(pattern)
    indices = []
    for frame in frames:
        m = regex.match(frame.name)
        if not m:
            continue
        try:
            indices.append(int(m.group(1)))
        except Exception:
            continue
    return min(indices) if indices else 0


def _next_sim_dir(output_root: Path) -> Path:
    existing = []
    for entry in output_root.glob("SIM_*"):
        if not entry.is_dir():
            continue
        suffix = entry.name.replace("SIM_", "")
        if suffix.isdigit():
            existing.append(int(suffix))
    next_idx = max(existing, default=0) + 1
    return output_root / f"SIM_{next_idx}"


def _run_ffmpeg(cmd: list[str]) -> int:
    print("Eseguo:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crea un video dai frame YOLO-World (png) in Artefatti."
    )
    parser.add_argument(
        "--yolo-dir",
        default="",
        help="Cartella dei frame (default: SIM_X/overlays se sim-name, altrimenti yolo_outputs/yolo)",
    )
    parser.add_argument(
        "--semantic-dir",
        default=(
            ARTIFACTS_ROOT / "vision_outputs" / "first_person" / "semantic"
        ).as_posix(),
        help="Cartella dei frame semantic (default: Artefatti/vision_outputs/first_person/semantic)",
    )
    parser.add_argument(
        "--output-root",
        default=(ARTIFACTS_ROOT / "Documentazione").as_posix(),
        help="Root output (default: Artefatti/Documentazione)",
    )
    parser.add_argument(
        "--sim-name",
        default="",
        help="Nome cartella simulazione (es. SIM_001). Se vuoto, crea la prossima.",
    )
    parser.add_argument(
        "--yolo-pattern",
        default="",
        help="Pattern frame (default: yolo_action_%05d.png se overlays, altrimenti yolo_%05d.png)",
    )
    parser.add_argument(
        "--semantic-pattern",
        default="semantic_frame_%05d.png",
        help="Pattern frame semantic (default: semantic_frame_%05d.png)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frame rate del video (default: 30)",
    )
    parser.add_argument(
        "--yolo-output-name",
        default="yolo.mp4",
        help="Nome video YOLO (default: yolo.mp4)",
    )
    parser.add_argument(
        "--combined-output-name",
        default="semantic_yolo.mp4",
        help="Nome video combinato (default: semantic_yolo.mp4)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sim_dir = output_root / args.sim_name if args.sim_name else _next_sim_dir(output_root)
    overlays_dir = sim_dir / "overlays"
    if args.yolo_dir:
        yolo_dir = Path(args.yolo_dir)
    else:
        yolo_dir = overlays_dir if args.sim_name else (ARTIFACTS_ROOT / "yolo_outputs" / "yolo")
    if args.yolo_pattern:
        yolo_pattern = args.yolo_pattern
    else:
        yolo_pattern = "yolo_action_%05d.png" if yolo_dir == overlays_dir else "yolo_%05d.png"
    semantic_dir = Path(args.semantic_dir)
    videos_dir = sim_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    frames = _find_frames(yolo_dir, yolo_pattern)
    if not frames:
        print(f"Nessun frame trovato in {yolo_dir} con pattern '{yolo_pattern}'.")
        return 1

    if shutil.which("ffmpeg") is None:
        print("ffmpeg non trovato nel PATH. Installalo o usa un altro metodo.")
        return 2

    yolo_output_path = videos_dir / args.yolo_output_name
    yolo_input = str(yolo_dir / yolo_pattern)
    yolo_start = _min_index(frames, yolo_pattern)
    cmd_yolo = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-start_number",
        str(yolo_start),
        "-i",
        yolo_input,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        yolo_output_path.as_posix(),
    ]
    rc = _run_ffmpeg(cmd_yolo)
    if rc != 0:
        print(f"ffmpeg (YOLO) ha fallito con codice {rc}.")
        return rc

    semantic_frames = _find_frames(semantic_dir, args.semantic_pattern)
    if not semantic_frames:
        print(
            f"Nessun frame semantic trovato in {semantic_dir} con pattern '{args.semantic_pattern}'."
        )
        return 1

    combined_output_path = videos_dir / args.combined_output_name
    semantic_input = str(semantic_dir / args.semantic_pattern)
    semantic_start = _min_index(semantic_frames, args.semantic_pattern)
    cmd_combined = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-start_number",
        str(semantic_start),
        "-i",
        semantic_input,
        "-framerate",
        str(args.fps),
        "-start_number",
        str(yolo_start),
        "-i",
        yolo_input,
        "-filter_complex",
        "hstack=inputs=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        combined_output_path.as_posix(),
    ]
    rc = _run_ffmpeg(cmd_combined)
    if rc != 0:
        print(f"ffmpeg (COMBINATO) ha fallito con codice {rc}.")
        return rc

    print(f"Video salvati in: {videos_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
