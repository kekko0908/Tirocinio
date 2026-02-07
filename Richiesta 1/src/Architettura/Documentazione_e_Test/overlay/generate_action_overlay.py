"""Crea overlay con azione VLM sui frame YOLO-World."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from Architettura.paths import ARTIFACTS_ROOT

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


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


def _apply_pattern(pattern: str, index: int) -> str:
    def repl(match: re.Match) -> str:
        width = match.group(1)
        if width:
            return f"{index:0{int(width)}d}"
        return str(index)

    return re.sub(r"%0?(\d*)d", repl, pattern)


def _frame_indices(input_dir: Path, pattern: str) -> list[int]:
    regex = _pattern_regex(pattern)
    indices = []
    for path in sorted(input_dir.glob(_pattern_to_glob(pattern))):
        if not path.is_file():
            continue
        m = regex.match(path.name)
        if not m:
            continue
        try:
            indices.append(int(m.group(1)))
        except Exception:
            continue
    return sorted(indices)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    # Rimuove code fences se presenti.
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    # Cerca il primo blocco JSON.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    payload = match.group(0)
    try:
        return json.loads(payload)
    except Exception:
        return None


def _format_action(action_data: dict | None) -> str:
    if not action_data:
        return "AZIONE: (non disponibile)"
    if "action_sequence" in action_data and action_data["action_sequence"]:
        action = action_data["action_sequence"][0].get("action")
        params = action_data["action_sequence"][0].get("parameters", {}) or {}
    else:
        action = action_data.get("action")
        params = action_data.get("parameters", {}) or {}
    if not action:
        return "AZIONE: (non disponibile)"
    if "degrees" in params:
        return f"AZIONE: {action} (deg={params['degrees']})"
    if "moveMagnitude" in params:
        return f"AZIONE: {action} (mag={params['moveMagnitude']})"
    return f"AZIONE: {action}"


def _format_state(action_data: dict | None) -> str:
    if not action_data:
        return "STATE: []"
    state = action_data.get("state")
    if state is None:
        state = action_data.get("STATO")
    if state is None:
        state = action_data.get("stato")
    if not state:
        return "STATE: []"
    return f"STATE: [{state}]"


def _load_font(size: int) -> "ImageFont.ImageFont | None":
    if ImageFont is None:
        return None
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_label(
    img: "Image.Image",
    text: str,
    font: "ImageFont.ImageFont | None",
) -> "Image.Image":
    draw = ImageDraw.Draw(img)
    padding = 8
    lines = text.splitlines() if text else [""]
    line_sizes = []
    max_w = 0
    total_h = 0
    spacing = 4
    for line in lines:
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = draw.textsize(line, font=font)
        line_sizes.append((text_w, text_h))
        max_w = max(max_w, text_w)
        total_h += text_h
    if len(lines) > 1:
        total_h += spacing * (len(lines) - 1)
    x0, y0 = 10, 10
    x1, y1 = x0 + max_w + padding * 2, y0 + total_h + padding * 2
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    y = y0 + padding
    for (line, (_, lh)) in zip(lines, line_sizes):
        draw.text((x0 + padding, y), line, fill=(255, 255, 255), font=font)
        y += lh + spacing
    return img


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crea overlay con azione VLM sui frame YOLO-World."
    )
    parser.add_argument(
        "--yolo-dir",
        default=(ARTIFACTS_ROOT / "yolo_outputs" / "yolo").as_posix(),
        help="Cartella frame YOLO (default: Artefatti/yolo_outputs/yolo)",
    )
    parser.add_argument(
        "--yolo-pattern",
        default="yolo_%05d.png",
        help="Pattern frame YOLO (default: yolo_%05d.png)",
    )
    parser.add_argument(
        "--responses-dir",
        default=(ARTIFACTS_ROOT / "vlm_outputs" / "responses").as_posix(),
        help="Cartella risposte VLM (default: Artefatti/vlm_outputs/responses)",
    )
    parser.add_argument(
        "--response-pattern",
        default="response_%05d.json",
        help="Pattern risposte (default: response_%05d.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=(ARTIFACTS_ROOT / "Documentazione" / "overlays").as_posix(),
        help="Output overlay (default: Artefatti/Documentazione/overlays)",
    )
    parser.add_argument(
        "--output-root",
        default=(ARTIFACTS_ROOT / "Documentazione").as_posix(),
        help="Root output overlay (default: Artefatti/Documentazione)",
    )
    parser.add_argument(
        "--sim-name",
        default="",
        help="Nome cartella simulazione (es. SIM_1). Se presente, usa output_root/SIM_X.",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=1,
        help="Sposta azione di N frame (default: 1, azione al frame successivo)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=38,
        help="Dimensione testo overlay (default: 24)",
    )
    args = parser.parse_args()

    if Image is None or ImageDraw is None:
        print("PIL non disponibile: impossibile creare overlay.")
        return 2

    yolo_dir = Path(args.yolo_dir)
    responses_dir = Path(args.responses_dir)
    output_root = Path(args.output_root)
    if args.sim_name:
        output_dir = output_root / args.sim_name / "overlays"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = _load_font(args.font_size)

    indices = _frame_indices(yolo_dir, args.yolo_pattern)
    if not indices:
        print(f"Nessun frame YOLO trovato in {yolo_dir}.")
        return 1

    yolo_regex = _pattern_regex(args.yolo_pattern)
    response_regex = _pattern_regex(args.response_pattern)

    for idx in indices:
        src_name = _apply_pattern(args.yolo_pattern, idx)
        src_path = yolo_dir / src_name
        if not src_path.exists():
            # fallback per pattern non %05d
            matches = [p for p in yolo_dir.glob(_pattern_to_glob(args.yolo_pattern)) if yolo_regex.match(p.name)]
            src_candidates = [p for p in matches if p.name == src_name]
            if src_candidates:
                src_path = src_candidates[0]
            else:
                continue

        response_idx = idx - args.shift
        action_text = "AZIONE: (inizio)"
        state_text = "STATE: []"
        if response_idx >= 0:
            resp_name = _apply_pattern(args.response_pattern, response_idx)
            resp_path = responses_dir / resp_name
            if not resp_path.exists():
                # fallback ricerca
                resp_candidates = [
                    p
                    for p in responses_dir.glob(_pattern_to_glob(args.response_pattern))
                    if response_regex.match(p.name)
                ]
                resp_path = next((p for p in resp_candidates if p.name == resp_name), None)
            if resp_path and resp_path.exists():
                try:
                    payload = json.loads(resp_path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                response_text = payload.get("response", "")
                action_data = _extract_json(response_text)
                action_text = _format_action(action_data)
                state_text = _format_state(action_data)

        action_text = f"{action_text}\n{state_text}"

        img = Image.open(src_path.as_posix()).convert("RGB")
        img = _draw_label(img, action_text, font=font)
        out_name = src_path.name.replace("yolo_", "yolo_action_")
        img.save((output_dir / out_name).as_posix())

    print(f"Overlay creati in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
