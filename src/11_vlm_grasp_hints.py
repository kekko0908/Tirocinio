import json
import argparse
from pathlib import Path
from PIL import Image
import re
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

DEFAULT_MODEL = "google/gemma-3-4b-it"

SYSTEM_PROMPT = """Sei un assistente per robot grasping in simulazione (AI2-THOR/ManipulaTHOR).
Dato un frame RGB e opzionalmente una bounding box YOLO (x1,y1,x2,y2 in pixel), devi proporre un grasp plan SEMPLICE.
Puoi ricevere anche un "Contesto extra" in testo (es. depth e punto 3D stimato): usalo per rendere i suggerimenti più consistenti.

Devi restituire SOLO JSON valido, senza testo extra, con questo schema:

{
  "target_label": "Apple",
  "grasp_pixel": [u, v],
  "approach": "from_edge|from_agent_side|from_left|from_right|from_front|from_back|top_down",
  "prefer_vertical_descend": true|false,
  "hover_height_hint_m": 0.30,
  "xz_offset_hint_m": [dx, dz],
  "notes": "max 1 riga"
}

Regole:
- grasp_pixel deve stare dentro l’oggetto (se bbox presente, dentro bbox).
- Se l’oggetto è su un piano (countertop/island), spesso prefer_vertical_descend=false e approach=from_edge.
- hover_height_hint_m è un valore relativo “sopra l’altezza dell’oggetto” (0.15–0.45).
- dx,dz piccoli (-0.20..+0.20), per evitare collisioni con il piano.
- Se nel contesto extra vedi che l'oggetto è su island/counter, proponi offset che evita il bordo (spesso dz negativo) e hover più alto.
"""

def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / "src").exists() and (p / "data").exists():
            return p
    for p in [cwd] + list(cwd.parents):
        if (p / "src").exists():
            return p
    return cwd

def load_bbox(bbox_path: Path):
    if not bbox_path.exists():
        return None
    data = json.loads(bbox_path.read_text(encoding="utf-8"))
    if "bbox" in data:
        return data["bbox"]
    keys = ["x1", "y1", "x2", "y2"]
    if all(k in data for k in keys):
        return [data["x1"], data["y1"], data["x2"], data["y2"]]
    return None

def load_bbox_from_state(state_path: Path):
    """
    Supporta:
    - centroid_state.json prodotto da src/03_yolo_seg_centroid.py (state['yolo']['bbox_xyxy'])
    - oppure json con chiavi bbox/x1..y2 (come load_bbox)
    """
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(data, dict):
        yolo = data.get("yolo")
        if isinstance(yolo, dict):
            bb = yolo.get("bbox_xyxy")
            if isinstance(bb, list) and len(bb) == 4:
                return bb

    # fallback: schema generico
    if "bbox" in data:
        return data["bbox"]
    keys = ["x1", "y1", "x2", "y2"]
    if all(k in data for k in keys):
        return [data["x1"], data["y1"], data["x2"], data["y2"]]
    return None

def resolve_image_path(img_arg: str, frames_dir_arg: str) -> Path:
    root = find_project_root()
    cwd = Path.cwd().resolve()

    frames_dir = Path(frames_dir_arg)
    if not frames_dir.is_absolute():
        frames_dir = (root / frames_dir).resolve()

    if img_arg:
        p = Path(img_arg)
        if p.is_absolute() and p.exists():
            return p.resolve()

        p1 = (cwd / p).resolve()
        if p1.exists():
            return p1

        p2 = (root / p).resolve()
        if p2.exists():
            return p2

    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"Nessuna PNG in {frames_dir}. CWD={cwd}. ROOT={root}.")
    return pngs[-1]

def extract_json_anywhere(text: str):
    """
    Estrae JSON in modo robusto:
    1) Cerca blocco ```json ... ```
    2) Se non c'è, cerca primo {...} e prova parse
    """
    # 1) fenced ```json
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        s = m.group(1)
        try:
            return json.loads(s)
        except Exception:
            pass

    # 2) fallback: primo {...} più grande
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = text[start:end+1]
        try:
            return json.loads(s)
        except Exception:
            pass

    return None

def load_json_safe(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--image", default="", help="Se omesso prende l'ultima PNG in frames_dir.")
    ap.add_argument("--frames_dir", default="data/frames/10_arm")
    ap.add_argument("--bbox_json", default="", help="Opzionale: bbox YOLO in json.")
    ap.add_argument("--bbox_state", default="", help="Opzionale: json di stato (es. data/state/centroid_state.json) da cui estrarre bbox.")
    ap.add_argument("--context_state", default="", help="Opzionale: json (es. data/state/pregrasp_point.json) da includere come contesto testuale.")
    ap.add_argument("--out", default="data/vlm/grasp_hints.json")
    ap.add_argument("--raw_out", default="data/vlm/raw_output.txt")
    args = ap.parse_args()

    root = find_project_root()
    image_path = resolve_image_path(args.image, args.frames_dir)

    print(f"[INFO] CWD : {Path.cwd().resolve()}")
    print(f"[INFO] ROOT: {root}")
    print(f"[INFO] IMG : {image_path}")

    bbox = None
    if args.bbox_state:
        bbox = load_bbox_from_state((root / args.bbox_state).resolve())
    if bbox is None and args.bbox_json:
        bbox = load_bbox((root / args.bbox_json).resolve())
    img = Image.open(image_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(args.model, token=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    user_text = "Guarda l'immagine e proponi un grasp plan JSON."
    if bbox:
        user_text += f"\nBBox YOLO (x1,y1,x2,y2): {bbox}"
    if args.context_state:
        ctx_path = (root / args.context_state).resolve()
        ctx = load_json_safe(ctx_path)
        if ctx is not None:
            # Limita la dimensione del contesto per non esplodere i token.
            ctx_txt = json.dumps(ctx, ensure_ascii=False)
            if len(ctx_txt) > 1800:
                ctx_txt = ctx_txt[:1800] + "..."
            user_text += "\nContesto extra (JSON): " + ctx_txt

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)

    text = processor.decode(out[0], skip_special_tokens=True)

    # salva sempre output raw
    raw_path = root / args.raw_out
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text, encoding="utf-8")
    print(f"[OK] Raw output salvato in: {raw_path}")

    hints = extract_json_anywhere(text)

    # se non riesce, fallback (così il pipeline non si blocca)
    if hints is None:
        hints = {
            "target_label": "Apple",
            "grasp_pixel": [W//2 if (W:=640) else 320, H//2 if (H:=480) else 240],
            "approach": "from_edge",
            "prefer_vertical_descend": False,
            "hover_height_hint_m": 0.30,
            "xz_offset_hint_m": [0.0, -0.10],
            "notes": "Fallback: JSON non estratto. Controlla raw_output.txt"
        }
        print("[WARN] Non sono riuscito a parsare JSON. Ho scritto un fallback hints.")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(hints, indent=2), encoding="utf-8")

    print(f"[OK] Hints JSON salvato in: {out_path}")
    print(json.dumps(hints, indent=2))

if __name__ == "__main__":
    main()
