"""Client VLM minimale per Ollama con test rapido."""

from __future__ import annotations

import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple

import requests
from PIL import Image


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")


def _ollama_api_url(path: str) -> str:
    base = OLLAMA_URL.rstrip("/")
    marker = "/api/"
    if marker in base:
        root = base.split(marker, 1)[0]
    else:
        root = base
    return f"{root}/api/{path.lstrip('/')}"


class VLMEngine:
    def __init__(self, model_id: str = "llama3.2-vision", action_set: Optional[List[str]] = None):
        self.model_id = model_id
        self.action_set = action_set or []
        print(f"[OLLAMA] Connesso al modello: {model_id}", flush=True)
        print(f"[OLLAMA] Endpoint: {OLLAMA_URL}", flush=True)

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _run_inference(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        data = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        }

        if image is not None:
            data["images"] = [self._image_to_base64(image)]

        try:
            response = requests.post(OLLAMA_URL, json=data, timeout=180)
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("response", "")).strip()
        except Exception as exc:
            print(f"[ERRORE OLLAMA] {exc}", flush=True)
            return "{}"

    def _runtime_info(self) -> Optional[dict[str, Any]]:
        ps_url = _ollama_api_url("ps")
        try:
            response = requests.get(ps_url, timeout=10)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", [])
            for model in models:
                name = str(model.get("name", ""))
                if name == self.model_id or name.split(":", 1)[0] == self.model_id.split(":", 1)[0]:
                    return model
            return None
        except Exception as exc:
            print(f"[OLLAMA] Check GPU non disponibile ({exc})", flush=True)
            return None

    def print_runtime_info(self, when: str) -> None:
        model = self._runtime_info()
        if not model:
            print(f"[OLLAMA] Runtime ({when}): modello non presente in /api/ps.", flush=True)
            return

        size_vram = int(model.get("size_vram", 0) or 0)
        size_total = int(model.get("size", 0) or 0)
        vram_gb = size_vram / (1024 ** 3)
        total_gb = size_total / (1024 ** 3)
        gpu_state = "SI" if size_vram > 0 else "NO"
        print(
            (
                f"[OLLAMA] Runtime ({when}): GPU={gpu_state}, "
                f"VRAM={vram_gb:.2f}GB, MODEL_RAM+VRAM={total_gb:.2f}GB, "
                f"expires_at={model.get('expires_at')}"
            ),
            flush=True,
        )

    def plan_subgoals(self, goal_text: str) -> str:
        prompt = f"Goal: {goal_text}. Return a JSON list of subgoals."
        return self._run_inference(prompt)

    def predict_bbox(self, image: Image.Image, target_label: str) -> Tuple[str, str]:
        prompt = (
            f"Find the {target_label}. "
            "Return bounding box [ymin, xmin, ymax, xmax] in JSON."
        )
        text = self._run_inference(prompt, image)
        return text, text

    def ask(self, text: str, image: Optional[Image.Image] = None) -> str:
        return self._run_inference(text, image=image)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test rapido Ollama VLM.")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "llama3.2-vision"),
        help="Model id Ollama (es. llama3.2-vision).",
    )
    parser.add_argument(
        "--prompt",
        default="Come stai?",
        help="Prompt di test.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path immagine da inviare al modello (JPEG/PNG).",
    )
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Mostra se il modello occupa VRAM via /api/ps prima e dopo la richiesta.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Stampa output in JSON.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    engine = VLMEngine(model_id=args.model)

    image_obj: Optional[Image.Image] = None
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[ERRORE] Immagine non trovata: {img_path}", flush=True)
            return 2
        try:
            with Image.open(img_path.as_posix()) as im:
                image_obj = im.convert("RGB")
        except Exception as exc:
            print(f"[ERRORE] Impossibile aprire immagine {img_path}: {exc}", flush=True)
            return 2

    if args.check_gpu:
        engine.print_runtime_info("prima")

    print("[OLLAMA] Invio richiesta di test...", flush=True)
    answer = engine.ask(args.prompt, image=image_obj)

    if args.check_gpu:
        engine.print_runtime_info("dopo")

    if args.json:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "prompt": args.prompt,
                    "image": args.image,
                    "response": answer,
                },
                ensure_ascii=False,
            )
        )
    else:
        print("\n--- RISPOSTA ---")
        print(answer if answer else "<vuota>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
