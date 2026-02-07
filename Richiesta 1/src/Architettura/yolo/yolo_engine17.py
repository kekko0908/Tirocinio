import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from Architettura.paths import YOLO_MODELS_DIR

# Definiamo cartella output debug fissa
DEBUG_DIR = Path(__file__).resolve().parents[2] / "data" / "Yolo_vs_VLM"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_model_path(model_path: str | os.PathLike[str]) -> str:
    p = Path(model_path)
    if p.exists():
        return str(p)

    if not p.is_absolute():
        local = YOLO_MODELS_DIR / p
        if local.exists():
            return str(local)

    return str(p)




# --- MAPPA AGGIORNATA ---
THOR_TO_YOLO_MAP = {
   
    # STOVIGLIE / OGGETTI
    "WineBottle": ["bottle", "wine glass"],  # NUOVO
    
}


def bbox_from_mask(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0.5)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


class YoloSegEngine:
    def __init__(self, model_path: str = "yolo11x-seg.pt"):
        resolved_path = _resolve_model_path(model_path)
        print(f"[YOLO-17] Richiesto modello: {model_path}")
        print(f"[YOLO-17] Path risolto: {resolved_path}")

        # Controllo se il file esiste localmente
        if os.path.exists(resolved_path):
            print(f"[YOLO-17] Trovato file locale: {resolved_path}")
        else:
            print("[YOLO-17] File non trovato, Ultralytics provera' a scaricarlo...")

        try:
            # Caricamento Diretto
            self.model = YOLO(resolved_path)

            # Verifica che sia davvero caricato
            print(f"[YOLO-17] MODELLO CARICATO CORRETTAMENTE: {self.model.ckpt_path}")

        except Exception as e:
            print("\n" + "=" * 60)
            print(f"[ERRORE CRITICO] Impossibile caricare {model_path}!")
            print(f"Errore: {e}")
            print("SOLUZIONE: Esegui 'pip install -U ultralytics' per aggiornare la libreria.")
            print("=" * 60 + "\n")
            raise e

    def analyze(self, source, target_label: str = "apple", conf: float = 0.20):
        # Recupera sinonimi
        valid_labels = THOR_TO_YOLO_MAP.get(target_label, [target_label.lower()])
        if target_label.lower() not in valid_labels:
            valid_labels.append(target_label.lower())

        print(f"\n[YOLO-17] Cerco '{target_label}'. Mapping: {valid_labels}")

        img_debug = None
        inference_source = source

        if isinstance(source, str):
            if not os.path.exists(source):
                return None, f"File missing: {source}"
            img_debug = cv2.imread(source)
            inference_source = img_debug
        else:
            img_debug = source

        # Inferenza
        results = self.model.predict(
            source=inference_source, conf=conf, imgsz=1024, verbose=False
        )

        if not results:
            return None, "No results"
        res0 = results[0]
        boxes = res0.boxes
        if boxes is None or len(boxes) == 0:
            return None, "No detections"

        detections = []
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy().astype(float)
        names = getattr(res0, "names", None) or {}

        # Debug Visual
        debug_vis = img_debug.copy() if img_debug is not None else None

        for i in range(len(confs)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            label = names.get(cls_ids[i], str(cls_ids[i])).lower()
            score = confs[i]
            is_valid = label in valid_labels

            color = (0, 255, 0) if is_valid else (0, 0, 255)
            if debug_vis is not None:
                cv2.rectangle(debug_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    debug_vis,
                    f"{label} {score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

            detections.append(
                {"label": label, "score": score, "bbox": [x1, y1, x2, y2], "is_valid": is_valid}
            )

        if debug_vis is not None:
            cv2.imwrite(str(DEBUG_DIR / "DEBUG_YOLO_internal.png"), debug_vis)

        # Selezione
        candidates = [d for d in detections if d["is_valid"]]
        if candidates:
            candidates.sort(key=lambda d: d["score"], reverse=True)
            best = candidates[0]
            return best["bbox"], f"Found '{best['label']}' conf={best['score']:.2f}"

        seen = ", ".join([d["label"] for d in detections])
        return None, f"Target '{target_label}' not found. Seen: {seen}"


if __name__ == "__main__":
    print("Modulo Engine pronto.")
