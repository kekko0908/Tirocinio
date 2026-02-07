import argparse
import re
import sys
from pathlib import Path

from Architettura.paths import ARTIFACTS_ROOT

# Aggiungiamo la root al path per importare yolo_engine_17 se sta nella root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
MODELS_DIR = Path("/home/kekko/Scrivania/Tirocinio-Root/MODELLI YOLO")
YOLO_OUT_DIR = ARTIFACTS_ROOT / "yolo_outputs" / "yolo"

try:
    from yolo_engine_17 import YoloSegEngine
except ImportError:
    print("[ERR] Impossibile importare yolo_engine_17. Assicurati che sia nella cartella principale.")
    sys.exit(1)


class YoloDetector:
    def __init__(self, model_path: str = "yolo11x-seg.pt"):
        print("[YOLO] Inizializzazione motore...")
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = MODELS_DIR / model_path
        self.engine = YoloSegEngine(model_path=model_path)

    def predict(self, image_source, target_name, conf: float = 0.20):
        """
        Esegue YOLO su immagine (path o array).
        Returns: bbox [xmin, ymin, xmax, ymax], debug_info
        """
        box, debug_info = self.engine.analyze(image_source, target_name, conf=conf)
        return box, debug_info


def _parse_conf_from_debug(debug_info: str | None) -> float | None:
    if not debug_info:
        return None
    m = re.search(r"conf=([0-9.]+)", debug_info)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _draw_bbox_image(img, bbox: list[int], out_path: Path, label: str, conf: float | None) -> None:
    from PIL import Image, ImageDraw

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    r = 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), fill=(255, 0, 0))
    conf_text = f"{conf:.2f}" if conf is not None else "n/a"
    overlay = f"{label} {conf_text}"
    draw.text((x1, max(0, y1 - 12)), overlay, fill=(0, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path.as_posix())


def save_yolo_png(
    image,
    frame_idx: int,
    target_name: str,
    model_path: str = "yolo11x-seg.pt",
    output_path: Path | None = None,
    conf: float = 0.20,
    image_is_bgr: bool = False,
):
    detector = YoloDetector(model_path=model_path)
    try:
        import numpy as np
        from PIL import Image

        if isinstance(image, Image.Image):
            image_rgb = np.asarray(image.convert("RGB"))
            image_bgr = image_rgb[:, :, ::-1]
        else:
            image_arr = image
            if image_is_bgr:
                image_bgr = image_arr
                try:
                    image_rgb = image_arr[:, :, ::-1]
                except Exception:
                    image_rgb = image_arr
            else:
                image_rgb = image_arr
                try:
                    image_bgr = image_arr[:, :, ::-1]
                except Exception:
                    image_bgr = image_arr
    except Exception:
        image_rgb = image
        image_bgr = image

    box, debug_info = detector.predict(image_bgr, target_name, conf=conf)
    if output_path is None:
        output_path = YOLO_OUT_DIR / f"yolo_{frame_idx:05d}.png"
    if box:
        det_conf = _parse_conf_from_debug(debug_info)
        _draw_bbox_image(image_rgb, box, output_path, target_name, det_conf)
    else:
        try:
            from PIL import Image

            Image.fromarray(image_rgb).convert("RGB").save(output_path.as_posix())
        except Exception:
            pass
    return box, debug_info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Esegue YOLO su una foto e ritorna il bounding box del target."
    )
    parser.add_argument("image_path", help="Percorso immagine input")
    parser.add_argument("target_name", help="Nome oggetto target")
    parser.add_argument(
        "--model",
        default="yolo11x-seg.pt",
        help="Nome modello (in MODELLI YOLO) o path assoluto",
    )
    parser.add_argument(
        "--out-dir",
        default="Tirocinio/data/frames",
        help="Cartella output per immagini con bbox",
    )
    args = parser.parse_args()

    detector = YoloDetector(model_path=args.model)
    box, debug_info = detector.predict(args.image_path, args.target_name)
    print(f"[YOLO] bbox={box}")
    if debug_info is not None:
        print(f"[YOLO] debug={debug_info}")
    if box:
        conf = _parse_conf_from_debug(debug_info)
        image_path = Path(args.image_path)
        out_dir = Path(args.out_dir)
        out_name = f"{image_path.stem}_bbox{image_path.suffix}"
        from PIL import Image
        img = Image.open(image_path.as_posix()).convert("RGB")
        _draw_bbox_image(img, box, out_dir / out_name, args.target_name, conf)
        print(f"[YOLO] saved_bbox={out_dir / out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
