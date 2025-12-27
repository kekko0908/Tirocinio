import sys
from pathlib import Path

# Aggiungiamo la root al path per importare yolo_engine_17 se sta nella root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    from yolo_engine_17 import YoloSegEngine
except ImportError:
    print("[ERR] Impossibile importare yolo_engine_17. Assicurati che sia nella cartella principale.")
    sys.exit(1)

class YoloDetector:
    def __init__(self, model_path="yolo11x-seg.pt"):
        print("[YOLO] Inizializzazione motore...")
        self.engine = YoloSegEngine(model_path=model_path)

    def predict(self, image_path, target_name):
        """
        Esegue YOLO sull'immagine salvata su disco.
        Returns: bbox [xmin, ymin, xmax, ymax], debug_info
        """
        # Chiama il metodo analyze del tuo engine originale
        box, debug_info = self.engine.analyze(str(image_path), target_name)
        return box, debug_info