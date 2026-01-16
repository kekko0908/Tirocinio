import time
import json
import os
import torch
import cv2
import math
import numpy as np
import re
import random
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from ai2thor.controller import Controller
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

# --- IMPORT YOLO ENGINE ---
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

try:
    from yolo_engine_17 import YoloSegEngine
except ImportError:
    print("[ERR] Manca yolo_engine_17.py")
    sys.exit(1)

# --- CONFIGURAZIONE ---
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "Yolo_vs_VLM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TYPE = "Apple"
MODEL_ID_VLM = "google/gemma-3-4b-it"

# --- UTILS GEOMETRICHE ---
def dist_xz(p1, p2):
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["z"] - p2["z"])**2)

def teleport_near_target_random_yaw(controller, target_type):
    print(f"[SETUP] Cerco {target_type}...")
    objs = controller.last_event.metadata["objects"]
    target = next((o for o in objs if o["objectType"] == target_type), None)
    if not target: return False, None
    center = target["axisAlignedBoundingBox"]["center"]
    controller.step("GetReachablePositions")
    reachable = controller.last_event.metadata.get("actionReturn", [])
    candidates = [p for p in reachable if 0.4 < dist_xz(p, center) < 0.75]
    if not candidates: best_pos = min(reachable, key=lambda p: dist_xz(p, center))
    else: best_pos = random.choice(candidates)
    
    print(f"[SETUP] Teleport a {dist_xz(best_pos, center):.2f}m dal target.")
    controller.step(action="TeleportFull", x=best_pos["x"], y=best_pos["y"], z=best_pos["z"], rotation={"x":0,"y":random.uniform(0,360),"z":0}, horizon=30.0, standing=True)
    return True, target["objectId"]

def find_and_center_target(controller, target_id):
    print("[SEARCH] Ricerca oggetto...")
    for _ in range(12): 
        if target_id in controller.last_event.instance_masks: break
        controller.step(action="RotateRight", degrees=30)
    
    if target_id not in controller.last_event.instance_masks: return False

    print("[CENTER] Centramento...")
    for _ in range(15):
        if target_id not in controller.last_event.instance_masks:
            controller.step("RotateLeft", degrees=10); continue
        mask = controller.last_event.instance_masks[target_id]
        cols = np.where(mask)[1]
        if len(cols) == 0: continue
        cx = int(np.mean(cols))
        if abs(cx - 320) < 30: return True
        deg = 5 if abs(cx - 320) < 50 else 10
        action = "RotateLeft" if cx < 320 else "RotateRight"
        controller.step(action, degrees=deg)
    return True 

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if float(boxAArea + boxBArea - inter) == 0: return 0.0
    return inter / float(boxAArea + boxBArea - inter)

def draw_box(img, box, color, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# --- VLM ---
class VLM_Predictor:
    def __init__(self):
        self.active = False
        self.processor = None
        self.model = None
        self.input_device = torch.device("cpu")
        self.max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "200"))

        device = self._pick_device()
        
        # Logica per quantizzazione 4-bit
        quant = os.getenv("VLM_QUANT", "auto").strip().lower()
        use_4bit = device == "cuda" and quant in {"auto", "4bit", "4-bit"} and self._bitsandbytes_available()

        if device == "cuda" and quant in {"auto", "4bit", "4-bit"} and not use_4bit:
            print("[VLM][WARN] bitsandbytes non installato: 4-bit disabilitato.")

        self._load(device=device, use_4bit=use_4bit)

    def _bitsandbytes_available(self) -> bool:
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            return False
        return True

    def _pick_device(self) -> str:
        # Verifica esplicita CUDA
        if not torch.cuda.is_available():
            print("[VLM] CUDA non disponibile. Uso CPU.")
            return "cpu"
        
        # --- MODIFICA QUI: Abbassato limite a 4GB per GPU ---
        min_gb = float(os.getenv("VLM_MIN_GPU_GB", "4.0")) 
        
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[VLM] GPU rilevata: {total_gb:.2f} GB (Richiesti min: {min_gb} GB)")
            
            if total_gb >= min_gb:
                return "cuda"
            else:
                print(f"[VLM] VRAM insufficiente ({total_gb:.2f} < {min_gb}). Fallback CPU.")
                return "cpu"
        except Exception as e:
            print(f"[VLM] Errore controllo GPU: {e}. Fallback CPU.")
            return "cpu"

    def _load(self, device: str, use_4bit: bool) -> None:
        self.input_device = torch.device("cuda:0" if device == "cuda" else "cpu")
        print(f"[VLM] Init {MODEL_ID_VLM} device={device} quant={'4bit' if use_4bit else 'none'}")

        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_ID_VLM, token=True, use_fast=True)

            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID_VLM,
                    token=True,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            elif device == "cuda":
                # Caricamento standard GPU (se non usi 4-bit)
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID_VLM,
                    token=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            else:
                # Caricamento CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID_VLM,
                    token=True,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )

            self.model.eval()
            self.active = True
        except Exception as e:
            print(f"[ERR] VLM Loading Failed: {e}")
            self.active = False

    def predict(self, image_rgb, width, height):
        if not self.active: return None
        pil_img = Image.fromarray(image_rgb)
        prompt = (
            f"You are an advanced computer vision system specialized in object localization. "
            f"Your task is to detect Apple in this image.\n"
            "INSTRUCTIONS:\n"
            "1. ANALYZE: First, identify the object by its visual features (shape, texture) distinguishing it from the background.\n"
            "2. LOCALIZE: Determine the bounding box coordinates using a normalized scale from 0 to 1000.\n"
            "   - (0,0) is the Top-Left corner.\n"
            "   - (1000,1000) is the Bottom-Right corner.\n"
            "3. FORMAT: Return strictly the list of 4 integers in this specific order: [ymin, xmin, ymax, xmax].\n"
            "   - ymin: Top edge\n"
            "   - xmin: Left edge\n"
            "   - ymax: Bottom edge\n"
            "   - xmax: Right edge\n"
            "EXAMPLE OUTPUT: [150, 320, 400, 580]\n"
            "Provide ONLY the list as output."
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
        inputs = self.processor(text=inputs, images=pil_img, return_tensors="pt")
        # Spostiamo gli input sul device corretto
        inputs = {k: (v.to(self.model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            
            decoded = self.processor.decode(out[0], skip_special_tokens=True)
            match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", decoded)
            if match:
                y1, x1, y2, x2 = map(int, match.groups())
                return [(x1/1000)*width, (y1/1000)*height, (x2/1000)*width, (y2/1000)*height]
        except Exception as e:
            print(f"[VLM ERROR] {e}")
            pass
        return None

# --- MAIN ---
def main():
    print("[INIT] Controller...")
    # NOTA: Lascia 500x500 per evitare crash su Linux
    controller = Controller(scene="FloorPlan1", agentMode="default", width=500, height=500, renderInstanceSegmentation=True)    
    
    # 1. SETUP
    success, target_id = teleport_near_target_random_yaw(controller, TARGET_TYPE)
    if not success: controller.stop(); return
    if not find_and_center_target(controller, target_id): controller.stop(); return
    
    time.sleep(0.5)
    event = controller.last_event
    rgb = event.frame # RGB
    h, w, _ = rgb.shape
    
    # --- SALVATAGGIO FRAME PULITO PER YOLO ---
    clean_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    clean_img_path = OUTPUT_DIR / "temp_yolo_input.png"
    cv2.imwrite(str(clean_img_path), clean_bgr)
    print(f"[INFO] Frame pulito salvato per YOLO: {clean_img_path}")

    # 2. GT
    mask = event.instance_masks.get(target_id)
    if mask is None: controller.stop(); return
    rows, cols = np.where(mask)
    gt_box = [float(np.min(cols)), float(np.min(rows)), float(np.max(cols)), float(np.max(rows))]

    # 3. TEST VLM
    print("\n[TEST] VLM...")
    vlm = VLM_Predictor()
    t0 = time.time()
    vlm_box = vlm.predict(rgb, w, h)
    vlm_time = time.time() - t0
    
    # 4. TEST YOLO
    print("\n[TEST] YOLO (from file)...")
    yolo_engine = YoloSegEngine() 
    t0 = time.time()
    yolo_box, debug_str = yolo_engine.analyze(str(clean_img_path), target_label="apple")
    yolo_time = time.time() - t0
    print(f"[YOLO Res]: {debug_str}")

    # 5. RESULTS
    iou_vlm = compute_iou(vlm_box, gt_box) if vlm_box else 0.0
    iou_yolo = compute_iou(yolo_box, gt_box) if yolo_box else 0.0
    
    print("\n=== RISULTATI ===")
    print(f"VLM  IoU: {iou_vlm:.4f} ({vlm_time:.2f}s)")
    print(f"YOLO IoU: {iou_yolo:.4f} ({yolo_time:.2f}s)")
    
    winner = "YOLO" if iou_yolo >= iou_vlm else "VLM"
    if iou_yolo == 0 and iou_vlm == 0: winner = "NESSUNO"
    print(f"VINCITORE: {winner}")
    
    # Visuals
    vis = clean_bgr.copy()
    draw_box(vis, gt_box, (0, 255, 0), "GT")
    if vlm_box: draw_box(vis, vlm_box, (0, 0, 255), f"VLM {iou_vlm:.2f}")
    if yolo_box: draw_box(vis, yolo_box, (255, 0, 0), f"YOLO {iou_yolo:.2f}")
    
    cv2.imwrite(str(OUTPUT_DIR / "final_result.png"), vis)
    print(f"Salvato in {OUTPUT_DIR}")
    
    controller.stop()

if __name__ == "__main__":
    main()