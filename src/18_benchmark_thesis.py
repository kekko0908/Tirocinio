import time
import json
import torch
import cv2
import math
import numpy as np
import re
import random
import sys
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from ai2thor.controller import Controller
from transformers import AutoProcessor, AutoModelForCausalLM

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
BASE_OUTPUT_DIR = ROOT / "data" / "Yolo_vs_VLM"
if BASE_OUTPUT_DIR.exists(): shutil.rmtree(BASE_OUTPUT_DIR)
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(BASE_OUTPUT_DIR / "oggetti_stanza").mkdir()

MODEL_ID_VLM = "google/gemma-3-4b-it"

# --- LISTA OGGETTI FISSA PER LA TESI ---
TARGET_LIST_NAMES = ["Apple", "Lettuce", "Plate", "Mug"]

# --- UTILS GEOMETRICHE ---
def dist_xz(p1, p2):
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["z"] - p2["z"])**2)

def get_specific_objects(controller):
    """
    Cerca SOLO gli oggetti specificati nella lista TARGET_LIST_NAMES.
    """
    print(f"[SETUP] Cerco oggetti specifici: {TARGET_LIST_NAMES}")
    objs = controller.last_event.metadata["objects"]
    
    found_candidates = []
    found_types = set()
    
    # Mischiamo per variare quale specifica mela prendiamo se ce ne sono tante
    random.shuffle(objs)
    
    # Cerchiamo un esemplare per ogni tipo richiesto
    for target_type in TARGET_LIST_NAMES:
        # Cerca un oggetto di quel tipo che sia visibile/raggiungibile (almeno pickupable)
        match = next((o for o in objs if o["objectType"] == target_type and o["pickupable"]), None)
        
        if match:
            found_candidates.append(match)
            found_types.add(target_type)
        else:
            print(f"[WARN] Oggetto '{target_type}' non trovato nella scena o non raccoglibile!")

    # Salva JSON
    json_path = BASE_OUTPUT_DIR / "oggetti_stanza" / "oggetti.json"
    data = [{"objectType": o["objectType"], "objectId": o["objectId"]} for o in found_candidates]
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"[SETUP] Trovati {len(found_candidates)}/{len(TARGET_LIST_NAMES)} oggetti.")
    return found_candidates

def teleport_and_center(controller, target_obj):
    center = target_obj["axisAlignedBoundingBox"]["center"]
    target_id = target_obj["objectId"]
    
    # 1. Trova posizioni
    controller.step("GetReachablePositions")
    reachable = controller.last_event.metadata.get("actionReturn", [])
    if not reachable: return False
    
    # CERCA POSIZIONE OTTIMALE (0.45m - 0.65m)
    candidates = [p for p in reachable if 0.45 < dist_xz(p, center) < 0.65]
    
    if candidates:
        best_pos = random.choice(candidates)
    else:
        # FALLBACK: Più vicino possibile ma > 0.35m
        reachable.sort(key=lambda p: dist_xz(p, center))
        valid_fallback = [p for p in reachable if dist_xz(p, center) > 0.35]
        best_pos = valid_fallback[0] if valid_fallback else reachable[0]
        print(f"[NAV WARN] Fallback position dist={dist_xz(best_pos, center):.2f}m")

    # 2. Teleport
    random_yaw = random.uniform(0, 360)
    controller.step(action="TeleportFull", x=best_pos["x"], y=best_pos["y"], z=best_pos["z"], rotation={"x":0, "y":random_yaw, "z":0}, horizon=30.0, standing=True)
    
    # 3. Ricerca (Search)
    print("   [NAV] Cerco target...")
    for _ in range(12):
        if target_id in controller.last_event.instance_masks: break
        controller.step("RotateRight", degrees=30)
    
    if target_id not in controller.last_event.instance_masks:
        print("   [NAV FAIL] Target non visibile.")
        return False
    
    # 4. Centramento (Centering)
    print("   [NAV] Centro target...")
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

# --- METRICHE ---
def compute_iou(boxA, boxB):
    if not boxA or not boxB: return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0.0

def draw_box(img, box, color, label):
    if not box: return
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# --- VLM ---
class VLM_Predictor:
    def __init__(self):
        print("[VLM] Init Gemma...")
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_ID_VLM, token=True, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID_VLM, token=True, torch_dtype=torch.bfloat16, device_map="auto")
            self.active = True
        except: self.active = False
    def predict(self, image_rgb, target_name):
        if not self.active: return None
        pil_img = Image.fromarray(image_rgb)
        # Prompt diretto
        prompt = f"Find the '{target_name}'. Return bbox [ymin, xmin, ymax, xmax] 0-1000."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
        inputs = self.processor(text=inputs, images=pil_img, return_tensors="pt").to(self.model.device)
        with torch.no_grad(): out = self.model.generate(**inputs, max_new_tokens=200)
        decoded = self.processor.decode(out[0], skip_special_tokens=True)
        try:
            match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", decoded)
            if match:
                y1, x1, y2, x2 = map(int, match.groups())
                return [(x1/1000)*640, (y1/1000)*480, (x2/1000)*640, (y2/1000)*480]
        except: pass
        return None

# --- MAIN ---
def main():
    print(">>> BENCHMARK TESI (Custom List) <<<")
    controller = Controller(scene="FloorPlan1", agentMode="default", width=640, height=480, renderInstanceSegmentation=True)
    vlm_engine = VLM_Predictor()
    yolo_engine = YoloSegEngine(model_path="yolo11x-seg.pt")
    
    # USA LA FUNZIONE DI RICERCA SPECIFICA
    targets = get_specific_objects(controller)
    
    results = []
    
    for i, obj in enumerate(targets):
        name = obj["objectType"]
        print(f"\n--- SIM {i+1}/{len(targets)}: {name} ---")
        sim_dir = BASE_OUTPUT_DIR / f"Sim_{i+1}_{name}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        if not teleport_and_center(controller, obj):
            print("Skipped (Nav Fail).")
            continue
            
        time.sleep(0.5)
        evt = controller.last_event
        rgb = evt.frame
        
        # Save Input
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        input_path = sim_dir / "input.png"
        cv2.imwrite(str(input_path), bgr)
        
        # GT
        mask = evt.instance_masks.get(obj["objectId"])
        if mask is None: continue
        rows, cols = np.where(mask)
        gt_box = [float(np.min(cols)), float(np.min(rows)), float(np.max(cols)), float(np.max(rows))]
        
        # Run Models
        t0 = time.time(); vlm_box = vlm_engine.predict(rgb, name); vlm_t = time.time()-t0
        t0 = time.time(); yolo_box, yolo_dbg = yolo_engine.analyze(str(input_path), name); yolo_t = time.time()-t0
        
        # Metrics
        iou_v = compute_iou(vlm_box, gt_box)
        iou_y = compute_iou(yolo_box, gt_box)
        win = "YOLO" if iou_y >= iou_v else "VLM"
        if iou_y==0 and iou_v==0: win = "NONE"
        
        print(f"Result: VLM={iou_v:.2f}, YOLO={iou_y:.2f} -> {win}")
        print(f"YOLO Debug: {yolo_dbg}")
        
        # Save Data
        res = {"obj": name, "vlm": {"iou": iou_v, "time": vlm_t}, "yolo": {"iou": iou_y, "time": yolo_t, "dbg": yolo_dbg}, "winner": win}
        results.append(res)
        with open(sim_dir / "data.json", "w") as f: json.dump(res, f, indent=2)
        
        # Draw
        vis = bgr.copy()
        draw_box(vis, gt_box, (0,255,0), "GT")
        draw_box(vis, vlm_box, (0,0,255), f"VLM {iou_v:.2f}")
        draw_box(vis, yolo_box, (255,0,0), f"YOLO {iou_y:.2f}")
        cv2.imwrite(str(sim_dir / "result.png"), vis)
        
    # Final Report
    if results:
        summary = {
            "avg_vlm": np.mean([r["vlm"]["iou"] for r in results]),
            "avg_yolo": np.mean([r["yolo"]["iou"] for r in results]),
            "yolo_wins": len([r for r in results if r["winner"]=="YOLO"]),
            "vlm_wins": len([r for r in results if r["winner"]=="VLM"]),
            "details": results
        }
        with open(BASE_OUTPUT_DIR / "FINAL.json", "w") as f: json.dump(summary, f, indent=2)
        print(f"\nFINITO. Report in {BASE_OUTPUT_DIR}")
    else:
        print("\nNESSUN RISULTATO UTILE.")
    
    controller.stop()

if __name__ == "__main__":
    main()