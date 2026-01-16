import time
import json
import cv2
import math
import random
import shutil
import numpy as np
from pathlib import Path
from ai2thor.controller import Controller

# --- CONFIGURAZIONE ---
ROOT = Path(__file__).resolve().parent[1]
BASE_OUTPUT_DIR = ROOT / "data" / "Yolo_vs_VLM"
if BASE_OUTPUT_DIR.exists(): shutil.rmtree(BASE_OUTPUT_DIR)
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LISTA TARGET FISSA
TARGET_LIST_NAMES = ["Apple", "Lettuce", "Plate", "Mug"]

# --- UTILS NAVIGAZIONE ---
def dist_xz(p1, p2):
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["z"] - p2["z"])**2)

def teleport_and_center(controller, target_obj):
    """Gestisce la navigazione e il puntamento della camera"""
    center = target_obj["axisAlignedBoundingBox"]["center"]
    target_id = target_obj["objectId"]
    
    controller.step("GetReachablePositions")
    reachable = controller.last_event.metadata.get("actionReturn", [])
    if not reachable: return False
    
    # Cerca posizione ottimale (0.45m - 0.70m)
    candidates = [p for p in reachable if 0.45 < dist_xz(p, center) < 0.70]
    best_pos = random.choice(candidates) if candidates else reachable[0]

    # Teleport
    controller.step(action="TeleportFull", x=best_pos["x"], y=best_pos["y"], z=best_pos["z"], 
                    rotation={"x":0, "y":random.uniform(0, 360), "z":0}, horizon=30.0, standing=True)
    
    # Ricerca visiva (rotazione)
    for _ in range(12):
        if target_id in controller.last_event.instance_masks: break
        controller.step("RotateRight", degrees=30)
    
    if target_id not in controller.last_event.instance_masks: return False

    # Centramento fine
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
    if not boxA or not boxB: return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0.0

def draw_result(img, gt, vlm, yolo, iou_v, iou_y):
    vis = img.copy()
    if gt: 
        cv2.rectangle(vis, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 255, 0), 2)
        cv2.putText(vis, "GT", (int(gt[0]), int(gt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if vlm: 
        cv2.rectangle(vis, (int(vlm[0]), int(vlm[1])), (int(vlm[2]), int(vlm[3])), (0, 0, 255), 2)
        cv2.putText(vis, f"VLM {iou_v:.2f}", (int(vlm[0]), int(vlm[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    if yolo: 
        cv2.rectangle(vis, (int(yolo[0]), int(yolo[1])), (int(yolo[2]), int(yolo[3])), (255, 0, 0), 2)
        cv2.putText(vis, f"YOLO {iou_y:.2f}", (int(yolo[0]), int(yolo[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return vis

# --- MAIN ---
def main():
    print(">>> AVVIO MAIN CONTROLLER <<<")
    
    # 1. Inizializzazione Moduli
    controller = Controller(scene="FloorPlan1", width=640, height=480, renderInstanceSegmentation=True)
    yolo_detector = YoloDetector()  # Istanza modulo YOLO
    vlm_predictor = VLMPredictor()  # Istanza modulo VLM
    
    # 2. Ricerca oggetti nella scena
    all_objs = controller.last_event.metadata["objects"]
    targets = []
    for name in TARGET_LIST_NAMES:
        candidates = [o for o in all_objs if o["objectType"] == name and o["pickupable"]]
        if candidates: targets.append(random.choice(candidates))
    
    print(f"Target Trovati: {[t['objectType'] for t in targets]}")
    results = []

    # 3. Loop Principale
    for i, obj in enumerate(targets):
        name = obj["objectType"]
        print(f"\n--- ELABORAZIONE {name} ({i+1}/{len(targets)}) ---")
        
        # A. Navigazione
        if not teleport_and_center(controller, obj):
            print(" [SKIP] Navigazione fallita.")
            continue
            
        # B. Setup Cartelle e File
        sim_dir = BASE_OUTPUT_DIR / f"Sim_{i+1}_{name}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        evt = controller.last_event
        rgb_img = evt.frame
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        
        # SALVATAGGIO FOTO INIZIALE (Input per tutti i moduli)
        init_path = sim_dir / "init.png"
        cv2.imwrite(str(init_path), bgr_img)
        print(f" [IO] Foto salvata in: {init_path}")

        # C. Esecuzione Moduli (Responsabilità Divise)
        
        # 1. Ground Truth (dal simulatore)
        gt_box = get_ground_truth_bbox(evt, obj["objectId"])
        
        # 2. YOLO (dal file immagine)
        yolo_box, yolo_dbg = yolo_detector.predict(init_path, name)
        
        # 3. VLM (dall'array RGB - o file se preferisci ricaricarlo)
        vlm_box = vlm_predictor.predict(rgb_img, name)
        
        # D. Calcolo Metriche e Salvataggio
        iou_v = compute_iou(vlm_box, gt_box)
        iou_y = compute_iou(yolo_box, gt_box)
        
        print(f" [RES] GT: {gt_box is not None} | YOLO IoU: {iou_y:.2f} | VLM IoU: {iou_v:.2f}")
        
        res_data = {
            "object": name,
            "ground_truth_box": gt_box,
            "yolo": {"box": yolo_box, "iou": iou_y, "debug": yolo_dbg},
            "vlm": {"box": vlm_box, "iou": iou_v},
            "winner": "YOLO" if iou_y >= iou_v else "VLM"
        }
        
        with open(sim_dir / "analysis.json", "w") as f:
            json.dump(res_data, f, indent=2)
            
        # E. Disegno Risultato Finale
        vis_img = draw_result(bgr_img, gt_box, vlm_box, yolo_box, iou_v, iou_y)
        cv2.imwrite(str(sim_dir / "final_result.png"), vis_img)
        results.append(res_data)

    # Report Finale
    print("\n--- BENCHMARK COMPLETATO ---")
    controller.stop()

if __name__ == "__main__":
    main()