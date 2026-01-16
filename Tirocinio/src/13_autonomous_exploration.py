import time
import math
import random
import numpy as np
import cv2
import json
import shutil
from pathlib import Path
from ai2thor.controller import Controller 
from common import EnvConfig, print_ok, print_warn, print_err

# --- CONFIGURAZIONE ---
ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = ROOT / "data" / "frames" / "13_autonav"
STATE_DIR = ROOT / "data" / "state"
OUTPUT_STATE_PATH = STATE_DIR / "arm_best_pose_state.json"

if FRAMES_DIR.exists(): shutil.rmtree(FRAMES_DIR)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TYPE = "Apple"
# Distanza ottimale: abbastanza vicina per vedere bene, ma non troppo per bloccare il braccio
MIN_DIST_TO_GRASP = 0.60  
OBSTACLE_THRESHOLD = 0.45 
MAX_STEPS = 800
DEBUG_EVERY = 1

# --- UTILS MATEMATICHE ---

def dist_xz(p1, p2):
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["z"] - p2["z"])**2)

# --- UTILS VISUAL DEBUG ---

def save_debug_frame(controller, step_name):
    if hasattr(controller.last_event, "frame"):
        img = controller.last_event.frame
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        path = FRAMES_DIR / f"{step_name}.png"
        cv2.imwrite(str(path), img_bgr)
        print(f"[DEBUG] Frame salvato: {step_name}.png")

# --- UTILS NAVIGAZIONE ---

def get_depth_center_distance(event):
    if event.depth_frame is None: return 10.0
    d = event.depth_frame
    h, w = d.shape
    # Guarda al centro dell'immagine
    center_patch = d[h//2 - 10 : h//2 + 10, w//2 - 10 : w//2 + 10]
    valid = center_patch[center_patch > 0]
    return np.mean(valid) if len(valid) > 0 else 10.0

def find_target_in_view(event, target_type):
    meta = event.metadata["objects"]
    visible = [o for o in meta if o["objectType"] == target_type and o["visible"]]
    if not visible: return False, None, None
    
    # Prendi il più vicino
    target = sorted(visible, key=lambda x: x["distance"])[0]
    tid = target["objectId"]
    
    cx = 320
    if event.instance_masks and tid in event.instance_masks:
        ys, xs = np.where(event.instance_masks[tid])
        if len(xs) > 0: cx = int(np.mean(xs))
            
    return True, tid, cx

def debug_step(step, state, seen, cx, dist, obstacle_dist, stuck_counter, action=None):
    if DEBUG_EVERY <= 0 or step % DEBUG_EVERY != 0:
        return
    parts = [f"[DBG] S{step:03d}", state, f"seen={seen}"]
    if cx is not None:
        parts.append(f"cx={cx}")
    if dist is not None:
        parts.append(f"dist={dist:.2f}")
    parts.append(f"obs={obstacle_dist:.2f}")
    parts.append(f"stuck={stuck_counter}")
    if action:
        parts.append(f"action={action}")
    print(" | ".join(parts))

# --- MAIN LOOP ---

def main():
    print_ok("1. Avvio fase NAVIGAZIONE (Ricerca & Approccio)...")
    
    # Usiamo 'default' perché è meglio per camminare senza incastrarsi
    controller = Controller(
        scene="FloorPlan1",
        agentMode="default",
        width=640, height=480,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        snapToGrid=False 
    )
    
    state = "SEARCH"
    target_id = None
    
    step_count = 0
    stuck_counter = 0
    last_pos = None
    success_nav = False
    
    # Parametri di allineamento fine
    CENTER_TOLERANCE = 15
    CENTER_X = 320
    
    try:
        while step_count < MAX_STEPS:
            step_count += 1
            event = controller.last_event
            
            # Rilevamento blocco (Stuck Detection)
            curr_pos = event.metadata["agent"]["position"]
            if last_pos and dist_xz(curr_pos, last_pos) < 0.01 and state == "APPROACH":
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_pos = curr_pos
            
            obstacle_dist = get_depth_center_distance(event)
            seen, tid, cx = find_target_in_view(event, TARGET_TYPE)
            
            # Log periodico
            if step_count % 20 == 0:
                print(f"[{state}] Step {step_count} | Obs: {obstacle_dist:.2f}m")

            # --- MACCHINA A STATI ---

            if state == "SEARCH":
                if seen:
                    print_ok(f"Target VISTO ({tid}). Inseguo...")
                    debug_step(step_count, state, seen, cx, None, obstacle_dist, stuck_counter, "to_APPROACH")
                    state = "APPROACH"
                    target_id = tid
                    continue
                
                # Strategia: Random Walk con evitamento ostacoli
                if obstacle_dist < OBSTACLE_THRESHOLD:
                    action = "RotateRight" if random.random() > 0.5 else "RotateLeft"
                    debug_step(step_count, state, seen, cx, None, obstacle_dist, stuck_counter, action)
                    controller.step(action=action, degrees=random.randint(45, 90))
                else:
                    if random.random() < 0.1:
                        debug_step(step_count, state, seen, cx, None, obstacle_dist, stuck_counter, "RotateRight(30)")
                        controller.step(action="RotateRight", degrees=30)
                    else:
                        debug_step(step_count, state, seen, cx, None, obstacle_dist, stuck_counter, "MoveAhead(0.25)")
                        controller.step(action="MoveAhead", moveMagnitude=0.25)
                        
            elif state == "APPROACH":
                if not seen:
                    # Perso target? Cercalo localmente
                    debug_step(step_count, state, seen, None, None, obstacle_dist, stuck_counter, "RotateRight(10)")
                    controller.step("RotateRight", degrees=10)
                    if stuck_counter > 5: state = "SEARCH"
                    continue
                
                obj_meta = next((o for o in event.metadata["objects"] if o["objectId"] == target_id), None)
                if not obj_meta:
                    debug_step(step_count, state, seen, cx, None, obstacle_dist, stuck_counter, "no_obj_meta")
                    continue
                dist = obj_meta["distance"]
                debug_step(step_count, state, seen, cx, dist, obstacle_dist, stuck_counter, None)
                
                # Se bloccato mentre vede il target, prova manovra evasiva
                if stuck_counter > 3:
                    print_warn(f"BLOCCATO. Manovra evasiva...")
                    evasive = "MoveRight" if random.random() > 0.5 else "MoveLeft"
                    debug_step(step_count, state, seen, cx, dist, obstacle_dist, stuck_counter, evasive)
                    controller.step(action=evasive, moveMagnitude=0.25)
                    stuck_counter = 0
                    continue

                # 1. ROTAZIONE (Centramento preciso)
                if cx < (CENTER_X - CENTER_TOLERANCE): 
                    debug_step(step_count, state, seen, cx, dist, obstacle_dist, stuck_counter, "RotateLeft(5)")
                    controller.step("RotateLeft", degrees=5)
                    continue
                elif cx > (CENTER_X + CENTER_TOLERANCE): 
                    debug_step(step_count, state, seen, cx, dist, obstacle_dist, stuck_counter, "RotateRight(5)")
                    controller.step("RotateRight", degrees=5)
                    continue
                
                # 2. AVANZAMENTO (Solo se centrato)
                if dist > MIN_DIST_TO_GRASP:
                    if obstacle_dist < 0.35:
                        print_warn("Stop anticipato (ostacolo vicinissimo).")
                        success_nav = True
                        break
                    
                    # Rallenta quando è vicino
                    step_sz = 0.15 if dist < 1.2 else 0.25
                    debug_step(step_count, state, seen, cx, dist, obstacle_dist, stuck_counter, f"MoveAhead({step_sz:.2f})")
                    ev = controller.step("MoveAhead", moveMagnitude=step_sz)
                    
                    # Se sbatte contro qualcosa (es. angolo del tavolo)
                    if not ev.metadata["lastActionSuccess"]:
                        debug_step(step_count, state, seen, cx, dist, obstacle_dist, stuck_counter, "MoveRight(0.10)")
                        controller.step("MoveRight", moveMagnitude=0.10)
                else:
                    print_ok(f"Distanza raggiunta ({dist:.2f}m) e CENTRATO!")
                    success_nav = True
                    break
        
        # --- FINE: SALVATAGGIO DATI ---
        if success_nav:
            print_ok("Navigazione completata con successo.")
            last_event = controller.last_event
            agent_pose = last_event.metadata["agent"]
            
            # Recupera info precise sul target per il prossimo script
            obj_meta = next((o for o in last_event.metadata["objects"] if o["objectId"] == target_id), None)
            target_world_center = obj_meta["axisAlignedBoundingBox"]["center"]
            
            # Creiamo il pacchetto dati per il braccio
            final_state = {
                "scene": "FloorPlan1",
                "target": {
                    "objectId": target_id,
                    "objectType": TARGET_TYPE
                },
                "agent": {
                    "position": agent_pose["position"],
                    "rotation": agent_pose["rotation"],
                    # Forziamo orizzonte basso per il braccio, così guarda il tavolo
                    "cameraHorizon": 30.0 
                },
                "bbox_center_world": target_world_center
            }
            
            # Salvataggio su disco
            with open(OUTPUT_STATE_PATH, "w") as f:
                json.dump(final_state, f, indent=2)
                
            print_ok(f"Stato salvato in: {OUTPUT_STATE_PATH}")
            save_debug_frame(controller, "13_nav_final_view")
            print_ok("Ora puoi eseguire lo script di manipolazione.")
            
        else:
            print_err("Navigazione fallita (timeout o target non trovato).")

    except KeyboardInterrupt:
        print("Interrotto dall'utente.")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()
