# Scopo
Pipeline modulare per AI2-THOR con planning (Gemma 3), navigazione guidata da VLM, detection YOLO e telemetria frame-by-frame.

# Moduli
- `pipeline_subgoal.py`: orchestratore (CLI + ciclo percezione/azione).
- `pipeline_modules/utils.py`: parsing, normalizzazione, json, IoU.
- `pipeline_modules/sensors.py`: stato robot e distanze da depth.
- `pipeline_modules/memory.py`: exploration memory e summary.
- `pipeline_modules/action_manager.py`: safety, anti-loop e macro-azioni.
- `pipeline_modules/vlm.py`: planner, navigazione VLM, bbox annotator.
- `pipeline_modules/yolo.py`: detection YOLO (bbox/centroide/maschera).
- `pipeline_modules/visualization.py`: overlay telemetria e frame evidenze.

# Esecuzione
Esempio base:
```
python PiPeline-SubGoal/pipeline_subgoal.py --goal "Cercami la mela" --scene FloorPlan1
```

Output salvati in:
- `PiPeline-SubGoal/outputs/frames/`
- `PiPeline-SubGoal/outputs/debug/`
- `PiPeline-SubGoal/outputs/plan.json`
- `PiPeline-SubGoal/outputs/run_log.json`
- `PiPeline-SubGoal/outputs/detection.json`

# Note operative
- YOLO e' invocato ad ogni step per default (`--yolo_every 1`).
- Depth e' abilitato di default per stimare distanze di prossimita'.
