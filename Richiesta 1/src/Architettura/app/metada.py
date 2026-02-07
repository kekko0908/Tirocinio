"""Stampa a schermo i campi disponibili nel metadata di AI2-THOR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Architettura.ai2thor.init_controller import SimConfig, create_controller


def main() -> None:
    config = SimConfig(scene="FloorPlan1")
    controller = create_controller(config)
    try:
        event = controller.step(action="Pass")
        metadata = getattr(event, "metadata", {}) or {}
        out_dir = Path(__file__).resolve().parent
        raw_path = out_dir / "metadata_dump.json"
        raw_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        descriptions = {
            "objects": "Lista di oggetti presenti nella scena con proprietà e visibilità.",
            "isSceneAtRest": "True se la scena è stabile (nessun movimento fisico).",
            "agent": "Stato dell'agente (posizione, rotazione, camera, ecc.).",
            "heldObjectPose": "Pose dell'oggetto tenuto in mano dall'agente.",
            "arm": "Stato del braccio (pose/giunti) dell'agente.",
            "fov": "Campo visivo della camera principale.",
            "cameraPosition": "Posizione attuale della camera principale.",
            "cameraOrthSize": "Dimensione ortografica della camera (se usata).",
            "thirdPartyCameras": "Lista delle third-party cameras registrate.",
            "collided": "True se l'ultima azione ha causato collisione.",
            "collidedObjects": "Oggetti con cui l'agente ha colliso.",
            "inventoryObjects": "Oggetti nell'inventario dell'agente.",
            "sceneName": "Nome della scena corrente.",
            "lastAction": "Ultima azione eseguita.",
            "errorMessage": "Messaggio di errore dell'ultima azione.",
            "errorCode": "Codice di errore dell'ultima azione.",
            "lastActionSuccess": "True se l'ultima azione è riuscita.",
            "screenWidth": "Larghezza del frame renderizzato.",
            "screenHeight": "Altezza del frame renderizzato.",
            "agentId": "ID numerico dell'agente.",
            "depthFormat": "Formato della depth map.",
            "colors": "Colori usati per la segmentazione semantica.",
            "flatSurfacesOnGrid": "Superfici piane rilevate sulla griglia.",
            "distances": "Distanze pre-calcolate (se presenti).",
            "normals": "Normali della scena (se presenti).",
            "isOpenableGrid": "Griglia degli oggetti apribili (se presente).",
            "segmentedObjectIds": "ID oggetti segmentati nel frame.",
            "objectIdsInBox": "ID oggetti presenti nei bounding box.",
            "actionIntReturn": "Ritorno intero dell'ultima action (se presente).",
            "actionFloatReturn": "Ritorno float dell'ultima action (se presente).",
            "actionStringsReturn": "Ritorno stringhe dell'ultima action (se presente).",
            "actionFloatsReturn": "Ritorno lista float dell'ultima action (se presente).",
            "actionVector3sReturn": "Ritorno lista Vector3 dell'ultima action (se presente).",
            "visibleRange": "Range di visibilità della camera.",
            "currentTime": "Tempo corrente della simulazione.",
            "sceneBounds": "Bounding box della scena.",
            "actionReturn": "Ritorno generico dell'ultima action (se presente).",
        }

        annotated = []
        for key in sorted(metadata.keys()):
            annotated.append(
                {
                    "description": descriptions.get(
                        key, "Descrizione non disponibile."
                    ),
                    "field": key,
                    "value": metadata.get(key),
                }
            )

        annotated_path = out_dir / "metadata_dump_annotated.json"
        annotated_path.write_text(
            json.dumps(annotated, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Metadata salvato in: {raw_path}")
        print(f"Metadata annotato in: {annotated_path}")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
