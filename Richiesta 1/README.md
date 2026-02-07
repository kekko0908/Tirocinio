# Sistema VLM + YOLO + AI2-THOR

Questa cartella contiene l'architettura runtime per navigazione AI2-THOR guidata da VLM, con supporto YOLO per overlay/target matching e pipeline di documentazione automatica.

## Struttura principale

- `src/Architettura/app/start.py`: entrypoint runtime (loop decisionale, retry/validazione, summary, documentazione finale).
- `src/Architettura/ai2thor/init_controller.py`: configurazione `SimConfig` e creazione controller.
- `src/Architettura/ai2thor/utils.py`: salvataggio artefatti first/third-person + integrazione YOLO.
- `src/Architettura/ai2thor/context.py`: costruzione contesto strutturato per VLM.
- `src/Architettura/vlm/client.py`: factory client VLM (HF o GGUF).
- `src/Architettura/vlm/runner.py`: prompt/context/images, invocazione VLM, salvataggio risposte.
- `src/Architettura/vlm/agent/action_executor.py`: parse robusto JSON + esecuzione azioni.
- `src/Architettura/yolo/utils.py`: overlay YOLO (`save_yolo_png`) e metadati target.
- `src/Architettura/Documentazione_e_Test/run/start_documentation.py`: overlay+video+pulizia artefatti.
- `src/Architettura/Documentazione_e_Test/run/start_test.py`: batch di simulazioni.
- `src/Architettura/Documentazione_e_Test/run/benchmark_models.py`: benchmark multi-modello con report.

## Output artefatti

Root artefatti:

- `src/Artefatti/`

Sottocartelle principali:

- `src/Artefatti/vision_outputs/first_person/rgb/`
- `src/Artefatti/vision_outputs/first_person/depth/`
- `src/Artefatti/vision_outputs/first_person/instance/`
- `src/Artefatti/vision_outputs/first_person/semantic/`
- `src/Artefatti/vision_outputs/third_person/`
- `src/Artefatti/yolo_outputs/yolo/`
- `src/Artefatti/vlm_outputs/responses/`
- `src/Artefatti/diagnostic/`
- `src/Artefatti/Documentazione/SIM_n/`

## Esecuzione (da `Richiesta 1/src`)

Esempio base:

```bash
python3 Architettura/app/start.py "cerca la bottiglia"
```

Esempio con variabili runtime:

```bash
export VLMPROMPT=5
export VLM_MODEL=qwen-vl2.5
export VLM_MMPROJ=qwen-vl-mmproj-2.5
export VLM_PRESET=accurate
export VLM_MAX_STEPS=170
export VLM_VALIDATION_RETRIES=2
export VLM_FINAL_DISTANCE_THRESHOLD=0.8
export VLM_FINAL_DISTANCE_TOLERANCE=0.03
python3 Architettura/app/start.py "cerca la bottiglia"
```

Variabili aggiuntive utili:

- `BENCHMARK_SUMMARY_PATH`: salva JSON summary di run.
- `DOC_SIM_NAME`: forza il nome output documentazione (`SIM_x`).
- `YOLO_TARGET_CONF`: confidenza minima per matching target.

## Documentazione tecnica

Sorgente primaria:

- `docs/Documentazione_Sistema_VLM_YOLO_AI2THOR.md`

Build PDF (da `Richiesta 1/`):

```bash
python3 docs/build_system_doc_pdf.py
```

Comando esplicito input/output:

```bash
python3 docs/build_system_doc_pdf.py \
  --in docs/Documentazione_Sistema_VLM_YOLO_AI2THOR.md \
  --out docs/Documentazione_Sistema_VLM_YOLO_AI2THOR_20260206.pdf
```

Convenzione naming PDF versionato:

- `Documentazione_Sistema_VLM_YOLO_AI2THOR_YYYYMMDD.pdf`

