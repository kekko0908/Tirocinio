"""Avvio minimo: inizializza AI2-THOR, stampa primitive, muove il robot."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Architettura.ai2thor.init_controller import SimConfig, create_controller
from Architettura.vlm.client import (
    create_vlm,
    resolve_model_id,
    resolve_mmproj_path,
)
from Architettura.vlm.vlm_prompt import select_system_prompt
from Architettura.vlm.runner import run_vlm_on_step
from Architettura.vlm.agent.action_executor import (
    execute_vlm_payload,
    parse_vlm_json,
    validate_vlm_semantics,
)
from Architettura.Documentazione_e_Test.run.start_documentation import run_documentation
from Architettura.ai2thor.utils import print_primitives
from Architettura.ai2thor.utils import save_artifacts, update_third_person_camera


def _build_validation_retry_prompt(base_prompt: str, errors: list[str]) -> str:
    lines = "\n".join(f"- {err}" for err in errors)
    return (
        f"{base_prompt}\n\n"
        "VALIDATION_ERROR:\n"
        f"{lines}\n\n"
        "Correggi il piano. Rispondi SOLO con JSON valido nel formato richiesto."
    )


def _utc_iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _compute_p95(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    clean.sort()
    idx = max(0, min(len(clean) - 1, int(round(0.95 * (len(clean) - 1)))))
    return clean[idx]


def _write_benchmark_summary(path_text: str, payload: dict) -> None:
    if not path_text:
        return
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _write_benchmark_phase(path_text: str, payload: dict) -> None:
    if not path_text:
        return
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def main():
    run_started_ts = time.time()
    run_started_iso = _utc_iso_from_ts(run_started_ts)
    summary_path = os.environ.get("BENCHMARK_SUMMARY_PATH", "").strip()
    phase_path = os.environ.get("BENCHMARK_PHASE_PATH", "").strip()
    doc_sim_name_raw = os.environ.get("DOC_SIM_NAME", "").strip()
    doc_sim_name = doc_sim_name_raw or None
    goal_input = sys.argv[1].strip() if len(sys.argv) > 1 else ""

    telemetry = {
        "steps_total": 0,
        "action_failures_count": 0,
        "json_parse_failures_count": 0,
        "semantic_validation_failures_count": 0,
        "retry_attempts_total": 0,
        "vlm_inference_seconds": [],
        "model_load_sec": None,
        "initial_target_distance_m": None,
        "min_target_distance_m": None,
        "final_target_distance_m": None,
        "time_to_target_sec": None,
    }
    exit_reason = "max_steps"
    target_reached = False
    run_ended_ts = run_started_ts

    print("Avvio controller AI2-THOR...", flush=True)
    config = SimConfig(scene="FloorPlan1")
    print(
        f"Config: {config.width}x{config.height} headless={config.headless}",
        flush=True,
    )
    controller = create_controller(config)
    print("Controller pronto.", flush=True)
    controller.step(
        action="AddThirdPartyCamera",
        position={"x": -1.25, "y": 1, "z": -1},
        rotation={"x": 90, "y": 0, "z": 0},
        fieldOfView=90,
    )

    delay_sec = 1
    run_vlm = True
    # Esempio GGUF: modelli_gguf/Qwen2.5-VL-7B-Instruct-Q8_0.gguf
    vlm_model = os.environ.get("VLM_MODEL", "qwen-vl2.5")
    vlm_mmproj_alias = os.environ.get("VLM_MMPROJ", "qwen-vl-mmproj-2.5")
    try:
        vlm_bench_start_delay_sec = float(
            os.environ.get("VLM_BENCH_START_DELAY_SEC", "0")
        )
    except ValueError:
        vlm_bench_start_delay_sec = 0.0
    if vlm_bench_start_delay_sec < 0:
        vlm_bench_start_delay_sec = 0.0
    try:
        max_steps = int(os.environ.get("VLM_MAX_STEPS", "170"))
    except ValueError:
        max_steps = 170
    if max_steps <= 0:
        max_steps = 170
    resolved_vlm_model = resolve_model_id(vlm_model)
    use_gguf = str(resolved_vlm_model).strip().lower().endswith(".gguf")
    clip_model_path = resolve_mmproj_path(vlm_mmproj_alias) if use_gguf else None
    vlm_preset = os.environ.get("VLM_PRESET", "accurate").strip().lower()
    # Preset "accurate": riduce variabilita' e deriva del testo.
    if use_gguf:
        vlm_preset_map = {
            "accurate": {
                "max_new_tokens": 900,
                "temperature": 0.1,
                "top_p": 0.9,
                "n_ctx": 4096,
                "n_batch": 256,
                "n_gpu_layers": -1,
            },
            "fast": {
                "max_new_tokens": 400,
                "temperature": 0.1,
                "top_p": 0.9,
                "n_ctx": 2048,
                "n_batch": 128,
                "n_gpu_layers": -1,
            },
        }
    else:
        vlm_preset_map = {
            "accurate": {
                "max_new_tokens": 900,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop_on_json": True,
            },
            "fast": {
                "max_new_tokens": 400,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop_on_json": True,
            },
        }
    if vlm_preset not in vlm_preset_map:
        print(f"[WARN] VLM_PRESET '{vlm_preset}' non valido. Uso 'accurate'.", flush=True)
        vlm_preset = "accurate"
    base_vlm_kwargs = dict(vlm_preset_map[vlm_preset])
    if use_gguf:
        base_vlm_kwargs["clip_model_path"] = clip_model_path
    print(f"VLM preset: {vlm_preset}", flush=True)
    vlm_validation_retries = max(0, int(os.environ.get("VLM_VALIDATION_RETRIES", "2")))
    vlm_final_distance_threshold = float(os.environ.get("VLM_FINAL_DISTANCE_THRESHOLD", "0.8"))
    # Tolleranza metrica per evitare rigidita' eccessiva vicino alla soglia finale.
    vlm_final_distance_tolerance = float(
        os.environ.get("VLM_FINAL_DISTANCE_TOLERANCE", "0.03")
    )
    final_distance_limit = vlm_final_distance_threshold + vlm_final_distance_tolerance
    # Quando usi GGUF con llama.cpp su GPU, evita conflitti ROCm con YOLO (torch)
    # forzando YOLO su CPU, salvo override esplicito dell'utente.
    if use_gguf and not os.environ.get("YOLO_DEVICE"):
        os.environ["YOLO_DEVICE"] = "cpu"

    torch = None
    if not use_gguf:
        try:
            import torch as _torch
        except Exception:
            torch = None
        else:
            torch = _torch

    # Usiamo un try/finally per garantire che il controller venga sempre chiuso
    # correttamente: anche se la VLM fallisce, c'e' un errore nel loop, o
    # l'esecuzione viene interrotta, il simulatore non resta appeso.
    should_document = True
    skip_auto_documentation = (
        os.environ.get("BENCHMARK_SKIP_AUTO_DOCUMENTATION", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    try:
        print("\nAvvio ciclo VLM...")
        vlm = None
        if run_vlm:
            # Testo grezzo passato da riga di comando (es. "trova la mela").
            user_input = goal_input
            if use_gguf:
                # Carica prima il backend GGUF, cosi' eventuali import torch successivi
                # (traduzione/YOLO) non rompono il caricamento ROCm di llama.cpp.
                prompt_key = os.environ.get("VLMPROMPT", "")
                system_prompt = select_system_prompt(prompt_key)
                vlm = create_vlm(
                    vlm_model,
                    user_prompt="",
                    system_prompt=system_prompt,
                    **base_vlm_kwargs,
                )
                # Preload immediato: inizializza llama.cpp prima di importare torch/YOLO.
                # Se VLM_REQUIRE_GPU=1 e la GPU non e' disponibile, fallisce qui in modo esplicito.
                model_load_start = time.perf_counter()
                vlm.load()
                telemetry["model_load_sec"] = time.perf_counter() - model_load_start
            # Import lazy: evita di caricare torch/ctranslate2 prima di llama_cpp in modalita' GGUF.
            from Architettura.translate import extract_target_phrase, translate_target
            # Estrae il target in italiano (per la VLM) e lo traduce in inglese (per YOLO).
            target_it = extract_target_phrase(user_input)
            translated_target = translate_target(user_input)
            # Lista target per YOLO (inglese).
            targets_norm = [translated_target] if translated_target else []
            # Target principale usato nel loop YOLO (stringa singola).
            current_target = targets_norm[0] if targets_norm else ""
            # Lista target per la VLM (italiano, senza traduzione).
            vlm_targets = [target_it] if target_it else []
            # Riga leggibile che finira' nel prompt, utile per debug/contesto.
            target_line = (
                f"Target Object: {', '.join(vlm_targets)}"
                if vlm_targets
                else "Target Object: (non specificato)"
            )

            base_task = (
                f"{target_line}\n"
                f"TARGET_LIST: {vlm_targets}\n"
                f"CURRENT_TARGET: {vlm_targets[0] if vlm_targets else ''}\n"
            )
            prompt_key = os.environ.get("VLMPROMPT", "")
            system_prompt = select_system_prompt(prompt_key)
            # Crea il client VLM: carica modello + processor e prepara config di generazione.
            if vlm is None:
                vlm = create_vlm(
                    vlm_model,
                    user_prompt=base_task,
                    system_prompt=system_prompt,
                    **base_vlm_kwargs,
                )
                model_load_start = time.perf_counter()
                vlm.load()
                telemetry["model_load_sec"] = time.perf_counter() - model_load_start
            else:
                vlm.user_prompt = base_task
        if vlm is not None:
            if vlm_bench_start_delay_sec > 0:
                print(
                    "[BENCH] Warmup post-load: "
                    f"attendo {vlm_bench_start_delay_sec:.1f}s prima di iniziare il run.",
                    flush=True,
                )
                time.sleep(vlm_bench_start_delay_sec)
            active_run_started_ts = time.time()
            try:
                _write_benchmark_phase(
                    phase_path,
                    {
                        "active_run_started_ts": active_run_started_ts,
                        "active_run_started_at": _utc_iso_from_ts(active_run_started_ts),
                    },
                )
            except Exception as exc:
                print(f"[WARN] Scrittura benchmark phase fallita: {exc}", flush=True)
            # Primo evento "neutro": serve ad avere un frame iniziale su cui
            # costruire il contesto e i primi artefatti senza muovere il robot.
            event = controller.step(action="Pass")
            # Contatore dei giri di scanning (rotazioni da 90 gradi).
            scan_count = 0
            # Flag che indica se lo scanning iniziale e' completato.
            scan_complete = False
            # Stringa mostrata una volta sola per ricordare alla VLM il formato JSON atteso.
            response_format = (
                "*** FORMATO RISPOSTA (JSON) ***\n"
                "{\n"
                "  \"reasoning\": \"Spiegazione del piano (3-5 azioni).\",\n"
                "  \"action_sequence\": [\n"
                "    {\"action\": \"API_Command\", \"parameters\": {\"moveMagnitude\": float, \"degrees\": float}},\n"
                "    {\"action\": \"API_Command\", \"parameters\": {\"moveMagnitude\": float, \"degrees\": float}}\n"
                "  ]\n"
                "}"
            )
            # Flag locale: parte False e diventa True dopo la prima stampa.
            # Serve a mostrare il formato JSON richiesto una sola volta.
            format_printed = False
            last_state = None
            for step_idx in range(max_steps):
                telemetry["steps_total"] = step_idx + 1
                # Aggiorna la camera third-person cosi' i frame esterni restano coerenti.
                update_third_person_camera(controller, event)
                # Salva gli artefatti del frame corrente (RGB/Depth/Semantic + YOLO/Oracle).
                paths = save_artifacts(
                    event,
                    step_idx,
                    target_label=current_target,
                )
                # Estrai id utili: tracking YOLO-World e objectId dell'oracolo (ground truth).
                if paths:
                    track_id = paths.get("yolo_track_id")
                    oracle_id = paths.get("oracle_object_id")
                else:
                    track_id = None
                    oracle_id = None
                # Metadata dell'ultima azione eseguita dal simulatore.
                last_action = event.metadata.get("lastAction") or "None"
                # True/False se l'azione precedente ha avuto successo.
                last_action_success = bool(event.metadata.get("lastActionSuccess", True))
                # Messaggio di errore associato all'ultima azione (se presente).
                last_error = event.metadata.get("errorMessage", "") or ""
                # Distanza stimata dal target (valore minimo tra tutti i target visti).
                target_distance = None
                # objectId dell'oggetto che fornisce quella distanza (debug/trace).
                target_distance_source = None
                try:
                    # Lista oggetti noti nel frame corrente, dai metadata AI2-THOR.
                    objects = event.metadata.get("objects") or []
                    # Se l'utente ha specificato un target, cerchiamo solo quel tipo.
                    if current_target:
                        for obj in objects:
                            # Confronto case-insensitive tra objectType e target normalizzato.
                            if str(obj.get("objectType", "")).lower() == current_target:
                                # Distanza dal robot al singolo oggetto.
                                dist = obj.get("distance")
                                # Se la distanza non e' disponibile, ignoriamo questo oggetto.
                                if dist is None:
                                    continue
                                # Teniamo la distanza minima per scegliere il target piu' vicino.
                                if target_distance is None or dist < target_distance:
                                    # Salviamo la distanza effettiva.
                                    target_distance = float(dist)
                                    # Salviamo anche l'objectId della sorgente.
                                    target_distance_source = obj.get("objectId")
                except Exception:
                    # In caso di errore nei metadata, azzeriamo per evitare dati incoerenti.
                    target_distance = None
                    target_distance_source = None
                telemetry["final_target_distance_m"] = target_distance
                if target_distance is not None:
                    prev_min = telemetry["min_target_distance_m"]
                    dist_value = float(target_distance)
                    if telemetry["initial_target_distance_m"] is None:
                        telemetry["initial_target_distance_m"] = dist_value
                    if prev_min is None or dist_value < prev_min:
                        telemetry["min_target_distance_m"] = dist_value
                    if (
                        telemetry["time_to_target_sec"] is None
                        and dist_value <= final_distance_limit
                    ):
                        telemetry["time_to_target_sec"] = time.time() - run_started_ts
                # Debug visivo: distanza target, tracking YOLO e oracolo.
                print(
                    f"[DEBUG] target_distance={target_distance} source={target_distance_source} "
                    f"track_id={track_id} oracle_id={oracle_id}",
                    flush=True,
                )
                # Se ByteTrack non assegna un id, avvisiamo perche' il tracking sara' meno stabile.
                if track_id is None:
                    print(
                        "[WARN] ByteTrack: track_id non assegnato (uso fallback conf/IoU).",
                        flush=True,
                    )
                # Prompt di step: include stato, task base e telemetria minima.
                step_prompt = (
                    f"{base_task}\n"
                )
                # Se non l'abbiamo ancora fatto, stampiamo una sola volta il
                # formato JSON atteso e il target corrente.
                if not format_printed:
                    print(response_format, flush=True)
                    print(f"TARGET: {current_target}", flush=True)
                    format_printed = True
                # run_vlm_on_step:
                # - costruisce il contesto (telemetria + stato + target),
                # - prepara le immagini (semantic/overlay o RGB),
                # - compone il prompt finale,
                # - chiama la VLM e salva la risposta su file.
                infer_start = time.perf_counter()
                vlm_output = run_vlm_on_step(
                    vlm_client=vlm,
                    event=event,
                    image=event.frame,
                    action_name="VLM",
                    frame_idx=step_idx,
                    paths=paths,
                    target_label=current_target,
                    user_prompt=step_prompt,
                    scan_count=scan_count,
                    last_state=last_state,
                )
                telemetry["vlm_inference_seconds"].append(time.perf_counter() - infer_start)
                result = None
                parsed_payload = None
                retry_prompt = step_prompt
                for attempt in range(vlm_validation_retries + 1):
                    if attempt > 0:
                        telemetry["retry_attempts_total"] += 1
                        infer_start = time.perf_counter()
                        vlm_output = run_vlm_on_step(
                            vlm_client=vlm,
                            event=event,
                            image=event.frame,
                            action_name="VLM",
                            frame_idx=step_idx,
                            paths=paths,
                            target_label=current_target,
                            user_prompt=retry_prompt,
                            scan_count=scan_count,
                            last_state=last_state,
                        )
                        telemetry["vlm_inference_seconds"].append(
                            time.perf_counter() - infer_start
                        )
                    try:
                        parsed_payload = parse_vlm_json(vlm_output)
                    except ValueError as exc:
                        telemetry["json_parse_failures_count"] += 1
                        if attempt < vlm_validation_retries:
                            retry_prompt = _build_validation_retry_prompt(
                                step_prompt,
                                [
                                    "Output JSON non valido.",
                                    f"Dettaglio parser: {exc}",
                                ],
                            )
                            continue
                        print(f"[WARN] JSON VLM ancora non valido: {exc}", flush=True)
                        event = controller.step(action="Pass")
                        result = {
                            "action": "Pass",
                            "success": False,
                            "event": event,
                            "error": "JSON non trovato nell'output VLM.",
                        }
                        break

                    semantic_errors = validate_vlm_semantics(
                        parsed_payload,
                        target_distance=target_distance,
                        final_distance_threshold=vlm_final_distance_threshold,
                        distance_tolerance=vlm_final_distance_tolerance,
                    )
                    if semantic_errors:
                        telemetry["semantic_validation_failures_count"] += 1
                        print(
                            "[WARN] Validazione semantica fallita: "
                            + " | ".join(semantic_errors),
                            flush=True,
                        )
                        if attempt < vlm_validation_retries:
                            retry_prompt = _build_validation_retry_prompt(
                                step_prompt, semantic_errors
                            )
                            continue
                        event = controller.step(action="Pass")
                        result = {
                            "action": "Pass",
                            "success": False,
                            "event": event,
                            "error": "Piano VLM non valido dopo retry di validazione.",
                        }
                        break
                    try:
                        # run_vlm_on_step decide; execute_vlm_payload esegue nel simulatore.
                        result = execute_vlm_payload(controller, parsed_payload)
                    except ValueError as exc:
                        if attempt < vlm_validation_retries:
                            retry_prompt = _build_validation_retry_prompt(
                                step_prompt,
                                [
                                    f"Azione non valida: {exc}",
                                ],
                            )
                            continue
                        event = controller.step(action="Pass")
                        result = {
                            "action": "Pass",
                            "success": False,
                            "event": event,
                            "error": f"Azione non valida dopo retry: {exc}",
                        }
                    break
                if result is None:
                    event = controller.step(action="Pass")
                    result = {
                        "action": "Pass",
                        "success": False,
                        "event": event,
                        "error": "Retry VLM esauriti senza piano eseguibile.",
                    }
                if result.get("state"):
                    last_state = result.get("state")
                # Se la VLM decide di fermarsi, interrompiamo il loop.
                if result["action"] == "Stop":
                    exit_reason = "stop"
                    # L'overlay documentazione usa shift=1 (azione N mostrata sul frame N+1).
                    # Senza questo salvataggio finale, l'ultimo stato/azione non compare nei video.
                    try:
                        final_frame_idx = step_idx + 1
                        final_event = result.get("event") or event
                        update_third_person_camera(controller, final_event)
                        save_artifacts(
                            final_event,
                            final_frame_idx,
                            target_label=current_target,
                        )
                    except Exception as exc:
                        print(
                            f"[WARN] Salvataggio frame finale post-stop fallito: {exc}",
                            flush=True,
                        )
                    print("VLM: Stop ricevuto, termino.")
                    break
                # Aggiorniamo l'evento con quello generato dall'ultima azione.
                event = result["event"]
                # Lista azioni realmente eseguite (puo' essere sequenza).
                actions_executed = result.get("actions_executed") or []
                if not actions_executed:
                    actions_executed = [{"action": result.get("action")}]
                # Aggiorniamo lo stato di scanning in base alle azioni fatte.
                if not scan_complete and scan_count < 4:
                    for step in actions_executed:
                        action_name = step.get("action")
                        if action_name in {"RotateRight", "RotateLeft"}:
                            scan_count += 1
                            if scan_count >= 4:
                                # Dopo 4 rotazioni da 90°, lo scanning iniziale è considerato concluso.
                                scan_complete = True
                        if action_name in {
                            "MoveAhead",
                            "MoveLeft",
                            "MoveRight",
                            "Stop",
                        }:
                            # Se il robot inizia a muoversi (o decide di fermarsi),
                            # interrompiamo lo scanning iniziale: da qui in poi siamo in
                            # fase "active search". Quindi scan_complete diventa True
                            # solo in questi casi, non "sempre".
                            scan_complete = True
                # Se l'azione fallisce, logghiamo l'errore per debug.
                if not result["success"]:
                    telemetry["action_failures_count"] += 1
                    print(f"Azione fallita: {result['error']}")
                # Pausa tra gli step per rendere la simulazione piu' leggibile.
                if delay_sec > 0:
                    time.sleep(delay_sec)
                # Libera memoria GPU, se disponibile, per evitare overflow.
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        print("\nInterruzione richiesta (Ctrl+C).", flush=True)
        if summary_path:
            # In benchmark mode evitiamo prompt interattivi.
            should_document = True
        else:
            try:
                answer = input("[DOC] Vuoi salvare documentazione? [y/N]: ")
            except (KeyboardInterrupt, EOFError):
                answer = ""
            should_document = answer.strip().lower() == "y"
    except Exception:
        exit_reason = "exception"
        raise
    # Questo blocco viene eseguito sempre: serve a rilasciare risorse e
    # fermare il server AI2-THOR in modo pulito.
    finally:
        run_ended_ts = time.time()
        run_ended_iso = _utc_iso_from_ts(run_ended_ts)
        if should_document and not skip_auto_documentation:
            try:
                print("[DOC] Avvio documentazione (overlay + video)...", flush=True)
                run_documentation(sim_name=doc_sim_name)
            except Exception as exc:
                print(f"[DOC] Documentazione fallita: {exc}", flush=True)
        elif should_document and skip_auto_documentation:
            print(
                "[DOC] Auto-documentazione disattivata: gestione demandata al benchmark.",
                flush=True,
            )
        try:
            controller.stop()
        except Exception as exc:
            print(f"[WARN] Chiusura controller fallita: {exc}", flush=True)

        min_target_distance = telemetry["min_target_distance_m"]
        target_reached = (
            min_target_distance is not None and float(min_target_distance) <= final_distance_limit
        )
        success = bool(target_reached) and exit_reason not in {
            "timeout",
            "exception",
            "keyboard_interrupt",
        }
        inference_values = telemetry["vlm_inference_seconds"]
        inference_mean = (
            sum(inference_values) / len(inference_values) if inference_values else None
        )
        summary_payload = {
            "model_alias": vlm_model,
            "resolved_model_id": resolved_vlm_model,
            "mmproj_alias": vlm_mmproj_alias if use_gguf else None,
            "goal": goal_input,
            "run_started_at": run_started_iso,
            "run_ended_at": run_ended_iso,
            "runtime_sec": max(0.0, run_ended_ts - run_started_ts),
            "exit_reason": exit_reason,
            "steps_total": telemetry["steps_total"],
            "initial_target_distance_m": telemetry["initial_target_distance_m"],
            "min_target_distance_m": telemetry["min_target_distance_m"],
            "final_target_distance_m": telemetry["final_target_distance_m"],
            "target_reached": target_reached,
            "success": success,
            "action_failures_count": telemetry["action_failures_count"],
            "json_parse_failures_count": telemetry["json_parse_failures_count"],
            "semantic_validation_failures_count": telemetry["semantic_validation_failures_count"],
            "retry_attempts_total": telemetry["retry_attempts_total"],
            "vlm_inference_mean_sec": inference_mean,
            "vlm_inference_p95_sec": _compute_p95(inference_values),
            "model_load_sec": telemetry["model_load_sec"],
            "documentation_sim_name": doc_sim_name,
            "final_distance_threshold": vlm_final_distance_threshold,
            "final_distance_tolerance": vlm_final_distance_tolerance,
            "final_distance_limit": final_distance_limit,
            "time_to_target_sec": telemetry["time_to_target_sec"],
            "bench_start_delay_sec": vlm_bench_start_delay_sec,
        }
        try:
            _write_benchmark_summary(summary_path, summary_payload)
        except Exception as exc:
            print(f"[WARN] Scrittura benchmark summary fallita: {exc}", flush=True)


if __name__ == "__main__":
    main()
