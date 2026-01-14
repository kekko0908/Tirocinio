# Scopo: integrazione VLM per planning, azioni e bbox.
from typing import Dict, List, Optional, Tuple

import time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from .utils import extract_json_anywhere, parse_vlm_bbox
from .vlm_prompts import (
    build_assess_approach_messages,
    build_choose_action_messages,
    build_plan_navigation_messages,
    build_plan_navigation_subgoals_messages,
    build_plan_subgoals_messages,
    build_predict_bbox_messages,
    build_probe_scene_messages,
)


class VLMEngine:
    def __init__(self, model_id: str, action_set: List[str]):
        """
        Inizializza il motore VLM e il processor.
        Carica il modello e imposta action_set.
        Gestisce fallback tra ImageTextToText e CausalLM.
        """
        self.model_id = model_id
        self.action_set = list(action_set)
        self.processor = AutoProcessor.from_pretrained(
            model_id, token=True, use_fast=True, trust_remote_code=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                token=True,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=True,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

    def _generate(self, inputs, max_new_tokens: int = 256) -> str:
        """
        Esegue una generazione testuale dal modello.
        Sposta input su device se CUDA disponibile.
        Ritorna il testo decodificato.
        """
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        return self.processor.decode(out[0], skip_special_tokens=True)

    def plan_subgoals(self, goal_text: str):
        """
        Richiede alla VLM un piano di subgoal (legacy).
        Costruisce prompt e parse del JSON.
        Ritorna il dict dei subgoal o None.
        """
        # DEPRECATED: non usata dalla FSM, tenuta solo per compatibilita/legacy.
        system, user = build_plan_subgoals_messages(goal_text)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, return_tensors="pt")
        text = self._generate(inputs)
        return extract_json_anywhere(text)

    def choose_action(self, image: Image.Image, subgoal_desc: str, history, context_summary: Dict):
        """
        Chiede alla VLM la prossima azione data immagine.
        Include storico e context_summary nel prompt.
        Ritorna dict JSON con azione e metadati.
        """
        system, user = build_choose_action_messages(self.action_set, subgoal_desc, history, context_summary)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        text = self._generate(inputs)
        return extract_json_anywhere(text)

    def plan_navigation(self, image: Image.Image, target_label: str, context_summary: Dict):
        """
        Chiede un piano di navigazione a breve termine.
        Usa immagine e contesto per generare nav_plan.
        Ritorna dict JSON con piano e confidence.
        """
        system, user = build_plan_navigation_messages(self.action_set, target_label, context_summary)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        text = self._generate(inputs)
        return extract_json_anywhere(text)

    def plan_navigation_subgoals(self, image: Image.Image, target_label: str, context_summary: Dict):
        """
        Chiede subgoal di navigazione con piano per ciascuno.
        Usa max_new_tokens elevato e gestisce parse.
        Ritorna dict con nav_subgoals o _parse_error.
        """
        system, user = build_plan_navigation_subgoals_messages(self.action_set, target_label, context_summary)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        t0 = time.time()
        text = self._generate(inputs, max_new_tokens=2048)
        dt = time.time() - t0
        print(f"[VLM] nav_subgoals generation took {dt:.2f}s")
        data = extract_json_anywhere(text)
        if isinstance(data, dict):
            return data
        return {"_raw_text": text, "_parse_error": True}

    def assess_approach(self, image: Image.Image, target_label: str, context_summary: Dict):
        """
        Valuta se l'approach diretto e possibile.
        Invia immagine e contesto alla VLM.
        Ritorna dict JSON con esito e confidenza.
        """
        system, user = build_assess_approach_messages(target_label, context_summary)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        text = self._generate(inputs)
        return extract_json_anywhere(text)

    def predict_bbox(self, image: Image.Image, target_label: str) -> Tuple[Optional[List[float]], str]:
        """
        Richiede alla VLM la bounding box del target.
        Parsa output testo e clampa ai limiti immagine.
        Ritorna (bbox, raw_text).
        """
        system, user = build_predict_bbox_messages(target_label)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        text = self._generate(inputs)
        w, h = image.size
        return parse_vlm_bbox(text, w, h), text

    def probe_scene(self, image: Image.Image, target_label: str):
        """
        Esegue un probe visivo della scena per il target.
        Parsa JSON di visibilita e hint di posizione.
        Ritorna (data, raw_text) con fallback.
        """
        system, user = build_probe_scene_messages(target_label)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        text = self._generate(inputs)
        data = extract_json_anywhere(text)
        if not isinstance(data, dict):
            data = {
                "target_visible": False,
                "target_visibility": "uncertain",
                "target_location_hint": "unknown",
                "related_objects": [],
                "confidence": 0,
            }
        return data, text
