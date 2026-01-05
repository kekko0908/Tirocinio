# Scopo: integrazione VLM per planning, azioni e bbox.
import json
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .utils import extract_json_anywhere, parse_vlm_bbox


class VLMEngine:
    def __init__(self, model_id: str, action_set: List[str]):
        self.model_id = model_id
        self.action_set = list(action_set)
        self.processor = AutoProcessor.from_pretrained(model_id, token=True, use_fast=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=True,
            torch_dtype=dtype,
            device_map=device_map,
        )

    def _generate(self, inputs) -> str:
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def plan_subgoals(self, goal_text: str):
        system = (
            "You are a planner for an AI2-THOR robot. "
            "Return JSON only with keys: target_type, subgoals. "
            "subgoals is a list of objects {id, type, description}."
        )
        user = (
            "Goal: "
            + goal_text
            + "\nCreate 3-5 subgoals for search and localization. "
            "Use types: explore, search, approach, localize."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, return_tensors="pt")
        text = self._generate(inputs)
        return extract_json_anywhere(text)

    def choose_action(self, image: Image.Image, subgoal_desc: str, history, context_summary: Dict):
        system = (
            "You control a robot in AI2-THOR. "
            "Pick exactly one action from: "
            + ", ".join(self.action_set)
            + ". Return JSON only: "
            "{\"action\": \"...\", \"reason\": \"short\", \"target_confidence\": 0.0-1.0, \"request_yolo\": true|false}."
        )
        hist = ", ".join(history[-6:]) if history else "none"
        ctx = json.dumps(context_summary, ensure_ascii=True)
        user = (
            f"Subgoal: {subgoal_desc}\n"
            f"Recent actions: {hist}\n"
            f"Context: {ctx}\n"
            "Pick the best next action to find the target. "
            "If the target is likely visible, set request_yolo=true."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        text = self._generate(inputs)
        return extract_json_anywhere(text)

    def predict_bbox(self, image: Image.Image, target_label: str) -> Tuple[Optional[List[float]], str]:
        system = (
            "Task: Bounding-box annotation. "
            f"You are given an RGB frame from AI2-THOR and a target object class: {target_label}. "
            "Provide a single bounding box tightly enclosing the target object. "
            "Output must be in pixel coordinates relative to the image: "
            "{x1, y1, x2, y2} with 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height. "
            "If the object is not visible, output NOT_VISIBLE."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Target: {target_label}"}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        text = self._generate(inputs)
        w, h = image.size
        return parse_vlm_bbox(text, w, h), text
