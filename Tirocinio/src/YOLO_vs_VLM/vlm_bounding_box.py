import torch
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class VLMPredictor:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        print(f"[VLM] Caricamento modello {model_id}...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, token=True, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                token=True, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            self.active = True
        except Exception as e:
            print(f"[VLM ERR] Errore caricamento: {e}")
            self.active = False

    def predict(self, image_array, target_name):
        """
        Crea il prompt e interroga il VLM.
        Returns: bbox [xmin, ymin, xmax, ymax]
        """
        if not self.active: 
            return None

        pil_img = Image.fromarray(image_array)
        
        # --- PROMPT DEFINITION ---
        # Qui definiamo la richiesta. Chiediamo le coordinate.
        prompt = f"Find the '{target_name}'. Return bbox [ymin, xmin, ymax, xmax] normalized 0-1000."
        
        messages = [{
            "role": "user", 
            "content": [{"type": "image"}, {"type": "text", "text": prompt}]
        }]

        # Inferenza
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)
        inputs = self.processor(text=inputs, images=pil_img, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=200)
            
        decoded = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Parsing della risposta (Regex per trovare [y1, x1, y2, x2])
        return self._parse_output(decoded)

    def _parse_output(self, text):
        try:
            match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
            if match:
                y1, x1, y2, x2 = map(int, match.groups())
                # Conversione da 0-1000 a coordinate pixel 640x480
                return [(x1/1000)*640, (y1/1000)*480, (x2/1000)*640, (y2/1000)*480]
        except:
            pass
        return None