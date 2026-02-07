"""Vision-Language Model (VLM) wrapper con switch facile del modello."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from Architettura.paths import PROJECT_ROOT
from Architettura.vlm.vlm_prompt import SYSTEM_PROMPT, build_prompt
try:
    from PIL import Image
except Exception:  # pragma: no cover - fallback se PIL manca
    Image = None

_TORCH_MODULE = None
_TORCH_IMPORT_ERROR = None
_TRANSFORMERS_IMPORT_ERROR = None
_TRANSFORMERS_IMPORTED = False
AutoModelForImageTextToText = None
AutoProcessor = None
StoppingCriteria = None
StoppingCriteriaList = None


MODEL_ALIASES: Dict[str, str] = {
    "qwen-vl2.5": "modelli_gguf/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
    "qwen-vl-3": "modelli_gguf/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q8_0.gguf",
    "gemma-3": "google/gemma-3-4b-it",

}

MMPROJ_ALIASES: Dict[str, str] = {
    # Alias mmproj consigliato per Qwen2.5-VL GGUF (naming comune su Unsloth).
    "qwen-vl-mmproj-2.5": "modelli_gguf/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-F16.gguf",
    "qwen-vl-mmproj-3": "modelli_gguf/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    
}


@dataclass
class VLMConfig:
    """Configurazione per caricare un VLM."""

    model_id: str  # ID del modello (HF repo o percorso locale/alias).
    system_prompt: str = SYSTEM_PROMPT  # Prompt di sistema (regole/global goal).
    user_prompt: str = ""  # Prompt utente base (es. task iniziale).
    device: str = "auto"  # "auto" sceglie cuda se disponibile, altrimenti cpu.
    dtype: str = "auto"  # Precisione (auto/bf16/fp16/fp32):
    # - fp32: piu' preciso ma piu' lento e usa piu' memoria,
    # - fp16: piu' veloce e leggero ma meno preciso,
    # - bf16: simile a fp16 ma con range numerico piu' ampio (spesso piu' stabile),
    # - auto: lascia scegliere al framework/modello.
    max_new_tokens: int = 700  # Numero massimo di token generati per risposta.
    temperature: float = 0.0  # Temperatura sampling (0 = deterministico).
    top_p: float = 0.9  # Nucleus sampling: limita la distribuzione dei token.
    trust_remote_code: bool = True  # Consente codice custom dei modelli HF.
    stop_on_json: bool = True  # Stop anticipato quando il JSON appare chiuso.


def _get_torch():
    global _TORCH_MODULE, _TORCH_IMPORT_ERROR
    if _TORCH_MODULE is not None:
        return _TORCH_MODULE
    if _TORCH_IMPORT_ERROR is not None:
        return None
    try:
        import torch as torch_module
    except Exception as exc:  # pragma: no cover - fallback se torch manca
        _TORCH_IMPORT_ERROR = exc
        return None
    _TORCH_MODULE = torch_module
    return _TORCH_MODULE


def _ensure_transformers_imported() -> None:
    global _TRANSFORMERS_IMPORTED
    global _TRANSFORMERS_IMPORT_ERROR
    global AutoModelForImageTextToText, AutoProcessor
    global StoppingCriteria, StoppingCriteriaList
    if _TRANSFORMERS_IMPORTED:
        return
    try:
        from transformers import AutoModelForImageTextToText as _AutoModelForImageTextToText
        from transformers import AutoProcessor as _AutoProcessor
        from transformers import StoppingCriteria as _StoppingCriteria
        from transformers import StoppingCriteriaList as _StoppingCriteriaList
    except Exception as exc:  # pragma: no cover - fallback se transformers manca
        _TRANSFORMERS_IMPORT_ERROR = exc
    else:
        AutoModelForImageTextToText = _AutoModelForImageTextToText
        AutoProcessor = _AutoProcessor
        StoppingCriteria = _StoppingCriteria
        StoppingCriteriaList = _StoppingCriteriaList
    _TRANSFORMERS_IMPORTED = True


def _build_stopping_criteria(stop_token_ids: list[int]):
    if not stop_token_ids or StoppingCriteria is None or StoppingCriteriaList is None:
        return None

    class _StopOnToken(StoppingCriteria):
        # Ferma la generazione quando appare uno dei token specificati.
        def __init__(self, token_ids: list[int]) -> None:
            self.stop_token_ids = set(token_ids)

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if input_ids is None or input_ids.size(0) == 0:
                return False
            last_id = int(input_ids[0, -1])
            return last_id in self.stop_token_ids

    return StoppingCriteriaList([_StopOnToken(stop_token_ids)])


def resolve_model_id(name_or_id: str) -> str:
    """Ritorna un model_id completo anche se si passa un alias breve."""
    # Se l'utente usa un alias (es. "gemma-3"), lo risolviamo nel vero ID HF.
    value = MODEL_ALIASES.get(name_or_id, name_or_id)
    return _resolve_local_artifact_path(value)


def resolve_mmproj_path(name_or_path: Optional[str]) -> Optional[str]:
    """Ritorna il path mmproj completo anche se si passa un alias breve."""
    if name_or_path is None:
        return None
    value = str(name_or_path).strip()
    if not value:
        return None
    resolved = MMPROJ_ALIASES.get(value, value)
    return _resolve_local_artifact_path(resolved)


def _resolve_local_artifact_path(value: str) -> str:
    """Risolvi path locali (GGUF/mmproj) in modo indipendente dal cwd."""
    text = str(value).strip()
    if not text:
        return text
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.as_posix()
    suffix = candidate.suffix.lower()
    # Se non e' un file locale noto, trattiamolo come model id HF.
    if suffix not in {".gguf", ".bin"}:
        return text
    search_roots = [
        PROJECT_ROOT,               # .../Richiesta 1/src
        PROJECT_ROOT.parent,        # .../Richiesta 1
        PROJECT_ROOT.parent.parent, # .../Tirocinio-Root
        Path.cwd(),                 # cwd corrente
    ]
    for root in search_roots:
        path = (root / candidate).resolve()
        if path.exists():
            return path.as_posix()
    # Fallback deterministico: usa il workspace root per evitare dipendenza da cwd.
    return (PROJECT_ROOT.parent.parent / candidate).resolve().as_posix()


def _ensure_dependencies() -> None:
    # Verifica che torch e transformers siano disponibili prima di usare il client.
    if _get_torch() is None:
        raise RuntimeError(
            "torch non disponibile. Installa torch per usare la VLM."
        ) from _TORCH_IMPORT_ERROR
    _ensure_transformers_imported()
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "transformers non disponibile. Installa transformers per usare la VLM."
        ) from _TRANSFORMERS_IMPORT_ERROR


def _resolve_dtype(dtype: str) -> Optional["torch.dtype"]:
    # Traduce una stringa (bf16/fp16/fp32/auto) nel dtype torch corrispondente.
    torch_module = _get_torch()
    if torch_module is None:
        return None
    normalized = dtype.lower()
    if normalized == "auto":
        return None
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Tipo dtype non supportato: {dtype}")


def _pick_device(device: str) -> str:
    # Se device="auto", usa GPU se disponibile; altrimenti CPU.
    if device != "auto":
        return device
    torch_module = _get_torch()
    if torch_module is not None and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_pil(image: Any) -> "Image.Image":
    # Converte input generico in PIL.Image (Pillow) per il processor.
    # Il "processor" e' l'oggetto di HuggingFace che trasforma testo+immagini
    # in tensori pronti per il modello (tokenizza il testo e pre-processa le immagini).
    if Image is None:
        raise RuntimeError("Serve PIL per convertire le immagini in input.")
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image)


def _to_pil_list(images: Any) -> list["Image.Image"]:
    # Normalizza input singolo o lista in una lista di PIL.Image.
    if isinstance(images, (list, tuple)):
        return [_to_pil(img) for img in images]
    return [_to_pil(images)]


class VLMClient:
    """Wrapper minimalista per VLM con generazione da immagine + prompt."""

    def __init__(self, config: VLMConfig):
        # Salva config e prepara componenti; il modello verra' caricato in modo lazy.
        _ensure_dependencies()
        self.config = config
        self.model_id = resolve_model_id(config.model_id)
        self.device = _pick_device(config.device)
        self.system_prompt = config.system_prompt
        self.user_prompt = config.user_prompt
        self._processor = None
        self._model = None

    def load(self) -> None:
        """Carica processor e modello in memoria (lazy)."""
        # Carica una sola volta processor + modello (evita reload ad ogni step).
        if self._model is not None:
            return
        dtype = _resolve_dtype(self.config.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=dtype if dtype is not None else "auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model.to(self.device)
        self._model.eval()

    def _build_inputs(self, image: Any, prompt: str) -> Dict[str, "torch.Tensor"]:
        # Prepara tensori di input per il modello (testo + immagini).
        if self._processor is None:
            raise RuntimeError("Processor non inizializzato.")
        pil_images = _to_pil_list(image)
        content = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = []
        if self.system_prompt and str(self.system_prompt).strip():
            messages.append(
                {
                    "role": "system",
                    "content": str(self.system_prompt).strip(),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        if hasattr(self._processor, "apply_chat_template"):
            # Usa il template chat del modello se disponibile (piu' corretto).
            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            return self._processor(text=text, images=pil_images, return_tensors="pt")
        text = prompt
        if self.system_prompt and str(self.system_prompt).strip():
            text = f"{str(self.system_prompt).strip()}\n\n{prompt}"
        return self._processor(text=text, images=pil_images, return_tensors="pt")

    def compose_prompt(
        self,
        user_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Costruisce il prompt completo: ruolo + richiesta + contesto."""
        # Unisce system prompt, richiesta utente e contesto in un'unica stringa.
        if user_prompt is None:
            user_prompt = self.user_prompt
        return build_prompt(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            context=context,
        )

    def generate(self, image: Any, prompt: Optional[str] = None) -> str:
        """Genera testo a partire da un'immagine e un prompt."""
        # Esegue la generazione vera e propria con stopping criteria opzionale.
        # Passi principali:
        # 1) carica modello/processor se non gia' presenti,
        # 2) costruisce il prompt finale (se non passato),
        # 3) prepara i tensori di input (testo + immagini),
        # 4) decide se usare sampling (temperature > 0),
        # 5) applica uno stop anticipato se serve (token "}" / eos),
        # 6) genera e decodifica il testo finale.
        if self._model is None or self._processor is None:
            self.load()
        if prompt is None:
            prompt = self.compose_prompt()
        inputs = self._build_inputs(image, prompt)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        torch_module = _get_torch()
        if torch_module is None:
            raise RuntimeError("torch non disponibile in fase di generazione.")
        do_sample = self.config.temperature > 0
        stopping_criteria = None
        eos_token_id = getattr(self._model.config, "eos_token_id", None)
        if self.config.stop_on_json:
            # Stop anticipato su token "}" o eos: evita testo extra fuori dal JSON.
            tokenizer = getattr(self._processor, "tokenizer", None)
            stop_token_ids: list[int] = []
            if eos_token_id is not None:
                if isinstance(eos_token_id, list):
                    stop_token_ids.extend(int(t) for t in eos_token_id)
                else:
                    stop_token_ids.append(int(eos_token_id))
            if tokenizer is not None:
                brace_ids = tokenizer.encode("}", add_special_tokens=False)
                if brace_ids:
                    stop_token_ids.append(int(brace_ids[-1]))
            stopping_criteria = _build_stopping_criteria(stop_token_ids)
        with torch_module.no_grad():
            # Generazione in no_grad per risparmiare memoria e velocizzare.
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            output_ids = output_ids[:, input_ids.shape[-1]:]
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()


def _is_gguf(model_name_or_id: str) -> bool:
    # Rileva se il modello e' un file GGUF (llama.cpp).
    name = str(model_name_or_id).strip()
    if name.lower().endswith(".gguf"):
        return True
    try:
        return Path(name).suffix.lower() == ".gguf"
    except Exception:
        return False


def create_vlm(model_name_or_id: str, **kwargs: Any):
    """Factory rapido per creare un client VLM (HF o GGUF)."""
    # Se e' GGUF, usa il backend llama.cpp; altrimenti usa HuggingFace.
    model_name_or_id = str(model_name_or_id).strip()
    resolved_model_id = resolve_model_id(model_name_or_id)
    if _is_gguf(resolved_model_id):
        clip_model_path = resolve_mmproj_path(kwargs.get("clip_model_path"))
        try:
            from Architettura.vlm.vlm_execute_llama import create_llama_vlm
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Backend GGUF non disponibile. Installa llama-cpp-python."
            ) from exc
        return create_llama_vlm(
            model_path=resolved_model_id,
            system_prompt=kwargs.get("system_prompt", SYSTEM_PROMPT),
            user_prompt=kwargs.get("user_prompt", ""),
            n_ctx=kwargs.get("n_ctx", 8192),
            n_gpu_layers=kwargs.get("n_gpu_layers", -1),
            n_batch=kwargs.get("n_batch", 512),
            max_tokens=kwargs.get("max_new_tokens", 1200),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 0.9),
            verbose=kwargs.get("verbose", True),
            clip_model_path=clip_model_path,
        )
    config = VLMConfig(model_id=resolved_model_id, **kwargs)
    return VLMClient(config)
