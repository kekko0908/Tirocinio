"""Backend GGUF (llama.cpp) per VLM multimodale."""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from Architettura.vlm.vlm_prompt import SYSTEM_PROMPT, build_prompt

try:
    from PIL import Image
except Exception:  # pragma: no cover - fallback se PIL manca
    Image = None

try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
except Exception as exc:  # pragma: no cover - fallback se llama-cpp-python manca
    llama_cpp = None
    Llama = None
    Qwen25VLChatHandler = None
    _LLAMA_IMPORT_ERROR = exc
else:
    _LLAMA_IMPORT_ERROR = None


@dataclass
class LlamaVLMConfig:
    model_path: str
    system_prompt: str = SYSTEM_PROMPT
    user_prompt: str = ""
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    n_batch: int = 512
    max_tokens: int = 1200
    temperature: float = 0.0
    top_p: float = 0.9
    verbose: bool = True
    clip_model_path: Optional[str] = None


def _ensure_dependencies() -> None:
    if _LLAMA_IMPORT_ERROR is not None:
        details = f" Dettaglio: {_LLAMA_IMPORT_ERROR}"
        raise RuntimeError(
            "llama-cpp-python non disponibile o non caricabile." + details
        ) from _LLAMA_IMPORT_ERROR
    if Image is None:
        raise RuntimeError("Serve Pillow (PIL) per usare il backend GGUF VLM.")


def _to_pil(image: Any) -> "Image.Image":
    if Image is None:
        raise RuntimeError("Serve PIL per convertire le immagini in input.")
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image)


def _to_pil_list(images: Any) -> list["Image.Image"]:
    if isinstance(images, (list, tuple)):
        return [_to_pil(img) for img in images]
    return [_to_pil(images)]


def _find_mmproj_for_model(model_path: str) -> Optional[str]:
    """Prova a trovare automaticamente un file mmproj vicino al modello."""
    model = Path(model_path).expanduser().resolve()
    if not model.exists():
        return None
    candidates = []
    patterns = (
        "*mmproj*.gguf",
        "*mmproj*.bin",
        "*mm-proj*.gguf",
        "*mm-proj*.bin",
        "*vision*.gguf",
    )
    for pattern in patterns:
        candidates.extend(model.parent.glob(pattern))
    # Evita di selezionare per errore il file modello principale.
    filtered = [p for p in candidates if p.resolve() != model]
    if not filtered:
        return None
    # Priorita': nome con qwen + mmproj, altrimenti primo ordinato.
    filtered.sort(key=lambda p: p.name.lower())
    for path in filtered:
        name = path.name.lower()
        if "qwen" in name and "mm" in name:
            return path.as_posix()
    return filtered[0].as_posix()


def _pil_to_data_url(image: "Image.Image") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


class LlamaVLMClient:
    def __init__(self, config: LlamaVLMConfig):
        _ensure_dependencies()
        self.config = config
        self.model_path = str(Path(config.model_path).expanduser())
        self.system_prompt = config.system_prompt
        self.user_prompt = config.user_prompt
        self._llm = None
        self._effective_n_gpu_layers = config.n_gpu_layers

    def load(self) -> None:
        if self._llm is not None:
            return

        model_path = Path(self.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Modello GGUF non trovato: {model_path}")
        if model_path.suffix.lower() != ".gguf":
            raise ValueError(f"Il backend GGUF richiede un file .gguf: {model_path}")

        clip_model_path = self.config.clip_model_path
        if clip_model_path:
            clip_candidate = Path(clip_model_path).expanduser().resolve()
            if not clip_candidate.exists():
                raise FileNotFoundError(
                    f"File mmproj/vision non trovato: {clip_candidate}"
                )
            clip_model_path = clip_candidate.as_posix()
        else:
            clip_model_path = _find_mmproj_for_model(model_path.as_posix())

        if not clip_model_path:
            raise RuntimeError(
                "Manca il file mmproj per Qwen2.5-VL GGUF. "
                "Passa clip_model_path oppure metti il file mmproj accanto al modello."
            )

        if self.config.n_gpu_layers != 0 and llama_cpp is not None:
            gpu_ok = llama_cpp.llama_supports_gpu_offload()
            if not gpu_ok:
                # Fallback robusto: non bloccare il run se GPU ROCm non disponibile.
                # Per ripristinare errore hard, imposta VLM_REQUIRE_GPU=1.
                if os.environ.get("VLM_REQUIRE_GPU", "").strip() in {"1", "true", "TRUE"}:
                    raise RuntimeError(
                        "llama-cpp-python installato senza supporto GPU offload. "
                        "Su AMD devi reinstallare con HIPBLAS (ROCm)."
                    )
                print(
                    "[WARN] GPU offload non disponibile in llama.cpp; passo a CPU (n_gpu_layers=0).",
                    flush=True,
                )
                self._effective_n_gpu_layers = 0
            else:
                self._effective_n_gpu_layers = self.config.n_gpu_layers
        else:
            self._effective_n_gpu_layers = self.config.n_gpu_layers

        chat_handler = Qwen25VLChatHandler(
            clip_model_path=clip_model_path,
            verbose=self.config.verbose,
        )

        self._llm = Llama(
            model_path=model_path.as_posix(),
            chat_handler=chat_handler,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_gpu_layers=self._effective_n_gpu_layers,
            verbose=self.config.verbose,
        )

    def compose_prompt(
        self,
        user_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if user_prompt is None:
            user_prompt = self.user_prompt
        return build_prompt(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            context=context,
        )

    def generate(self, image: Any, prompt: Optional[str] = None) -> str:
        if self._llm is None:
            self.load()
        if prompt is None:
            prompt = self.compose_prompt()

        pil_images = _to_pil_list(image)
        messages = []
        if self.system_prompt and str(self.system_prompt).strip():
            messages.append(
                {
                    "role": "system",
                    "content": str(self.system_prompt).strip(),
                }
            )
        content = [
            {"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}}
            for img in pil_images
        ]
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        return str(text).strip()


def create_llama_vlm(
    model_path: str,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt: str = "",
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    n_batch: int = 512,
    max_tokens: int = 1200,
    temperature: float = 0.0,
    top_p: float = 0.9,
    verbose: bool = True,
    clip_model_path: Optional[str] = None,
) -> LlamaVLMClient:
    config = LlamaVLMConfig(
        model_path=model_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        verbose=verbose,
        clip_model_path=clip_model_path,
    )
    return LlamaVLMClient(config)
