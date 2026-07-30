"""Optional local ASR adapters.

Dependencies are imported only when an adapter is selected, so the HTTP service
can explain a missing runtime instead of failing during module import.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

DEFAULT_QWEN_MODEL = str(Path(__file__).resolve().parent / "models" / "Qwen3-ASR-0.6B")
DEFAULT_BREEZE_MODEL = str(Path(__file__).resolve().parent / "models" / "Breeze-ASR-25")


class Transcriber(Protocol):
    name: str

    def transcribe(self, wav_path: Path) -> str: ...


class QwenMlxTranscriber:
    name = "qwen3-asr-0.6b-mlx"

    def __init__(self, model: str = "Qwen/Qwen3-ASR-0.6B") -> None:
        try:
            from mlx_qwen3_asr import Session
        except ImportError as exc:
            raise RuntimeError(
                "Qwen MLX runtime is not installed. Run: pip install -r requirements-qwen.txt"
            ) from exc
        self.name = model
        self._session = Session(model=model)

    def transcribe(self, wav_path: Path) -> str:
        result = self._session.transcribe(str(wav_path))
        return str(result.text).strip()


class BreezeTwisterTranscriber:
    """Offline benchmark adapter for MediaTek Breeze-ASR-25 / Twister.

    This runner is intentionally not exposed by the live service yet. It uses
    the model card's Transformers path and should be benchmarked on the target
    Mac before it is promoted to an interactive backend.
    """

    name = "MediaTek-Research/Breeze-ASR-25"

    def __init__(self, model: str = DEFAULT_BREEZE_MODEL) -> None:
        try:
            import torch
            from transformers import AutomaticSpeechRecognitionPipeline, WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Breeze benchmark dependencies are not installed. Run: pip install -r requirements-breeze.txt"
            ) from exc

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        self.name = model
        processor = WhisperProcessor.from_pretrained(model)
        asr_model = WhisperForConditionalGeneration.from_pretrained(model, torch_dtype=dtype).to(device).eval()
        self._pipeline = AutomaticSpeechRecognitionPipeline(
            model=asr_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=0,
            device=torch.device(device),
        )

    def transcribe(self, wav_path: Path) -> str:
        output = self._pipeline(str(wav_path))
        return str(output["text"]).strip()


def create_transcriber(kind: str, model: str | None = None) -> Transcriber:
    if kind == "qwen-mlx":
        return QwenMlxTranscriber(model or os.environ.get("LOCAL_STT_MODEL", DEFAULT_QWEN_MODEL))
    if kind == "breeze":
        return BreezeTwisterTranscriber(model or os.environ.get("BREEZE_ASR_MODEL", DEFAULT_BREEZE_MODEL))
    raise ValueError(f"Unknown backend {kind!r}; expected qwen-mlx or breeze")
