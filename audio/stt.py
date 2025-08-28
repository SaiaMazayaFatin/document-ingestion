from transformers import pipeline as hf_pipeline
from typing import List, Dict, Any
from settings import settings
import os
try:
    import torch
except Exception:  # torch may not be installed with GPU support
    torch = None  # type: ignore

_ASR_PIPELINE = None  # global cache to avoid reloading model for every batch

def _select_device():
    # Allow override via env STT_DEVICE (e.g., "cuda:0" or "cpu")
    override = os.getenv("STT_DEVICE")
    if override:
        return override
    if torch is not None and torch.cuda.is_available():
        return 0  # HF pipeline accepts int GPU index
    return "cpu"


def _get_asr():
    global _ASR_PIPELINE
    if _ASR_PIPELINE is None:
        device = _select_device()
        print(f"[stt] loading model {settings.STT_MODEL} on device={device}")
        _ASR_PIPELINE = hf_pipeline(
            "automatic-speech-recognition",
            model=settings.STT_MODEL,
            chunk_length_s=None,
            return_timestamps=True,
            device=device,
        )
    return _ASR_PIPELINE

def stt_batch_30s(chunk_paths: List[str], language: str | None = None) -> List[Dict[str, Any]]:
    """Transcribe list of 30s audio chunk paths.

    Caches underlying HF pipeline for performance.
    """
    asr = _get_asr()
    outs = []
    for p in chunk_paths:
        if language:
            o = asr(p, generate_kwargs={"language": language})
        else:
            o = asr(p)
        outs.append({"file": p, "text": o.get("text", ""), "chunks": o.get("chunks")})
    return outs

def merge_text(segments: List[Dict[str, Any]]) -> str:
    return "\n".join([s.get("text", "").strip() for s in segments if s.get("text")])
