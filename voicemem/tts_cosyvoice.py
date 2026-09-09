"""CosyVoice 3 local streaming TTS provider.

The heavy CosyVoice source tree is imported only when the provider is first
used. Synchronous inference runs in one worker thread and yields PCM16 mono
24 kHz chunks to asyncio.
"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 24000
_END = object()


def prepare_instruction(instruction: str | None, language: str) -> str:
    language_text = "Italian" if language == "it" else "English"
    body = instruction.strip() if instruction else "Speak naturally and warmly."
    if not body.lower().startswith("speak in "):
        body = f"Speak in {language_text}. {body}"
    return body if body.endswith("<|endofprompt|>") else body + "<|endofprompt|>"


def _to_pcm16(result: Any) -> bytes:
    audio = result.get("tts_speech") if isinstance(result, dict) else result
    if audio is None:
        return b""
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(audio).all():
        raise RuntimeError("CosyVoice returned non-finite audio")
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class CosyVoice3TTS:
    """Lazy, serialized CosyVoice3 ``inference_instruct2`` provider."""

    MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    sample_rate = SAMPLE_RATE
    max_concurrency = 1

    def __init__(self, model: str | None = None, ref_audio: str | None = None,
                 default_language: str = "it", speed: float = 1.0,
                 load_trt: bool = False, load_vllm: bool = False,
                 fp16: bool = False, **_: Any) -> None:
        if default_language not in ("it", "en"):
            raise ValueError("CosyVoice supporta solo le lingue it ed en")
        self.model_path = model or os.environ.get("VOICEMEM_COSYVOICE_MODEL") or self.MODEL_ID
        self.ref_audio = ref_audio or os.environ.get("VOICEMEM_COSYVOICE_REF_AUDIO")
        self.default_language = default_language
        self.speed = speed
        self.load_trt = load_trt
        self.load_vllm = load_vllm
        self.fp16 = fp16
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from cosyvoice.cli.cosyvoice import AutoModel
                    except ImportError as exc:
                        raise RuntimeError(
                            "CosyVoice non installato. Esegui scripts/setup_cosyvoice.sh"
                        ) from exc
                    self._model = AutoModel(
                        model_dir=self.model_path,
                        load_trt=self.load_trt,
                        load_vllm=self.load_vllm,
                        fp16=self.fp16,
                    )
        return self._model

    async def stream(self, text: str, instruction: str | None = None):
        if not self.ref_audio:
            raise RuntimeError("CosyVoice richiede VOICEMEM_COSYVOICE_REF_AUDIO")
        if not Path(self.ref_audio).is_file():
            raise FileNotFoundError(f"Reference voice non trovata: {self.ref_audio}")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        cancelled = threading.Event()
        error: list[BaseException] = []
        prepared = prepare_instruction(instruction, self.default_language)

        def worker():
            try:
                with self._infer_lock:
                    generator = self._load().inference_instruct2(
                        tts_text=text,
                        instruct_text=prepared,
                        prompt_wav=self.ref_audio,
                        stream=True,
                        speed=self.speed,
                    )
                    for result in generator:
                        if cancelled.is_set():
                            break
                        pcm = _to_pcm16(result)
                        if pcm:
                            asyncio.run_coroutine_threadsafe(queue.put(pcm), loop).result()
            except BaseException as exc:
                error.append(exc)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_END), loop).result()

        thread = threading.Thread(target=worker, name="cosyvoice3-tts", daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is _END:
                    break
                yield item
        finally:
            cancelled.set()
        if error:
            raise error[0]
