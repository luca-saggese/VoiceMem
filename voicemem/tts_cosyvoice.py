"""CosyVoice 3 local streaming TTS provider.

The heavy CosyVoice source tree is imported only when the provider is first
used. Synchronous inference runs in one worker thread and yields PCM16 mono
24 kHz chunks to asyncio.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import traceback
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
        default_model = Path(__file__).resolve().parents[1] / "models" / "tts" / "Fun-CosyVoice3-0.5B-2512"
        self.model_path = model or os.environ.get("VOICEMEM_COSYVOICE_MODEL") or str(default_model)
        default_ref = Path(__file__).resolve().parents[1] / "assets" / "italian.wav"
        self.ref_audio = (ref_audio or os.environ.get("VOICEMEM_COSYVOICE_REF_AUDIO")
                  or (str(default_ref) if default_ref.is_file() else None))
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
                    model_path = Path(self.model_path).expanduser()
                    if not model_path.is_dir():
                        raise FileNotFoundError(
                            "Modello CosyVoice3 locale non trovato: "
                            f"{model_path}. Esegui scripts/download_models.sh "
                            "oppure imposta VOICEMEM_COSYVOICE_MODEL su una directory locale."
                        )
                    # Prevent optional CosyVoice frontends from contacting ModelScope.
                    # Italian/English synthesis does not need the Chinese wetext frontend.
                    os.environ.setdefault("COSYVOICE_LOCAL_ONLY", "1")
                    try:
                        os.environ.setdefault("MODELSCOPE_CACHE", str(Path.home() / ".cache" / "modelscope"))
                        os.environ.setdefault("MODELSCOPE_SDK_DEBUG", "0")
                        import sys
                        repo_root = Path(__file__).resolve().parents[1]
                        cosy_root = repo_root / "third_party" / "CosyVoice"
                        matcha_root = cosy_root / "third_party" / "Matcha-TTS"
                        for path in (cosy_root, matcha_root):
                            if path.is_dir() and str(path) not in sys.path:
                                sys.path.insert(0, str(path))
                        logging.getLogger("modelscope").setLevel(logging.ERROR)
                        logging.getLogger("funasr").setLevel(logging.WARNING)
                        logging.getLogger("lightning").setLevel(logging.WARNING)
                        from cosyvoice.cli.cosyvoice import AutoModel
                    except ImportError as exc:
                        raise RuntimeError(
                            "CosyVoice non installato. Esegui scripts/setup_cosyvoice.sh"
                        ) from exc
                    self._model = AutoModel(
                        model_dir=str(model_path),
                        load_trt=self.load_trt,
                        load_vllm=self.load_vllm,
                        fp16=self.fp16,
                    )
        return self._model

    def warmup(self) -> None:
        """Carica il modello prima della prima risposta web."""
        self._load()

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

        def put_from_worker(item):
            """Invia un item senza bloccare il thread CosyVoice dopo cancellation."""
            if loop.is_closed():
                return
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            try:
                future.result(timeout=0.5)
            except (asyncio.TimeoutError, RuntimeError, concurrent.futures.CancelledError):
                future.cancel()
            finally:
                # Se l'event loop si chiude mentre il worker sta terminando,
                # evitare coroutine Queue.put lasciate non awaitate.
                if not future.done():
                    future.cancel()

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
                            put_from_worker(pcm)
            except BaseException as exc:
                error.append(exc)
                if not isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    traceback.print_exc()
            finally:
                put_from_worker(_END)

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
