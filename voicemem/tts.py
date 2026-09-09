"""Testo → voce. Il layer di risposta produce solo testo (vedi ``voicemem/reply.py``), qui avviene la conversione vocale, è un layer opzionale.

Due backend, entrambi producono **24kHz PCM16**: default API OpenAI, ``TTS_BACKEND=local`` (o
``reply.tts.provider == "local"``) usa piper offline.

``speak_stream()`` è "sintetizza mentre genera": quando viene completata una frase viene inviata alla sintesi, non si aspetta che tutto il testo sia generato —
se si sintetizza dopo aver generato tutto, le parole sono già finite ma l'audio non è ancora partito (nei test la prima frame del TTS richiede ~1.2s).
"""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache

import numpy as np

from voicemem.utils.audio.stream_io import resample

TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_BACKEND = os.environ.get("TTS_BACKEND", "openai")   # openai(api) | local(modello piccolo offline)
SAMPLE_RATE = 24000

# Come tagliare i segmenti per il TTS determina quanto tempo ci vuole per la prima voce. Nei test la latenza della prima frame di gpt-4o-mini-tts cresce con la lunghezza del testo:
# 8 caratteri 615ms / 25 caratteri 902ms / 100 caratteri 1318ms — quindi **il primo segmento deve essere il più corto possibile** (prima voce in uscita),
# i segmenti successivi possono essere più lunghi (meno chiamate, tono coerente).
_SENT_END  = "。！？!?…\n"          # Fine frase: punto normale di taglio
_SOFT_END  = "，,、；;：: "          # Pausa interna: usato solo per il primo segmento, per far uscire la prima voce prima
_FIRST_MIN = 6                      # Il primo segmento accumula almeno questi caratteri, invia a qualsiasi pausa incontri
_FIRST_MAX = 20                     # Se non ci sono pause, il primo segmento non può aspettare oltre
_SENT_MIN  = 12                     # Lunghezza minima dei segmenti successivi
_SENT_MAX  = 60                     # Limite massimo dei segmenti successivi: anche se il LLM non respira deve tagliare

_client = None


def _openai():
    """Il client viene creato solo al primo utilizzo, ``import voicemem.tts`` non richiede quindi una key."""
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI()
    return _client


def _tts_cfg(reply):
    seg = (reply or {}).get("tts") or {}
    return seg.get("provider"), (seg.get("config") or {})


def cut_point(buf: str, first: bool) -> bool:
    """Questo segmento è sufficiente per inviare alla sintesi."""
    s = buf.strip()
    if not s:
        return False
    if first:                                  # Prima voce: anche la virgola conta, se proprio non ci sono altre opzioni taglia per lunghezza
        return (len(s) >= _FIRST_MIN and s[-1] in _SENT_END + _SOFT_END) or len(s) >= _FIRST_MAX
    return (len(s) >= _SENT_MIN and s[-1] in _SENT_END) or len(s) >= _SENT_MAX


async def tts_stream(text, reply=None):
    """TTS intercambiabile: default usa API OpenAI; reply.tts.provider==local (o TTS_BACKEND=local)
    usa un piccolo modello locale offline. Entrambi producono flusso 24kHz PCM16, il chiamante li tratta allo stesso modo.

    **Taglia ai confini dei campioni**: il flusso http è tagliato per pacchetti di rete, nei test su 69 blocchi 62 hanno byte dispari, mentre
    un campione PCM16 occupa 2 byte — il consumatore ``Int16Array``/``np.frombuffer`` va in errore con lunghezza dispari
    e quel blocco audio intero viene perso. Qui si lascia mezzo campione che attraversa i blocchi al blocco successivo, garantendo che ogni blocco prodotto
    contenga campioni completi.
    """
    provider, cfg = _tts_cfg(reply)
    backend_name = provider or TTS_BACKEND          # Allineato alla semantica esistente di TTS_BACKEND
    backend = {"local": _local_tts_stream, "voxcpm": _voxcpm_tts_stream}.get(
        backend_name, _openai_tts_stream)
    tail = b""
    async for chunk in backend(text, cfg.get("model")):
        buf = tail + chunk
        cut = len(buf) & ~1                         # Arrotonda per difetto al pari
        tail = buf[cut:]
        if cut:
            yield buf[:cut]
    if tail:
        yield tail + b"\x00"                        # Completa l'ultimo mezzo campione


async def _openai_tts_stream(text, model=None):
    """API online: OpenAI TTS (gpt-4o-mini-tts), response_format=pcm è 24k PCM16."""
    async with _openai().audio.speech.with_streaming_response.create(
            model=model or TTS_MODEL, voice="alloy", input=text, response_format="pcm") as resp:
        async for chunk in resp.iter_bytes():
            yield chunk


@lru_cache(maxsize=1)
def _piper_voice():
    """Modello piccolo offline: default piper (onnx puro offline, cinese e inglese). Installa: pip install piper-tts;
    VOICEMEM_TTS_MODEL punta al .onnx della voce. Per cambiare con kokoro / edge-tts ecc., modifica solo questa funzione
    e il campionamento di _local_tts_stream qui sotto. L'api di piper cambia con le versioni, vedi la sua documentazione."""
    from piper import PiperVoice
    return PiperVoice.load(os.environ["VOICEMEM_TTS_MODEL"])


async def _local_tts_stream(text, model=None):
    """TTS locale offline: sintesi → risampola a 24k → yield a blocchi, interfaccia identica alla versione online.
    (La voce offline è specificata da VOICEMEM_TTS_MODEL; il parametro model serve solo per allineare la firma con la versione online.)"""
    v = _piper_voice()
    sr = getattr(getattr(v, "config", None), "sample_rate", 22050)
    for raw in v.synthesize_stream_raw(text):          # Generatore sincrono, int16 bytes @ sr
        f = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        out = resample(f, src=sr, dst=SAMPLE_RATE)     # Uniforma a 24k
        yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


@lru_cache(maxsize=1)
def _voxcpm_model():
    """Modello grande offline: VoxCPM2 (2B, output 48k, cinese+inglese+multilingue). Installa: pip install voxcpm.
    VOICEMEM_TTS_MODEL può puntare a una directory locale, default usa openbmb/VoxCPM2 su HF (usa cache locale)."""
    from voxcpm import VoxCPM
    return VoxCPM.from_pretrained(
        os.environ.get("VOICEMEM_TTS_MODEL") or "openbmb/VoxCPM2", load_denoiser=False)


async def _voxcpm_tts_stream(text, model=None):
    """Stessa forma di _local_tts_stream: sintesi → risampola a 24k → yield a blocchi."""
    m = _voxcpm_model()
    sr = m.tts_model.sample_rate
    for f in m.generate_streaming(text=text):
        out = resample(np.asarray(f, np.float32).reshape(-1), src=sr, dst=SAMPLE_RATE)
        yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


async def speak_stream(deltas, reply=None, on_delta=None):
    """Flusso di incrementi testo → flusso vocale. Sintesi e generazione **parallele**: quando viene completata una frase viene messa in coda, un'altra coroutine
    la prende per sintetizzare, suona mentre genera.

    ``deltas``: iteratore asincrono (``vm.reply_stream(turn)`` lo è).
    ``on_delta``: callback ogni volta che si riceve un incremento di testo (passalo se vuoi scrivere mentre parli).
    """
    queue: asyncio.Queue = asyncio.Queue()
    out: asyncio.Queue = asyncio.Queue()

    async def synth():
        while (seg := await queue.get()) is not None:
            async for pcm in tts_stream(seg, reply):
                await out.put(pcm)
        await out.put(None)

    worker = asyncio.create_task(synth())

    async def feed():
        buf, sent = "", 0
        try:
            async for d in deltas:
                if on_delta:
                    on_delta(d)
                buf += d
                if cut_point(buf, first=sent == 0):
                    await queue.put(buf.strip())
                    buf, sent = "", sent + 1
            if buf.strip():
                await queue.put(buf.strip())
        finally:
            await queue.put(None)               # Anche se la generazione va in errore, synth() deve terminare

    feeder = asyncio.create_task(feed())
    try:
        while (pcm := await out.get()) is not None:
            yield pcm
    finally:
        for t in (feeder, worker):
            if not t.done():
                t.cancel()
        await asyncio.gather(feeder, worker, return_exceptions=True)
