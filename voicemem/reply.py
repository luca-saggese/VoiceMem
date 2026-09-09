"""Layer di risposta: il passo dopo che il core ha consegnato il ``Turn`` — due percorsi, un'unica interfaccia.

``voicemem/stream.py`` è il lato input (audio → memoria), qui è il lato output (memoria → risposta). Due percorsi:

    # Percorso A: usa quello integrato (API compatibile con OpenAI, streaming)
    vm = VoiceMem.from_config({"reply": {"provider": "openai",
                                         "config": {"model": "gpt-4o-mini"}}})

    # Percorso B: usa il tuo modello/funzione
    vm = VoiceMem(reply=my_fn)

I due percorsi offrono la stessa interfaccia di chiamata::

    answer = await vm.reply(turn)                      # ricevi tutto, restituisce la stringa completa
    async for delta in vm.reply_stream(turn):  ...     # streaming, carattere per carattere

``my_fn`` può essere scritta in uno di questi modi, ``normalize()`` li unifica tutti in un async generator::

    def       my_fn(text, memory_context) -> str          # sincrono: automaticamente girato in thread, non blocca l'event loop
    async def my_fn(text, memory_context) -> str          # coroutine
    async def my_fn(text, memory_context): yield delta    # async generator (streaming)

**TTS non è qui.** Il layer di risposta produce solo testo; per l'audio usa ``voicemem/tts.py`` —
``speak_stream(vm.reply_stream(turn))`` sintetizza mentre genera, vedi examples/03_simple_agent_with_voicemem_memory.py.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import AsyncIterator, Callable
from voicemem.llm_config import resolve_api_key, resolve_base_url, resolve_model

# memory_context è solo "cosa ricordi dell'utente", non contiene richieste di personalità/stile di per sé, quindi il provider integrato
# lo attacca dopo questa frase, invece di usarlo intero come system prompt.
DEFAULT_SYSTEM = "Sei un assistente vocale, rispondi brevemente e naturalmente."


def compose_system(memory_context: str, system: str | None = None) -> str:
    """Personalità + memoria → system prompt. Entrambi possono essere vuoti."""
    parts = [system or DEFAULT_SYSTEM]
    if memory_context:
        parts.append(memory_context)
    return "\n\n".join(parts)


def openai_reply(model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, system: str | None = None) -> Callable:
    """Provider di risposta integrato: API compatibile con OpenAI, output streaming. Restituisce una funzione async generator.



    模型走 ``reply`` 角色：``model`` 参数 → ``VOICEMEM_REPLY_MODEL`` → 跟随 ``chat``。
    回复是用户直接听得见的一路，所以单独留了一个角色让它能和后台整理记忆的模型
    分开配；不配就跟着 chat 走，不会出现"设了模型但回复还在用默认值"这种一半生效。
    ``import voicemem`` 不会因此要求有 key（client 首次调用时才建）。

    """
    client = None

    async def fn(text: str, memory_context: str = "") -> AsyncIterator[str]:
        nonlocal client
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=resolve_api_key(api_key),
                base_url=resolve_base_url(base_url),
            )
        stream = await client.chat.completions.create(
            model=resolve_model(model, "reply"),
            stream=True,
            messages=[{"role": "system", "content": compose_system(memory_context, system)},
                      {"role": "user", "content": text}],
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return fn


def normalize(fn: Callable) -> Callable:
    """Normalizza qualsiasi forma di funzione di risposta a un unico tipo: "funzione async generator".

    Le funzioni sincrone usano ``asyncio.to_thread`` — la generazione della risposta dura secondi, eseguirla direttamente nel event loop bloccherebbe
    il thread di lettura del microfono. Se il valore di ritorno è esso stesso un oggetto asincronamente iterabile (es. un lambda che avvolge un generatore di altri), viene comunque espanso in streaming.
    """
    if inspect.isasyncgenfunction(fn):
        return fn

    if inspect.iscoroutinefunction(fn):
        async def gen(text: str, memory_context: str = "") -> AsyncIterator[str]:
            out = await fn(text, memory_context)
            if hasattr(out, "__aiter__"):
                async for delta in out:
                    yield delta
            else:
                yield out
        return gen

    async def gen(text: str, memory_context: str = "") -> AsyncIterator[str]:
        out = await asyncio.to_thread(fn, text, memory_context)
        if hasattr(out, "__aiter__"):
            async for delta in out:
                yield delta
        else:
            yield out
    return gen


async def capture(deltas: AsyncIterator[str], on_done: Callable[[str], None]) -> AsyncIterator[str]:
    """Pass-through di ogni delta così com'è, quando ha finito consegna la frase intera a ``on_done``.

    Anche la parte detta dall'agent dovrebbe entrare nella memoria, ma non si deve far scrivere una riga in più al chiamante, né aspettare di avere tutto per restituire.
    Quando viene interrotto, ``finally`` consegna la parte già restituita — quanto sente l'utente viene salvato quanto basta.
    """
    parts: list[str] = []
    try:
        async for delta in deltas:
            parts.append(delta)
            yield delta
    finally:
        on_done("".join(parts))


def unpack(turn_or_text, memory_context: str = "") -> tuple[str, str]:
    """Comodità per ``vm.reply(turn)``: Turn / StreamState viene decomposto direttamente in (text, memory_context)."""
    text = getattr(turn_or_text, "text", None)
    if text is not None and hasattr(turn_or_text, "memory_context"):
        return text, (memory_context or turn_or_text.memory_context)
    return turn_or_text, memory_context
