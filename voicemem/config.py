"""Punto di configurazione unificato: un dict per configurare tutti i modelli locali/api (modello from_config ispirato a mem0).

Ogni componente è scritto come ``{"provider": ..., "config": {...}}``, aprendo un dict sai se ogni modello
usa locale o api. ``build_kwargs(config)`` analizza questo dict dichiarativo nei parametri di iniezione che
``VoiceMem(**kwargs)`` può consumare — è uno strato di zucchero sintattico **sopra** il meccanismo di iniezione esistente
``VoiceMem(embedding=fn, schema=fn, …)``, non modifica alcun comportamento esistente.

Un config completo è così (la config di ogni sezione è opzionale, se omessa usa i default integrati) ::

    CONFIG = {
        "api_key": "sk-...",              # Livello top, passato a VoiceMem (scritto anche in OPENAI_API_KEY)
        "base_url": None,                 # Livello top, passato a VoiceMem
        "mode": "multi_modal",            # Livello top, passato a VoiceMem

        "embedding": {"provider": "local"},                 # Vettori memoria usano E5 locale
        "slots":     {"provider": "local"},                 # Classificazione slot usa E5 locale (0 LLM)
        "vad":       {"provider": "silero"},                # Rileva "ha finito di parlare"; custom per sostituire con il proprio
        "memory_engine": {"provider": "mem0"},              # Backend database vettoriale (default mem0)
        "llm": {"provider": "openai",                       # LLM interno cervello sinistro/destro (etichettatura/attribution…)
                "config": {"model": "gpt-4o-mini", "api_key": "sk-...", "base_url": None}},

        # Sezione reply: modello di risposta. Entrambe le forme sono accettate —
        "reply": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
        # oppure la forma annidata del demo (usata da web/run.py), il core prende solo la sezione llm,
        # tts / realtime restano gestiti dal web demo stesso:
        # "reply": {
        #     "llm":      {"provider": "openai", "config": {"model": "gpt-4o"}},
        #     "tts":      {"provider": "openai", "config": {"model": "gpt-4o-mini-tts"}},
        #     "realtime": {"provider": "openai", "config": {"model": "gpt-realtime"}},
        # },
    }

Mappatura provider → implementazione integrata (semplice da capire a colpo d'occhio):

    embedding.provider     local  -> LocalE5Embedder (E5 locale, 0 rete)
                           openai -> OpenAILocalEmbedder (OpenAI Embeddings API)
    slots.provider         local  -> LocalQueryClassifier (E5 locale, 0 LLM)
                           openai -> QuerySlotClassifier (1 chiamata LLM)
    vad.provider           silero -> make_vad (integrato, config può dare model / threshold)
                           custom -> l'oggetto config.obj (deve avere is_speech(frame)->bool)
    memory_engine.provider mem0   -> Mem0BackendStore (default, anche omettendo usa il default integrato)
    llm.provider           openai -> scrive su OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL
    reply.provider         openai -> voicemem.reply.openai_reply (integrato, streaming)
                           custom -> l'oggetto callable in config.fn (equivalente a VoiceMem(reply=fn))

Provider non riconosciuti generano un errore chiaro.
"""
from __future__ import annotations

import os


def _split(component: dict | None) -> tuple[str, dict]:
    """Separa ``{"provider": ..., "config": {...}}`` in (provider, config); config è opzionale."""
    component = component or {}
    provider = component.get("provider")
    cfg = component.get("config") or {}
    return provider, cfg


def _bad(component: str, provider, known) -> None:
    raise ValueError(
        f"{component}.provider={provider!r} sconosciuto; opzioni: {' / '.join(known)}"
    )


def _embedding_factory(provider, cfg):
    """embedding: local -> LocalE5Embedder; openai -> OpenAILocalEmbedder."""
    if provider == "local":
        def make():
            from voicemem.leftbrain.local_e5_embedder import LocalE5Embedder
            return LocalE5Embedder()
        return make
    if provider == "openai":
        def make():
            from voicemem.leftbrain.local_memory_store import (
                OpenAILocalEmbedder, OpenAILocalEmbedderConfig,
            )
            return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(
                model=cfg.get("model"),
                api_key=cfg.get("api_key"),
                base_url=cfg.get("base_url"),
                dimensions=cfg.get("dimensions"),
            ))
        return make
    _bad("embedding", provider, ["local", "openai"])


def _slots_factory(provider, cfg):
    """slots: local -> LocalQueryClassifier; openai -> QuerySlotClassifier."""
    if provider == "local":
        def make():
            from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
            # Condivide la stessa istanza E5 con l'embedder locale (risparmia memoria), a meno che il chiamante non passi esplicitamente un model.
            kw = dict(cfg)
            if "model" not in kw:
                from voicemem.leftbrain.local_e5_embedder import shared_e5
                kw["model"] = shared_e5()
            return LocalQueryClassifier(**kw)
        return make
    if provider == "openai":
        def make():
            from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
            return QuerySlotClassifier()
        return make
    _bad("slots", provider, ["local", "openai"])


def _vad_factory(provider, cfg):
    """vad: silero -> make_vad integrato (configurabile model/threshold); custom -> l'oggetto config.obj."""
    if provider in (None, "silero"):
        def make():
            from voicemem.utils.audio.stream_io import make_vad
            return make_vad(model=cfg.get("model"), threshold=cfg.get("threshold", 0.5))
        return make
    if provider == "custom":
        obj = cfg.get("obj")
        if obj is None or not hasattr(obj, "is_speech"):
            raise ValueError(
                'vad.provider="custom" richiede config.obj con un oggetto che ha is_speech(frame)->bool '
                "; più semplice iniettare direttamente: VoiceMem(vad=lambda: MyVad())"
            )
        return lambda: obj
    _bad("vad", provider, ["silero", "custom"])


def _memory_engine_factory(provider, cfg):
    """memory_engine: mem0 -> Mem0BackendStore (default, anche omettendo usa il default integrato)."""
    if provider == "mem0":
        # None → lascia che VoiceMem usi il memory_engine default integrato (cioè Mem0BackendStore).
        # Non serve costruirlo esplicitamente qui: il default integrato è già mem0, ometterlo è più semplice e semanticamente coerente.
        return None
    _bad("memory_engine", provider, ["mem0"])


# Nella forma annidata del demo della sezione reply (CONFIG["reply"] di web/run.py), questi tre sono nomi di sotto-sezioni invece che
# provider/config; il core riconosce solo llm al suo interno, tts / realtime restano gestiti dal web demo stesso.
_REPLY_DEMO_KEYS = ("llm", "tts", "realtime")


def _reply_factory(provider, cfg):
    """reply: openai -> provider integrato streaming; custom -> usa direttamente la funzione config.fn."""
    if provider in (None, "openai"):
        from voicemem.reply import openai_reply
        return openai_reply(model=cfg.get("model"), api_key=cfg.get("api_key"),
                            base_url=cfg.get("base_url"), system=cfg.get("system"))
    if provider == "custom":
        fn = cfg.get("fn")
        if not callable(fn):
            raise ValueError(
                'reply.provider="custom" richiede config.fn con un oggetto callable; '
                "; passare la funzione direttamente è più semplice: VoiceMem(reply=fn)"
            )
        return fn
    _bad("reply", provider, ["openai", "custom"])


def build_kwargs(config: dict) -> dict:
    """Analizza il dict config unificato nel dict di parametri di iniezione che ``VoiceMem(**kwargs)`` può consumare.

    Il dict restituito contiene solo le voci effettivamente fornite (i componenti default non inseriscono la chiave, lasciando a VoiceMem usare i default integrati):
    ``api_key`` / ``base_url`` / ``mode`` + funzioni di iniezione come ``embedding`` / ``schema`` /
    ``memory_engine`` (factory senza parametri, semanticamente equivalente a ``VoiceMem(embedding=lambda: ...)``).

    La sezione ``reply`` viene analizzata in ``VoiceMem(reply=fn)``; nella forma annidata del demo si prende solo ``llm``,
    ``tts`` / ``realtime`` restano letti dal web demo stesso.
    """
    config = config or {}
    kwargs: dict = {}

    # ── Livello top: api_key / base_url / mode passati direttamente ──
    if config.get("api_key") is not None:
        kwargs["api_key"] = config["api_key"]
    if config.get("base_url") is not None:
        kwargs["base_url"] = config["base_url"]
    if config.get("mode") is not None:
        kwargs["mode"] = config["mode"]
    if config.get("memory_root") is not None:
        kwargs["memory_root"] = config["memory_root"]
    if config.get("user_id") is not None:
        kwargs["user_id"] = config["user_id"]
    if config.get("space") is not None:
        kwargs["space"] = config["space"]

    # ── embedding: la chiave di iniezione di VoiceMem è embedding ──
    if "embedding" in config:
        provider, cfg = _split(config["embedding"])
        kwargs["embedding"] = _embedding_factory(provider, cfg)

    # ── slots: mappatura alla chiave di iniezione schema di VoiceMem (classificatore per Classify)──
    if "slots" in config:
        provider, cfg = _split(config["slots"])
        kwargs["schema"] = _slots_factory(provider, cfg)

    # ── vad: VAD che rileva "ha finito di parlare" (usato da VoiceStream)──
    if "vad" in config:
        provider, cfg = _split(config["vad"])
        kwargs["vad"] = _vad_factory(provider, cfg)

    # ── memory_engine: mem0 è il default integrato, se restituisce None non sovrascrive ──
    if "memory_engine" in config:
        provider, cfg = _split(config["memory_engine"])
        factory = _memory_engine_factory(provider, cfg)
        if factory is not None:
            kwargs["memory_engine"] = factory

    # ── llm: LLM interno cervello sinistro/destro. Il codice esistente legge OPENAI_MODEL / OPENAI_API_KEY /
    #    OPENAI_BASE_URL come env, qui si scrivono questi env dal config (api_key/base_url
    #    anche passati a VoiceMem params, coerente con il livello top).──
    if "llm" in config:
        provider, cfg = _split(config["llm"])
        if provider not in (None, "openai"):
            _bad("llm", provider, ["openai"])
        if cfg.get("model"):
            os.environ["OPENAI_MODEL"] = cfg["model"]
        if cfg.get("api_key"):
            os.environ["OPENAI_API_KEY"] = cfg["api_key"]
            kwargs.setdefault("api_key", cfg["api_key"])
        if cfg.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = cfg["base_url"]
            kwargs.setdefault("base_url", cfg["base_url"])

    # ── reply: modello di risposta. Due forme — piatta {"provider","config"}, o quella del demo
    #    {"llm","tts","realtime"} annidata (il core prende solo llm, tts/realtime restano letti dal web).──
    if "reply" in config:
        seg = config["reply"] or {}
        if any(k in seg for k in _REPLY_DEMO_KEYS):
            seg = seg.get("llm") or {}
        provider, cfg = _split(seg)
        kwargs["reply"] = _reply_factory(provider, cfg)

    return kwargs
