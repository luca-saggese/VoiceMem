"""Punto di configurazione unificato: un dict per configurare tutti i modelli locali/api (modello from_config ispirato a mem0).


Ogni componente è scritto come ``{"provider": ..., "config": {...}}``, aprendo un dict sai se ogni modello
usa locale o api. ``build_kwargs(config)`` analizza questo dict dichiarativo nei parametri di iniezione che
``VoiceMem(**kwargs)`` può consumare — è uno strato di zucchero sintattico **sopra** il meccanismo di iniezione esistente
``VoiceMem(embedding=fn, schema=fn, …)``, non modifica alcun comportamento esistente.


Un config completo è così (la config di ogni sezione è opzionale, se omessa usa i default integrati) ::

    CONFIG = {
        "api_key": "sk-...",              # Livello top, passato a VoiceMem (scritto anche in OPENAI_API_KEY)
        "base_url": None,                 # Endpoint OpenAI-compatible, per esempio OpenRouter
        "model_name": None,               # Alias del modello chat, per esempio openai/gpt-4o-mini
        "mode": "multi_modal",            # Livello top, passato a VoiceMem


        "embedding": {"provider": "local"},                 # Vettori memoria usano E5 locale
        "slots":     {"provider": "local"},                 # Classificazione slot usa E5 locale (0 LLM)
        "vad":       {"provider": "silero"},                # Rileva "ha finito di parlare"; custom per sostituire con il proprio
        "memory_engine": {"provider": "mem0"},              # Backend database vettoriale (default mem0)
        "tts": {"provider": "openai",                       # Audio (layer opzionale)
                "config": {"model": "gpt-4o-mini-tts", "voice": "coral"}},
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
    tts.provider           openai -> OpenAITTS (API OpenAI TTS, configurabile voice/instructions)
                           local  -> PiperTTS (piper offline, alias piper)
                           voxcpm -> VoxCPMTTS (VoxCPM2 offline)
                           breeze -> BreezeTTS (servizio streaming Breeze TTS 2, usa tono di voce con linguaggio naturale
                                     ; licenza non commerciale, quindi non è il default)
    models                 Nomi dei modelli per i cinque ruoli, vedi voicemem/llm_config.py
    llm.provider           openai -> scrive su OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL
    reply.provider         openai -> voicemem.reply.openai_reply (integrato, streaming)
                           custom -> l'oggetto callable in config.fn (equivalente a VoiceMem(reply=fn))


Provider non riconosciuti generano un errore chiaro.
"""
from __future__ import annotations

import os
from voicemem.llm_config import MODELS


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

    """embedding: local -> E5 locale; openai -> OpenAILocalEmbedder;
    gli altri nomi vengono passati a EmbedderFactory di mem0 (ollama / huggingface / gemini / …)."""

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

    # Gli altri li lascia a mem0 — ha自带的十多个 provider (ollama / huggingface / gemini /
    # bedrock / azure_openai / vertexai / together / lmstudio / fastembed /
    # langchain), e mem0 è già una dipendenza, non serve scriverli uno per uno.
    # I due integrati non seguono questo percorso: local è E5 locale che mem0 non ha, openai supporta qui parametri extra
    # come dimensions.
    from voicemem.leftbrain.mem0_embedder import mem0_providers
    known = mem0_providers()
    if provider in known:
        def make():
            from voicemem.leftbrain.mem0_embedder import Mem0Embedder
            return Mem0Embedder(provider, cfg)
        return make
    _bad("embedding", provider, ["local", "openai", *sorted(known)])


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





def _tts_factory(provider, cfg):
    """tts: openai -> API OpenAI TTS; local/piper -> piper offline; voxcpm -> VoxCPM2.

    Se provider è omesso segue la variabile d'ambiente TTS_BACKEND. Il nome del provider viene validato qui — se si aspetta il primo errore audio, sarebbe già a metà conversazione.
    """
    from voicemem.tts import TTS_PROVIDERS
    if provider is not None and str(provider).lower() not in TTS_PROVIDERS:
        _bad("tts", provider, sorted(set(TTS_PROVIDERS)))

    def make():
        from voicemem.tts import make_tts
        return make_tts(provider, **cfg)
    return make


# Nella forma annidata del demo della sezione reply (CONFIG["reply"] di web/run.py), questi tre sono nomi di sotto-sezioni invece che
# provider/config; il core riconosce solo llm al suo interno, tts / realtime restano gestiti dal web demo stesso.
_REPLY_DEMO_KEYS = ("llm", "tts", "realtime")

#: Chiavi di livello superiore riconosciute da build_kwargs. Un nome di chiave sbagliato prima veniva **silenziosamente ignorato** — la configurazione sembrava scritta ma
#: in realtà non aveva effetto, molto più difficile da debuggare di un errore (MODELS.update ha lo stesso atteggiamento verso i nomi dei ruoli).
_KNOWN_TOP = {
    "api_key", "base_url", "model_name", "mode", "memory_root", "user_id", "space", "models",
    "embedding", "slots", "vad", "memory_engine", "llm", "tts", "reply",
    "top_k", "memory_language",
}


def _check_keys(config: dict) -> None:
    unknown = sorted(set(config) - _KNOWN_TOP)
    if unknown:
        raise ValueError(f"config contiene chiavi non riconosciute: {', '.join(unknown)}."
                         f"Le disponibili sono: {', '.join(sorted(_KNOWN_TOP))}")
    # La sezione reply ha due forme: piatta {"provider","config"}, o quella del demo
    # {"llm","tts","realtime"} annidata. Nella forma annidata il core consuma solo llm, tts va allo slot di livello superiore con lo stesso nome,
    # realtime viene letto dal web demo stesso — l'appartenenza è scritta qui, non è "analizzato ma ignorato".
    seg = config.get("reply")
    if isinstance(seg, dict) and any(k in seg for k in _REPLY_DEMO_KEYS):
        bad = sorted(set(seg) - set(_REPLY_DEMO_KEYS))
        if bad:
            raise ValueError(f"La sezione reply (forma annidata) contiene chiavi non riconosciute: {', '.join(bad)}."
                             f"Le disponibili sono: {', '.join(_REPLY_DEMO_KEYS)}")


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
    _check_keys(config)

    # ── Livello top: api_key / base_url / mode passati direttamente ──
    if config.get("api_key") is not None:
        kwargs["api_key"] = config["api_key"]
    if config.get("base_url") is not None:
        kwargs["base_url"] = config["base_url"]
    if config.get("model_name") is not None:
        kwargs["model_name"] = config["model_name"]
    if config.get("mode") is not None:
        kwargs["mode"] = config["mode"]
    if config.get("memory_root") is not None:
        kwargs["memory_root"] = config["memory_root"]
    if config.get("user_id") is not None:
        kwargs["user_id"] = config["user_id"]
    if config.get("space") is not None:
        kwargs["space"] = config["space"]
    if config.get("top_k") is not None:
        kwargs["top_k"] = config["top_k"]
    if config.get("memory_language") is not None:
        kwargs["memory_language"] = config["memory_language"]

    # ── models: nomi dei modelli per i cinque ruoli, ognuno selezionabile separatamente (chat / reply / embedding /
    #    tts / realtime, vedi voicemem/llm_config.py). Più esterno rispetto al model nelle sezioni dei componenti qui sotto —
    #    la scrittura nella sezione del componente è più vicina, ma ha comunque priorità.──
    if config.get("models"):
        MODELS.update(config["models"])

    # ── embedding: la chiave di iniezione di VoiceMem è embedding ──
    if "embedding" in config:
        provider, cfg = _split(config["embedding"])
        kwargs["embedding"] = _embedding_factory(provider, cfg)

    # ── slots: mappatura alla chiave di iniezione schema di VoiceMem (classificatore per Classify)──
    if "slots" in config:
        provider, cfg = _split(config["slots"])
        kwargs["slots"] = _slots_factory(provider, cfg)

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

    # ── llm: LLM interno cervello sinistro/destro. ``llm.model`` è una scorciatoia per ``models.chat`` (stessa cosa,
    #    se entrambi sono scritti prevale la sezione llm解析ata dopo); questa sezione gestisce anche api_key / base_url.
    #    Il codice esistente legge OPENAI_MODEL / OPENAI_API_KEY /
    #    OPENAI_BASE_URL come env, qui si scrivono questi env dal config (api_key/base_url
    #    anche passati a VoiceMem params, coerente con il livello top).──

    if "llm" in config:
        provider, cfg = _split(config["llm"])
        if provider not in (None, "openai"):
            _bad("llm", provider, ["openai"])
        if cfg.get("model"):
            # Solo su MODELS, non scrivere anche env: ho verificato, OPENAI_MODEL non viene letto da nessun
            # library di terze parti (né openai SDK né mem0 lo leggono), scriverlo serve solo a mantenere uno stato globale coerente.
            # api_key / base_url sono diversi — quei SDK e mem0 li leggono direttamente, devono essere scritti.
            MODELS.update(chat=cfg["model"])
        if cfg.get("api_key"):
            os.environ["OPENAI_API_KEY"] = cfg["api_key"]
            kwargs.setdefault("api_key", cfg["api_key"])
        if cfg.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = cfg["base_url"]
            kwargs.setdefault("base_url", cfg["base_url"])


    # ── tts: nono slot intercambiabile. Riconosciuto in due posti — livello top "tts", o reply.tts del demo
    #    (web/run.py ha sempre scritto così, prima il core non leggeva, era inutile; ora legge). Livello top ha priorità.──
    tts_seg = config.get("tts")
    if tts_seg is None:
        _r = config.get("reply") or {}
        if any(k in _r for k in _REPLY_DEMO_KEYS):
            tts_seg = _r.get("tts")
    if tts_seg is not None:
        provider, cfg = _split(tts_seg)
        kwargs["tts"] = _tts_factory(provider, cfg)

    # ── reply：回复模型。两种写法——扁平 {"provider","config"}，或 demo 那份
    #    {"llm","tts","realtime"} 嵌套（核心只取 llm，tts/realtime 仍由 web 自己读）。──

    if "reply" in config:
        seg = config["reply"] or {}
        if any(k in seg for k in _REPLY_DEMO_KEYS):
            seg = seg.get("llm") or {}
        provider, cfg = _split(seg)
        kwargs["reply"] = _reply_factory(provider, cfg)

    return kwargs
