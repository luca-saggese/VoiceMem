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
        "tts": {"provider": "openai",                       # 出声（可选的一层）
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
    tts.provider           openai -> OpenAITTS（OpenAI TTS api，可配 voice/instructions）
                           local  -> PiperTTS（离线 piper，别名 piper）
                           voxcpm -> VoxCPMTTS（离线 VoxCPM2）
                           breeze -> BreezeTTS（Breeze TTS 2 流式服务，可用自然语言
                                     指挥语气；权重非商用许可，故不做默认）
    models                 五个角色的模型名，见 voicemem/llm_config.py
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

    """embedding：local -> 本地 E5；openai -> OpenAILocalEmbedder；
    其余名字转给 mem0 的 EmbedderFactory（ollama / huggingface / gemini / …）。"""

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

    # 其余的交给 mem0——它自带十来个 provider（ollama / huggingface / gemini /
    # bedrock / azure_openai / vertexai / together / lmstudio / fastembed /
    # langchain），而 mem0 本来就是依赖，没必要各写一遍。
    # 内置那两个不走这条：local 是 mem0 没有的本地 E5，openai 这边多支持
    # dimensions 这类参数。
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
    """tts：openai -> OpenAI TTS api；local/piper -> 离线 piper；voxcpm -> VoxCPM2。

    provider 省了就跟 TTS_BACKEND 环境变量。provider 名在这儿就校验掉——留到第一次
    出声才报错的话，那已经是在对话中间了。
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

#: build_kwargs 认识的顶层键。写错一个键名以前是**静默忽略**——配置看着写了、
#: 实际一点没生效，比报错难查得多（MODELS.update 对角色名也是同样的态度）。
_KNOWN_TOP = {
    "api_key", "base_url", "mode", "memory_root", "user_id", "space", "models",
    "embedding", "slots", "vad", "memory_engine", "llm", "tts", "reply",
    "top_k", "memory_language",
}


def _check_keys(config: dict) -> None:
    unknown = sorted(set(config) - _KNOWN_TOP)
    if unknown:
        raise ValueError(f"config 里有不认识的键：{', '.join(unknown)}。"
                         f"可用的是：{', '.join(sorted(_KNOWN_TOP))}")
    # reply 段有两种形状：扁平 {"provider","config"}，或 demo 那份
    # {"llm","tts","realtime"} 嵌套。嵌套里核心只消费 llm，tts 落到顶层同名能力，
    # realtime 归 web demo 自己读——归属写在这儿，不是"解析了却不管"。
    seg = config.get("reply")
    if isinstance(seg, dict) and any(k in seg for k in _REPLY_DEMO_KEYS):
        bad = sorted(set(seg) - set(_REPLY_DEMO_KEYS))
        if bad:
            raise ValueError(f"reply 段（嵌套写法）里有不认识的键：{', '.join(bad)}。"
                             f"可用的是：{', '.join(_REPLY_DEMO_KEYS)}")


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

    # ── models：五个角色的模型名，每个都能单独选（chat / reply / embedding /
    #    tts / realtime，见 voicemem/llm_config.py）。比下面各组件段里的 model
    #    靠外——组件段写的更就近，仍然优先。──
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

    # ── llm：左右脑内部 LLM。``llm.model`` 是 ``models.chat`` 的简写（同一件事，
    #    两个都写时后解析的 llm 段生效）；这一段还额外管 api_key / base_url。
    #    现有代码读 OPENAI_MODEL / OPENAI_API_KEY /
    #    OPENAI_BASE_URL 这些 env，这里把 config 落到这些 env（api_key/base_url
    #    也透传给 VoiceMem 参数，保持和顶层一致）。──

    if "llm" in config:
        provider, cfg = _split(config["llm"])
        if provider not in (None, "openai"):
            _bad("llm", provider, ["openai"])
        if cfg.get("model"):
            # 只落在 MODELS 上，不再顺手写 env：查过了，OPENAI_MODEL 没有任何
            # 第三方库读（openai SDK / mem0 都不读），写它纯粹是让全局状态多一份
            # 拷贝。api_key / base_url 不同——那两个 SDK 和 mem0 自己读，必须写。
            MODELS.update(chat=cfg["model"])
        if cfg.get("api_key"):
            os.environ["OPENAI_API_KEY"] = cfg["api_key"]
            kwargs.setdefault("api_key", cfg["api_key"])
        if cfg.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = cfg["base_url"]
            kwargs.setdefault("base_url", cfg["base_url"])


    # ── tts：第九个可替换位。两处都认——顶层 "tts"，或 demo 那份 reply.tts
    #    （web/run.py 一直这么写，以前核心不读、白写了，现在读）。顶层优先。──
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
