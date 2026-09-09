"""Factory di implementazioni default integrate per le varie capacità di voicemem (nome util -> factory senza parametri).

Le Utils di core.py le usano per creare i default; passare funzioni a VoiceMem(embedding=..., slots=...) sovrascrive le voci corrispondenti.
Nove posizioni: embedding / schema / entity / emotion / voiceprint / asr / vad /
memory_engine / tts. Le prime otto sono sul percorso core (quale caricare dipende da _NEED secondo mode),
tts non è incluso — il sistema memoria arriva solo al testo, l'audio è un layer opzionale.
Messe qui invece che in core.py, così il facade di livello superiore parla solo di "scheletro del sistema", senza essere gonfiato dagli import delle implementazioni default specifiche.
"""
from __future__ import annotations

import os


def default_utils(base_url, memory_root):
    def embedding():
        from voicemem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
        return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=base_url))
    def slots():
        # Classificatore E5 locale default: 0 LLM, 0 rete — nel budget di prefetch speculativo 0-300ms non si può usare la rete,
        # e Classify è proprio su quel percorso (_speculate in voicemem/stream.py).
        # sentence-transformers non è nelle dipendenze base (viene installato con l'extra [demo]), se manca fa fallback alla
        # versione LLM con una riga di spiegazione — un fallback silenzioso significa iniziare a spendere di nascosto.
        # VOICEMEM_SLOTS=openai forza l'uso della versione LLM (quando serve estrazione entity / discesa sottoclassi).
        if os.environ.get("VOICEMEM_SLOTS", "local").lower() != "openai":
            try:
                from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
                from voicemem.leftbrain.local_e5_embedder import shared_e5
                return LocalQueryClassifier(model=shared_e5())   # Condivide la stessa istanza E5 con l'embedder locale
            except ImportError as e:
                print(f"[slots] classificatore locale non disponibile ({e}) → fallback a QuerySlotClassifier versione LLM."
                      "Installa sentence-transformers (o pip install -e '.[demo]') per usare la versione locale.",
                      flush=True)
        from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
        return QuerySlotClassifier()
    def entity():
        from voicemem.leftbrain.cognitive_graph.annotator import CognitiveAnnotator, CognitiveAnnotatorConfig
        return CognitiveAnnotator(CognitiveAnnotatorConfig(base_url=base_url))
    def emotion():
        from voicemem.utils.audio.emotion.paper_emotion_detector import PaperAlignedEmotionDetector
        return PaperAlignedEmotionDetector()
    def voiceprint():
        from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
        return SpeakerEncoder(device="cpu")
    def asr():
        # Nemotron 3.5 ASR Streaming 0.6B (EN/IT) — unico backend ASR.
        # La lingua viene dalla Memory Space tramite lang.py.
        from voicemem.utils.audio.asr import NemotronStreamingASR
        from voicemem.utils.common.paths import models_dir
        from voicemem.lang import memory_language
        space_lang = memory_language()
        model_dir = models_dir() / "asr" / "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
        return NemotronStreamingASR(model_dir=model_dir, language=space_lang)
    def vad():
        # 判「说完了」的 VAD。默认内置 silero；换自己的传一个有 is_speech(frame)->bool
        # 的对象即可（VoiceMem(vad=lambda: MyVad()) 或 config 的 vad 段）。
        from voicemem.utils.audio.stream_io import make_vad
        return make_vad()
    def tts():
        # 第九个可替换位。核心链路不用它——记忆系统只到文本为止，出声是调用方的事，
        # 所以 tts 不进 _NEED（warmup 不会拉起来），谁要出声谁 utils.get("tts")。
        # 不把 base_url 传下去：那个通常指向自建 LLM/embedding 服务，多半没有
        # /audio/speech，跟过去只会在出声时才炸。要换端点用 OPENAI_TTS_BASE_URL。
        from voicemem.tts import make_tts
        return make_tts()
    def memory_engine():
        from pathlib import Path
        from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore
        # memory_root 由 Orchestrator 传下来（已解析过默认值）；这里的兜底只在
        # 直接构造 default_utils 时用得上，跟上面保持同一个默认。
        return Mem0BackendStore(embedding(),
                                memory_root=Path(memory_root or Path.cwd() / "voicemem_memory"))
    return {"embedding": embedding, "slots": slots, "entity": entity, "emotion": emotion,
            "voiceprint": voiceprint, "asr": asr, "vad": vad, "memory_engine": memory_engine,
            "tts": tts}