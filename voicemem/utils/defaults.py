"""Factory di implementazioni default integrate per le varie capacità di voicemem (nome util -> factory senza parametri).

Le Utils di core.py le usano per creare i default; passare funzioni a VoiceMem(embedding=..., schema=...) sovrascrive le voci corrispondenti.
Messe qui invece che in core.py, così il facade di livello superiore parla solo di "scheletro del sistema", senza essere gonfiato dagli import delle implementazioni default specifiche."
"""
from __future__ import annotations

import os


def default_utils(base_url, memory_root):
    def embedding():
        from voicemem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
        return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=base_url))
    def schema():
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
        # FunASR paraformer-zh-streaming default (più accurato per cinese); VOICEMEM_ASR=sherpa fa fallback a
        # sherpa-onnx streaming zipformer (bilingue cinese-inglese, puro onnx senza dipendenze torch).
        if os.environ.get("VOICEMEM_ASR", "funasr").lower() == "sherpa":
            from voicemem.utils.audio.asr import StreamingASR
            from voicemem.utils.common.paths import model_path
            return StreamingASR(str(model_path(
                "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20", kind="asr")))
        from voicemem.utils.audio.asr import FunASRStreamingASR
        return FunASRStreamingASR()
    def vad():
        # VAD per rilevare "ha finito di parlare". Silero integrato come default; per usare il proprio passa un oggetto con is_speech(frame)->bool
        # (VoiceMem(vad=lambda: MyVad()) o la sezione vad del config).
        from voicemem.utils.audio.stream_io import make_vad
        return make_vad()
    def memory_engine():
        from pathlib import Path
        from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore
        # memory_root viene passato da Orchestrator (valori default già risolti); il fallback qui è utile solo
        # quando si costruisce default_utils direttamente, mantiene lo stesso default di sopra.
        return Mem0BackendStore(embedding(),
                                memory_root=Path(memory_root or Path.cwd() / "voicemem_memory"))
    return {"embedding": embedding, "schema": schema, "entity": entity, "emotion": emotion,
            "voiceprint": voiceprint, "asr": asr, "vad": vad, "memory_engine": memory_engine}
