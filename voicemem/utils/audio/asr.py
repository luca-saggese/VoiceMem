"""语音转文字：流式识别（实时 partial）+ 非流式精转写（最终文本）。

流式实现，接口一致（``feed(samples) -> 累积文本`` / ``flush()`` / ``reset()``），
由 ``utils/defaults.py`` 的 ``asr`` 工厂选择：

  · ``NemotronStreamingASR``  Nemotron 3.5 ASR Streaming 0.6B (EN/IT, **默认**)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# ── Nemotron 3.5 ASR Streaming 0.6B (EN/IT) ────────────────────────────────────

class NemotronStreamingASR:
    """sherpa-onnx streaming recognizer per Nemotron 3.5 ASR (EN/IT multilingual).

    Contratto: ``feed(samples) -> str``, ``flush() -> str``, ``reset() -> None``.

    La lingua viene applicata ad ogni nuovo stream dopo reset/flush.
    Il recognizer viene costruito una sola volta; lo ``OnlineStream`` viene ricreato
    ad ogni ``reset()``/``flush()``.

    Lingue ammesse: ``("it", "en")``.
    File modello attesi nella directory del modello:

        encoder.int8.onnx
        decoder.int8.onnx
        joiner.int8.onnx
        tokens.txt
    """

    SUPPORTED_LANGUAGES = ("it", "en")

    def __init__(self, model_dir: str | Path, language: str = "it") -> None:
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Lingua non supportata: {language!r}. "
                f"Lingue ammesse: {', '.join(self.SUPPORTED_LANGUAGES)}"
            )
        model_dir = Path(model_dir)
        # Validazione esplicita dei file modello prima di creare il recognizer
        for fname in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
            fpath = model_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"File modello mancante: {fpath}\n"
                    f"Nemotron richiede: encoder.int8.onnx, decoder.int8.onnx, "
                    f"joiner.int8.onnx, tokens.txt"
                )
        self.language = language
        import sherpa_onnx
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model_dir / "tokens.txt"),
            encoder=str(model_dir / "encoder.int8.onnx"),
            decoder=str(model_dir / "decoder.int8.onnx"),
            joiner=str(model_dir / "joiner.int8.onnx"),
            num_threads=4,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )
        self._stream = self.rec.create_stream()
        self._text = ""

    def feed(self, samples) -> str:
        """Incollona campioni audio (float32 @ 16kHz). Restituisce testo parziale."""
        self._stream.set_option("language", self.language)
        self._stream.accept_waveform(SAMPLE_RATE, np.asarray(samples, dtype=np.float32))
        while self.rec.is_ready(self._stream):
            self.rec.decode_stream(self._stream)
        result = self.rec.get_result(self._stream)
        if result and getattr(result, "text", ""):
            self._text += result.text
        return self._text

    def flush(self) -> str:
        """Finalizza il testo corrente. Restituisce il testo completo della turnazione."""
        result = self.rec.get_result(self._stream)
        if result and getattr(result, "text", ""):
            self._text += result.text
        # Ricrea lo stream per la prossima utterance
        self._stream = self.rec.create_stream()
        self._stream.set_option("language", self.language)
        return self._text

    def reset(self) -> None:
        """Pulisce stato e testo, ricrea uno stream pulito con la lingua corrente."""
        self._text = ""
        self._stream = self.rec.create_stream()
        self._stream.set_option("language", self.language)


# ── Legacy ASR implementations (da eliminare in STEP 6) ────────────────────────

SENSEVOICE_EMOTION_MAP = {
    "NEUTRAL": "中性",
    "HAPPY": "开心",
    "ANGRY": "愤怒",
    "SAD": "悲伤",
    "FEARFUL": "恐惧",
    "FEAR": "恐惧",
    "DISGUSTED": "厌恶",
    "SURPRISED": "惊讶",
}


def pick_device() -> str:
    """自动选最佳设备: cuda > mps(Apple M) > cpu。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Transcriber:
    """SenseVoiceSmall 出最终文本（中英），比流式 ASR 更准，锁定一轮时用这个。"""

    def __init__(self, device: str) -> None:
        from funasr import AutoModel        # 懒 import：只有用非流式精转写才需要 funasr
        from voicemem.utils.common.paths import hf_model
        _name = hf_model("emotion", "FunAudioLLM/SenseVoiceSmall", "asr")
        self.model = AutoModel(model=_name, hub="hf",
                               device=device, disable_update=True,
                               trust_remote_code=False)

    def _generate(self, audio) -> str:
        res = self.model.generate(input=audio, cache={}, language="zh",
                                  use_itn=True, ban_emo_unk=True)
        if not res:
            return ""
        return res[0].get("text", "") or ""

    def run(self, audio) -> str:
        return re.sub(r"<\|[^|]*\|>", "", self._generate(audio)).strip()

    def run_with_emotion(self, audio) -> tuple[str, str]:
        """一次 SenseVoice 推理同时取得文本和声学情绪 token。"""
        raw = self._generate(audio)
        tags = re.findall(r"<\|([^|]+)\|>", raw.upper())
        emotion = next((SENSEVOICE_EMOTION_MAP[tag] for tag in tags
                        if tag in SENSEVOICE_EMOTION_MAP), "中性")
        return re.sub(r"<\|[^|]*\|>", "", raw).strip(), emotion
