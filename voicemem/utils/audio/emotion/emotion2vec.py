"""Acoustic emotion classification with emotion2vec+ base.

The classifier is deliberately independent from ASR: it consumes the complete
utterance audio after VAD closes the turn and returns the raw nine-class scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


EXPECTED_LABELS = frozenset({
    "angry", "disgusted", "fearful", "happy", "neutral",
    "other", "sad", "surprised", "unknown",
})


@dataclass(frozen=True)
class AcousticEmotionResult:
    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)


class Emotion2VecClassifier:
    """Single emotion2vec+ model instance, reused across utterances."""

    MODEL_ID = "iic/emotion2vec_plus_base"

    def __init__(self, model: Any = None, model_id: str = MODEL_ID) -> None:
        if model is None:
            from funasr import AutoModel
            model = AutoModel(model=model_id, hub="hf")
        self.model = model

    def classify(self, samples: np.ndarray, sample_rate: int = 16000) -> AcousticEmotionResult:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            raise ValueError("Emotion2VecClassifier richiede audio a 16000 Hz")
        if samples.size == 0:
            return AcousticEmotionResult("unknown", 0.0, {})
        try:
            raw = self.model.generate(
                input=samples,
                granularity="utterance",
                extract_embedding=False,
            )
            return self._parse(raw)
        except Exception as exc:
            print(f"[emotion2vec] classificazione saltata: {exc}", flush=True)
            return AcousticEmotionResult("unknown", 0.0, {})

    @staticmethod
    def _parse(raw: Any) -> AcousticEmotionResult:
        item = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        if not isinstance(item, dict):
            return AcousticEmotionResult("unknown", 0.0, {})
        labels = item.get("labels") or []
        raw_scores = item.get("scores") or []
        if not labels or not raw_scores or len(labels) != len(raw_scores):
            return AcousticEmotionResult("unknown", 0.0, {})
        scores: dict[str, float] = {}
        for label, score in zip(labels, raw_scores):
            label = str(label).strip().lower()
            # FunASR emotion2vec restituisce label bilingui, ad esempio
            # ``开心/happy`` e ``<unk>`` invece del nome inglese puro.
            if "/" in label:
                label = label.rsplit("/", 1)[-1].strip()
            if label in {"<unk>", "unk"}:
                label = "unknown"
            if label in EXPECTED_LABELS:
                try:
                    scores[label] = float(score)
                except (TypeError, ValueError):
                    continue
        if not scores:
            return AcousticEmotionResult("unknown", 0.0, {})
        label = max(scores, key=scores.get)
        return AcousticEmotionResult(label, scores[label], scores)
