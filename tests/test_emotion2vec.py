"""Real emotion2vec+ integration tests.

Run with ``VOICEMEM_REAL_EMOTION2VEC=1``. The first run downloads
``iic/emotion2vec_plus_base`` through FunASR/Hugging Face.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicemem.utils.audio.emotion.emotion2vec import (
    EXPECTED_LABELS,
    AcousticEmotionResult,
    Emotion2VecClassifier,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_classifier() -> Emotion2VecClassifier:
    if os.environ.get("VOICEMEM_REAL_EMOTION2VEC") != "1":
        pytest.skip("set VOICEMEM_REAL_EMOTION2VEC=1 to run real emotion2vec tests")
    return Emotion2VecClassifier()


def _read_fixture(name: str) -> np.ndarray:
    samples, sample_rate = sf.read(ROOT / "assets" / name, dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    assert sample_rate == 16000
    return np.asarray(samples, dtype=np.float32).reshape(-1)


def test_real_emotion2vec_returns_valid_nine_class_distribution(real_classifier):
    result = real_classifier.classify(_read_fixture("input.wav"), 16000)
    assert isinstance(result, AcousticEmotionResult)
    assert result.label in EXPECTED_LABELS
    assert isinstance(result.score, float)
    assert result.scores
    assert set(result.scores) <= EXPECTED_LABELS
    assert all(isinstance(score, float) for score in result.scores.values())


def test_real_emotion2vec_handles_multiple_utterances_without_reloading(real_classifier):
    model_identity = id(real_classifier.model)
    results = [
        real_classifier.classify(_read_fixture(name), 16000)
        for name in ("input.wav", "speech.wav", "question.wav")
    ]
    assert id(real_classifier.model) == model_identity
    assert len(results) == 3
    assert all(result.label in EXPECTED_LABELS for result in results)


def test_real_emotion2vec_empty_audio_is_safe(real_classifier):
    result = real_classifier.classify(np.array([], dtype=np.float32), 16000)
    assert result == AcousticEmotionResult("unknown", 0.0, {})
