"""Opt-in end-to-end tests derived from the README examples.

These tests intentionally require external models/API credentials and are skipped
by the normal test suite. Run them with ``VOICEMEM_E2E=1`` and ``OPENAI_API_KEY``
set after downloading the models described in the README.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIO_FIXTURE = ROOT / "assets" / "input.wav"


def _e2e_enabled() -> bool:
    return os.environ.get("VOICEMEM_E2E", "0") == "1"


def _require_e2e_dependencies(audio: bool = False) -> None:
    if not _e2e_enabled():
        pytest.skip("opt-in E2E test: set VOICEMEM_E2E=1")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("E2E test requires OPENAI_API_KEY")
    if audio:
        model_dir = ROOT / "models" / "asr"
        required = (
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        )
        if not any(all((path / name).exists() for name in required)
                   for path in model_dir.glob("*") if path.is_dir()):
            pytest.skip("E2E audio test requires the Nemotron model in models/asr")
        if not AUDIO_FIXTURE.exists():
            pytest.fail(f"Missing audio fixture: {AUDIO_FIXTURE}")


def _local_leftbrain_config(memory_root: Path) -> dict:
    return {
        "mode": "leftbrain_only",
        "api_key": os.environ["OPENAI_API_KEY"],
        "memory_root": str(memory_root),
        "embedding": {"provider": "local"},
        "slots": {"provider": "local"},
    }


def test_readme_leftbrain_text_end_to_end(tmp_path: Path) -> None:
    """README leftbrain_only example: ingest factual text, then retrieve it."""
    _require_e2e_dependencies()

    from voicemem import VoiceMem

    vm = VoiceMem.from_config(_local_leftbrain_config(tmp_path / "text-space"))
    result = vm.ingest("Sono vegetariano e allergico alla frutta secca.")

    assert isinstance(result, dict)
    assert result["facts_count"] >= 1
    assert result["memory_ids"]

    search_result = vm.search("Quali sono le mie restrizioni alimentari?", top_k=5)
    assert search_result.result_leftbrain
    retrieved = " ".join(search_result.result_leftbrain).lower()
    assert "vegetar" in retrieved or "frutta secca" in retrieved


def test_readme_audio_wav_end_to_end(tmp_path: Path) -> None:
    """README audio example: ingest the WAV fixture and retrieve its memory."""
    _require_e2e_dependencies(audio=True)

    from voicemem import VoiceMem

    config = _local_leftbrain_config(tmp_path / "audio-space")
    config["mode"] = "normal"
    vm = VoiceMem.from_config(config)

    transcript = vm.transcribe(str(AUDIO_FIXTURE))
    assert isinstance(transcript, str)
    assert transcript.strip(), "Nemotron returned an empty transcript"

    # input.wav è un fixture audio del repository classificato come musica dal
    # detector; il README supporta esplicitamente text+audio, così il testo
    # fattuale viene salvato mentre l'audio attraversa percezione/voiceprint.
    result = vm.ingest(
        text="Sono vegetariano e allergico alla frutta secca.",
        audio=str(AUDIO_FIXTURE),
    )
    assert isinstance(result, dict)
    assert result["facts_count"] >= 1 or result["memory_ids"]

    search_result = vm.search("Quali informazioni importanti contiene questa frase?", top_k=5)
    assert search_result.result_leftbrain or search_result.result_rightbrain
