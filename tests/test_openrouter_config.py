"""Tests for OpenAI-compatible endpoint/model configuration."""
from __future__ import annotations

import inspect

from voicemem.config import build_kwargs
from voicemem.core import VoiceMem
from voicemem.llm_config import MODELS, resolve_base_url, resolve_model


def test_voice_mem_exposes_model_name_and_base_url():
    signature = inspect.signature(VoiceMem.__init__)
    assert "base_url" in signature.parameters
    assert "model_name" in signature.parameters


def test_from_config_accepts_openrouter_endpoint_and_model(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    kwargs = build_kwargs({
        "api_key": "sk-or-test",
        "base_url": "https://openrouter.ai/api/v1",
        "model_name": "openai/gpt-4o-mini",
        "mode": "leftbrain_only",
    })

    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert kwargs["model_name"] == "openai/gpt-4o-mini"


def test_model_name_resolves_as_chat_model(monkeypatch):
    previous = MODELS._over.copy()
    try:
        MODELS._over.clear()
        MODELS.update(chat="openai/gpt-4o-mini")
        assert resolve_model(role="chat") == "openai/gpt-4o-mini"
    finally:
        MODELS._over.clear()
        MODELS._over.update(previous)


def test_base_url_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    assert resolve_base_url("https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1"
    assert resolve_base_url() == "https://api.openai.com/v1"
