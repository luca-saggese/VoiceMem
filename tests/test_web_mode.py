from __future__ import annotations

import os
import sys


# web/run.py parses argv at import time; import it with an empty argv.
def _load_run(monkeypatch, base_url: str):
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(sys, "argv", ["run.py"])
    sys.modules.pop("run", None)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "web"))
    import run
    return run


def test_openrouter_defaults_to_llm_tts(monkeypatch):
    run = _load_run(monkeypatch, "https://openrouter.ai/api/v1")
    assert run.ARGS.mode == "llm_tts"


def test_openai_defaults_to_llm_tts(monkeypatch):
    run = _load_run(monkeypatch, "https://api.openai.com/v1")
    assert run.ARGS.mode == "llm_tts"
