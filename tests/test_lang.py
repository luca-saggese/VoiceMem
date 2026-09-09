"""Unit tests per voicemem.lang — STEP 4."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLangModule:
    def test_default_is_italian(self):
        """Il default è 'it'."""
        from voicemem.lang import DEFAULT
        assert DEFAULT == "it"

    def test_supported_languages(self):
        """SUPPORTED contiene solo it ed en."""
        from voicemem.lang import SUPPORTED
        assert SUPPORTED == ("it", "en")

    def test_it_accepted(self):
        """'it' è accettato come lingua supportata."""
        from voicemem.lang import _check
        result = _check("it")
        assert result == "it"

    def test_en_accepted(self):
        """'en' è accettato come lingua supportata."""
        from voicemem.lang import _check
        result = _check("en")
        assert result == "en"

    def test_zh_rejected(self):
        """'zh' non è supportato → ValueError."""
        from voicemem.lang import _check
        with pytest.raises(ValueError, match="memory_language"):
            _check("zh")

    def test_zh_cn_rejected(self):
        """Tag lingua sconosciuto → ValueError."""
        from voicemem.lang import _check
        with pytest.raises(ValueError, match="memory_language"):
            _check("zh-CN")

    def test_memory_language_returns_it_by_default(self):
        """memory_language() restituisce 'it' di default."""
        from voicemem.lang import memory_language
        # Reset override per test pulito
        from voicemem import lang
        lang._override = None
        assert memory_language() == "it"

    def test_memory_language_respects_override(self):
        """memory_language() rispetta l'override."""
        from voicemem.lang import memory_language, set_memory_language
        set_memory_language("en")
        assert memory_language() == "en"
        set_memory_language(None)  # reset

    def test_is_it_function_exists(self):
        """is_it() esiste e funziona."""
        from voicemem.lang import is_it
        from voicemem import lang
        lang._override = "it"
        assert is_it() is True

    def test_is_en_function_exists(self):
        """is_en() esiste e funziona."""
        from voicemem.lang import is_en
        from voicemem import lang
        lang._override = "en"
        assert is_en() is True

    def test_no_is_zh_function(self):
        """Non deve esistere is_zh() — sostituita da is_it()/is_en()."""
        from voicemem import lang
        assert not hasattr(lang, "is_zh"), "is_zh() deve essere eliminato"

    def test_no_chinese_in_supported(self):
        """Nessun riferimento a zh in SUPPORTED o DEFAULT."""
        from voicemem.lang import SUPPORTED, DEFAULT
        assert "zh" not in SUPPORTED
        assert "zh" not in DEFAULT
        assert "chinese" not in DEFAULT.lower()

    def test_label_rule_uses_italian(self):
        """label_rule() usa Italiano quando la lingua è it."""
        from voicemem.lang import label_rule
        from voicemem import lang
        lang._override = "it"
        rule = label_rule()
        assert "Italiano" in rule
        assert "Chinese" not in rule
        lang._override = None

    def test_display_emotion_italian(self):
        """display_emotion() restituisce traduzione italiana."""
        from voicemem.lang import display_emotion
        from voicemem import lang
        lang._override = "it"
        assert display_emotion("anxious") == "ansioso"
        assert display_emotion("happy") == "felice"
        lang._override = None

    def test_display_emotion_english(self):
        """display_emotion() restituisce traduzione inglese."""
        from voicemem.lang import display_emotion
        from voicemem import lang
        lang._override = "en"
        assert display_emotion("ansioso") == "anxious"
        assert display_emotion("felice") == "happy"
        lang._override = None
