"""Unit tests per NemotronStreamingASR — STEP 1."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Assicura che voicemem sia importabile dal venv
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════ Helpers ═══════════════════

def _make_mock_sherpa_module():
    """Crea un modulo sherpa_onnx completamente mockato."""
    mock_sherpa = MagicMock()
    mock_rec_class = MagicMock()
    mock_rec_instance = MagicMock()
    mock_rec_instance.is_ready.return_value = False
    mock_rec_instance.get_result.return_value = MagicMock(text="")
    # Ogni chiamata a create_stream deve restituire un nuovo stream mock
    mock_rec_instance.create_stream.side_effect = lambda: MagicMock()
    mock_rec_class.from_transducer.return_value = mock_rec_instance
    mock_sherpa.OnlineRecognizer = mock_rec_class
    return mock_sherpa


def _create_model_dir(tmp_path: Path) -> Path:
    """Crea una directory modello fake con tutti i file necessari."""
    model_dir = tmp_path / "nemotron-asr"
    model_dir.mkdir()
    (model_dir / "encoder.int8.onnx").write_bytes(b"fake")
    (model_dir / "decoder.int8.onnx").write_bytes(b"fake")
    (model_dir / "joiner.int8.onnx").write_bytes(b"fake")
    (model_dir / "tokens.txt").write_text("test\n1\n")
    return model_dir


# ═══════════════════ Language validation ═══════════════════

class TestNemotronLanguageValidation:
    def test_supported_languages_it(self, tmp_path):
        """it è una lingua supportata."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir, language="it")
            assert asr.language == "it"

    def test_supported_languages_en(self, tmp_path):
        """en è una lingua supportata."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir, language="en")
            assert asr.language == "en"

    def test_unsupported_language_raises(self, tmp_path):
        """zh deve sollevare ValueError."""
        model_dir = _create_model_dir(tmp_path)
        from voicemem.utils.audio.asr import NemotronStreamingASR
        with pytest.raises(ValueError, match="Lingua non supportata"):
            NemotronStreamingASR(model_dir, language="zh")

    def test_default_language_is_italian(self, tmp_path):
        """Il default della lingua è 'it'."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir)
            assert asr.language == "it"

    def test_all_supported_languages(self):
        """SUPPORTED_LANGUAGES contiene solo it ed en."""
        from voicemem.utils.audio.asr import NemotronStreamingASR
        assert NemotronStreamingASR.SUPPORTED_LANGUAGES == ("it", "en")


# ═══════════════════ Missing model file ═══════════════════

class TestMissingModelFile:
    def test_missing_encoder_raises(self, tmp_path):
        """Mancanza encoder.int8.onnx → FileNotFoundError."""
        model_dir = tmp_path / "incomplete"
        model_dir.mkdir()
        (model_dir / "decoder.int8.onnx").write_bytes(b"fake")
        (model_dir / "joiner.int8.onnx").write_bytes(b"fake")
        (model_dir / "tokens.txt").write_text("test\n1\n")
        from voicemem.utils.audio.asr import NemotronStreamingASR
        with pytest.raises(FileNotFoundError, match="encoder.int8.onnx"):
            NemotronStreamingASR(model_dir, language="it")

    def test_missing_tokens_raises(self, tmp_path):
        """Mancanza tokens.txt → FileNotFoundError."""
        model_dir = tmp_path / "incomplete2"
        model_dir.mkdir()
        (model_dir / "encoder.int8.onnx").write_bytes(b"fake")
        (model_dir / "decoder.int8.onnx").write_bytes(b"fake")
        (model_dir / "joiner.int8.onnx").write_bytes(b"fake")
        from voicemem.utils.audio.asr import NemotronStreamingASR
        with pytest.raises(FileNotFoundError, match="tokens.txt"):
            NemotronStreamingASR(model_dir, language="it")


# ═══════════════════ Reset creates clean stream ═══════════════════

class TestResetCreatesCleanStream:
    def test_reset_clears_text(self, tmp_path):
        """reset() pulisce il testo accumulato."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir, language="it")
            asr._text = "ciao mondo"
            asr.reset()
            assert asr._text == ""

    def test_reset_creates_new_stream(self, tmp_path):
        """reset() ricrea un nuovo OnlineStream."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir, language="it")
            initial_stream = asr._stream
            asr.reset()
            # Il nuovo stream deve essere diverso dall'originale
            assert asr._stream is not initial_stream
            # set_option("language") deve essere chiamato sul nuovo stream
            asr._stream.set_option.assert_called_with("language", "it")

    def test_reset_does_not_keep_previous_text(self, tmp_path):
        """reset() non mantiene testo precedente."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir, language="en")
            asr._text = "previous text"
            asr.reset()
            assert asr._text == ""

    def test_feed_returns_str(self, tmp_path):
        """feed() restituisce sempre str."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir, language="it")
            samples = np.zeros(16000, dtype=np.float32)
            result = asr.feed(samples)
            assert isinstance(result, str)

    def test_flush_returns_str(self, tmp_path):
        """flush() restituisce sempre str."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir, language="it")
            result = asr.flush()
            assert isinstance(result, str)


# ═══════════════════ Stream language option ═══════════════════

class TestStreamLanguageOption:
    def test_stream_set_option_called_on_feed(self, tmp_path):
        """feed() applica la lingua allo stream."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir, language="it")
            asr._stream.set_option.reset_mock()
            asr.feed(np.zeros(16000, dtype=np.float32))
            asr._stream.set_option.assert_called_with("language", "it")

    def test_stream_set_option_called_on_flush(self, tmp_path):
        """flush() ricrea lo stream e applica la lingua."""
        mock_sherpa = _make_mock_sherpa_module()
        model_dir = _create_model_dir(tmp_path)
        with patch.dict(sys.modules, {"sherpa_onnx": mock_sherpa}):
            from voicemem.utils.audio.asr import NemotronStreamingASR
            asr = NemotronStreamingASR(model_dir, language="en")
            initial_stream = asr._stream
            asr.flush()
            assert asr._stream is not initial_stream
            asr._stream.set_option.assert_called_with("language", "en")
