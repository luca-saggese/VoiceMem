import asyncio

import numpy as np

from voicemem.tts_cosyvoice import CosyVoice3TTS, prepare_instruction


def test_prepare_instruction_adds_language_and_delimiter():
    value = prepare_instruction("Parla lentamente", "it")
    assert "Speak in Italian" in value
    assert value.endswith("<|endofprompt|>")


def test_cosyvoice_pcm_conversion_rejects_nonfinite():
    class Tensor:
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return np.array([np.nan], dtype=np.float32)

    try:
        from voicemem.tts_cosyvoice import _to_pcm16
        _to_pcm16({"tts_speech": Tensor()})
    except RuntimeError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite audio was not rejected")


def test_cosyvoice_stream_uses_worker_and_pcm():
    class FakeModel:
        def inference_instruct2(self, **kwargs):
            assert kwargs["stream"] is True
            yield {"tts_speech": np.array([0.0, 0.5], dtype=np.float32)}

    async def run():
        provider = CosyVoice3TTS(model="unused", ref_audio=__file__)
        provider._model = FakeModel()
        chunks = [chunk async for chunk in provider.stream("ciao")]
        assert chunks == [(0).to_bytes(2, "little", signed=True) + (16383).to_bytes(2, "little", signed=True)]

    asyncio.run(run())


def test_cosyvoice_stream_cancellation_does_not_block_event_loop():
    """La cancellazione del consumer non deve lasciare una put infinita."""
    import time

    class SlowModel:
        def inference_instruct2(self, **kwargs):
            for _ in range(100):
                time.sleep(0.001)
                yield {"tts_speech": np.zeros(240, dtype=np.float32)}

    async def run():
        provider = CosyVoice3TTS(model="unused", ref_audio=__file__)
        provider._model = SlowModel()
        stream = provider.stream("ciao")
        task = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.05)

    asyncio.run(run())
