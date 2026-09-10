"""Xiaozhi WebSocket transport for VoiceMem.

Xiaozhi sends Opus frames and receives JSON control messages plus Opus frames.
The VoiceMem pipeline itself continues to consume and produce PCM16 mono audio.
"""
from __future__ import annotations

import json
import struct
import time
import uuid

import av
import numpy as np
from fastapi import WebSocket


INPUT_RATE = 16000
OUTPUT_RATE = 24000
FRAME_SAMPLES = int(OUTPUT_RATE * 0.06)


class XiaozhiAudio:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.decoder = av.CodecContext.create("opus", "r")
        self.decoder.sample_rate = INPUT_RATE
        self.decoder.layout = "mono"
        self.resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=OUTPUT_RATE)
        self.encoder = av.CodecContext.create("libopus", "w")
        self.encoder.sample_rate = OUTPUT_RATE
        self.encoder.layout = "mono"
        self.encoder.format = "s16"
        self.packet_sink = None
        self.input_pcm = bytearray()
        self.output_pcm = bytearray()

    def decode(self, packet: bytes) -> bytes:
        decoded = bytearray()
        for frame in self.decoder.decode(av.Packet(packet)):
            for resampled in self.resampler.resample(frame):
                decoded.extend(resampled.to_ndarray().reshape(-1).tobytes())
        return bytes(decoded)

    async def encode_and_send(self, pcm: bytes, flush: bool = False) -> None:
        self.output_pcm.extend(pcm)
        frame_bytes = FRAME_SAMPLES * 2
        while len(self.output_pcm) >= frame_bytes:
            await self._encode_frame(bytes(self.output_pcm[:frame_bytes]))
            del self.output_pcm[:frame_bytes]
        if flush and self.output_pcm:
            padded = bytes(self.output_pcm) + b"\x00" * (frame_bytes - len(self.output_pcm))
            await self._encode_frame(padded)
            self.output_pcm.clear()

    async def _encode_frame(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = OUTPUT_RATE
        for packet in self.encoder.encode(frame):
            payload = bytes(packet)
            if self.packet_sink is not None:
                await self.packet_sink(payload)
            else:
                await self.websocket.send_bytes(payload)


class XiaozhiTransport:
    def __init__(self, websocket: WebSocket, audio: XiaozhiAudio, session_id: str, mqtt_gateway: bool = False):
        self.websocket = websocket
        self.audio = audio
        self.session_id = session_id
        self.mqtt_gateway = mqtt_gateway
        self.audio_sequence = 0
        self.started_sentence = False
        self.reply_text = ""
        self.input_sink = None
        if mqtt_gateway:
            audio.packet_sink = self._send_gateway_audio

    async def receive(self):
        message = await self.websocket.receive()
        if message.get("bytes") is not None:
            payload = message["bytes"]
            if self.mqtt_gateway:
                if len(payload) < 16:
                    return {"bytes": b""}
                payload_length = struct.unpack_from(">I", payload, 12)[0]
                payload = payload[16:16 + payload_length]
            pcm = self.audio.decode(payload)
            if self.input_sink is not None and pcm:
                await self.input_sink(pcm)
            return {"bytes": pcm}
        return message

    async def _send_gateway_audio(self, payload: bytes) -> None:
        self.audio_sequence += 1
        timestamp = int(time.time() * 1000) & 0xFFFFFFFF
        header = b"\x00" * 8 + struct.pack(">II", timestamp, len(payload))
        await self.websocket.send_bytes(header + payload)

    async def send_json(self, message: dict) -> None:
        message_type = message.get("type")
        if message_type == "user_transcript":
            await self.websocket.send_json({"type": "stt", "text": message.get("text", ""), "session_id": self.session_id})
        elif message_type == "answer_start":
            self.reply_text = ""
            self.started_sentence = False
            await self.websocket.send_json({"type": "tts", "state": "start", "session_id": self.session_id})
        elif message_type == "answer_delta":
            self.reply_text += message.get("text", "")
        elif message_type == "answer_interrupt":
            await self.audio.encode_and_send(b"", flush=True)
            await self.websocket.send_json({"type": "tts", "state": "stop", "session_id": self.session_id})
        elif message_type == "answer_done":
            await self.audio.encode_and_send(b"", flush=True)
            await self.websocket.send_json({"type": "tts", "state": "stop", "session_id": self.session_id})

    async def send_audio(self, pcm: bytes) -> None:
        if not self.started_sentence:
            self.started_sentence = True
            await self.websocket.send_json({"type": "tts", "state": "sentence_start", "text": self.reply_text, "session_id": self.session_id})
        await self.audio.encode_and_send(pcm)


async def read_hello(websocket: WebSocket, session_id: str) -> dict:
    message = await websocket.receive_json()
    if message.get("type") != "hello":
        raise ValueError("Xiaozhi hello mancante")
    await websocket.send_json({
        "type": "hello",
        "version": 1,
        "transport": "websocket",
        "session_id": session_id,
        "audio_params": message.get("audio_params") or {
            "format": "opus", "sample_rate": OUTPUT_RATE, "channels": 1, "frame_duration": 60,
        },
    })
    return message
