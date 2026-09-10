"""Embedded Xiaozhi MQTT+UDP broker and WebSocket bridge.

The device speaks MQTT 3.1.1 for control and sends encrypted Opus datagrams
through UDP. The application conversation remains on the existing Xiaozhi
WebSocket endpoint, using the same framing as xiaozhi-mqtt-gateway.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import random
import struct
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import websockets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


HEADER_SIZE = 16


def _read_u16(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from(">H", data, offset)[0], offset + 2


def _read_utf(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _read_u16(data, offset)
    end = offset + length
    return data[offset:end].decode("utf-8"), end


def _mqtt_packet(packet_type: int, payload: bytes, flags: int = 0) -> bytes:
    remaining = len(payload)
    encoded = bytearray()
    while True:
        digit = remaining % 128
        remaining //= 128
        if remaining:
            digit |= 0x80
        encoded.append(digit)
        if not remaining:
            break
    return bytes([(packet_type << 4) | flags]) + bytes(encoded) + payload


def _mqtt_publish(topic: str, payload: bytes) -> bytes:
    topic_bytes = topic.encode()
    return _mqtt_packet(3, struct.pack(">H", len(topic_bytes)) + topic_bytes + payload)


def _crypt(payload: bytes, key: bytes, header: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.CTR(header))
    return cipher.encryptor().update(payload)


@dataclass
class MqttDevice:
    broker: "XiaozhiMqttBroker"
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    client_id: str
    device_id: str
    username: str
    registered: bool = True
    connection_id: int = field(default_factory=lambda: random.randrange(1, 0xFFFFFFFF))
    bridge: Optional[websockets.WebSocketClientProtocol] = None
    bridge_task: Optional[asyncio.Task] = None
    udp_key: bytes = b""
    udp_nonce: bytes = b""
    udp_remote: Optional[tuple[str, int]] = None
    udp_sequence: int = 0
    last_device_sequence: int = -1
    device_sequence: int = 0
    closed: bool = False

    async def send_json(self, message: dict) -> None:
        await self.send_mqtt(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode())

    async def send_mqtt(self, payload: bytes) -> None:
        if not self.closed:
            self.writer.write(_mqtt_publish(f"devices/p2p/{self.device_id.replace(':', '_')}", payload))
            await self.writer.drain()

    async def start_bridge(self, hello: dict) -> None:
        if self.bridge_task and not self.bridge_task.done():
            return
        self.bridge_task = asyncio.create_task(self._bridge(hello))

    async def _bridge(self, hello: dict) -> None:
        self.udp_key = os.urandom(16)
        self.udp_nonce = self._udp_header(0, 0, 0)
        ws_url = self.broker.websocket_url
        token = os.environ.get("VOICEMEM_XIAOZHI_DEVICE_TOKEN", "")
        headers = {
            "device-id": self.device_id,
            "client-id": self.client_id,
            "authorization": f"Bearer {token}" if token else "",
        }
        try:
            try:
                websocket = await websockets.connect(ws_url, additional_headers=headers)
            except TypeError:
                websocket = await websockets.connect(ws_url, extra_headers=headers)
            async with websocket as ws:
                self.bridge = ws
                await ws.send(json.dumps(hello, ensure_ascii=False))
                welcome = json.loads(await ws.recv())
                await self.send_json({
                    "type": "hello",
                    "version": hello.get("version", 3),
                    "session_id": welcome.get("session_id", uuid.uuid4().hex),
                    "transport": "udp",
                    "udp": {
                        "server": self.broker.public_host,
                        "port": self.broker.udp_port,
                        "encryption": "aes-128-ctr",
                        "key": self.udp_key.hex(),
                        "nonce": self.udp_nonce.hex(),
                    },
                    "audio_params": welcome.get("audio_params", hello.get("audio_params", {})),
                })
                async for message in ws:
                    if isinstance(message, str):
                        await self.send_mqtt(message.encode())
                    else:
                        await self.send_udp_audio(message)
        except Exception as exc:
            if not self.closed:
                await self.send_json({"type": "error", "message": f"MQTT bridge error: {exc}"})
        finally:
            self.bridge = None
            if not self.closed:
                await self.send_json({"type": "goodbye"})

    async def handle_json(self, message: dict) -> None:
        if not self.registered:
            await self.send_json({
                "type": "error",
                "code": "device_not_registered",
                "message": f"Device non registrato. Registra questo ID: {self.device_id}",
                "device_id": self.device_id,
            })
            print(
                f"[xiaozhi][mqtt] device_id={self.device_id} non registrato; "
                "inviato ID al device senza avviare il bridge",
                flush=True,
            )
            return
        if message.get("type") == "hello":
            await self.start_bridge(message)
            return
        if message.get("type") == "goodbye":
            await self.close_bridge()
            return
        if self.bridge is not None:
            await self.bridge.send(json.dumps(message, ensure_ascii=False))

    async def send_udp_audio(self, websocket_frame: bytes) -> None:
        if len(websocket_frame) < HEADER_SIZE or not self.udp_remote:
            return
        timestamp = struct.unpack_from(">I", websocket_frame, 8)[0]
        payload_length = struct.unpack_from(">I", websocket_frame, 12)[0]
        payload = websocket_frame[16:16 + payload_length]
        self.udp_sequence += 1
        header = self._udp_header(len(payload), timestamp, self.udp_sequence)
        self.broker.udp.sendto(header + _crypt(payload, self.udp_key, header), self.udp_remote)

    def _udp_header(self, length: int, timestamp: int, sequence: int) -> bytes:
        return struct.pack(">BBHIII", 1, 0, length, self.connection_id, timestamp & 0xFFFFFFFF, sequence & 0xFFFFFFFF)

    async def close_bridge(self) -> None:
        bridge = self.bridge
        self.bridge = None
        if bridge is not None:
            await bridge.close()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.close_bridge()
        if self.bridge_task and not self.bridge_task.done():
            self.bridge_task.cancel()
        self.writer.close()
        await self.writer.wait_closed()


class XiaozhiMqttBroker:
    def __init__(self, websocket_url: str, authenticate_device: Optional[Callable[[str], object]] = None):
        self.host = os.environ.get("VOICEMEM_XIAOZHI_MQTT_HOST", "0.0.0.0")
        self.port = int(os.environ.get("VOICEMEM_XIAOZHI_MQTT_PORT", "1883"))
        self.udp_host = os.environ.get("VOICEMEM_XIAOZHI_UDP_HOST", "0.0.0.0")
        self.udp_port = int(os.environ.get("VOICEMEM_XIAOZHI_UDP_PORT", "8884"))
        self.public_host = os.environ.get("VOICEMEM_PUBLIC_IP", "127.0.0.1")
        self.websocket_url = websocket_url
        self.authenticate_device = authenticate_device
        self.signature_key = os.environ.get("VOICEMEM_XIAOZHI_MQTT_SIGNATURE_KEY", os.environ.get("MQTT_SIGNATURE_KEY", ""))
        self.server: asyncio.AbstractServer | None = None
        self.udp: asyncio.DatagramTransport | None = None
        self.devices: dict[int, MqttDevice] = {}

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            self.udp, _ = await loop.create_datagram_endpoint(
                lambda: _UdpProtocol(self), local_addr=(self.udp_host, self.udp_port)
            )
            print(f"[xiaozhi][mqtt] UDP listener attivo su {self.udp_host}:{self.udp_port}", flush=True)
            self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
            print(f"[xiaozhi][mqtt] TCP listener attivo su {self.host}:{self.port}", flush=True)
            print(f"[xiaozhi][mqtt] WebSocket bridge: {self.websocket_url}", flush=True)
        except Exception:
            print(f"[xiaozhi][mqtt] ERRORE avvio listener TCP={self.host}:{self.port} UDP={self.udp_host}:{self.udp_port}", flush=True)
            raise

    async def stop(self) -> None:
        for device in list(self.devices.values()):
            await device.close()
        self.devices.clear()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        if self.udp:
            self.udp.close()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        device = None
        peer = writer.get_extra_info("peername")
        print(f"[xiaozhi][mqtt] connessione TCP da {peer}", flush=True)
        try:
            packet_type, payload = await self._read_packet(reader)
            print(f"[xiaozhi][mqtt] {peer} primo pacchetto type={packet_type} bytes={len(payload)}", flush=True)
            if packet_type != 1:
                print(f"[xiaozhi][mqtt] {peer} rifiutato: atteso CONNECT, ricevuto type={packet_type}", flush=True)
                return
            client_id, username, password = self._parse_connect(payload)
            print(
                f"[xiaozhi][mqtt] {peer} CONNECT client_id={client_id!r} "
                f"username_present={bool(username)} password_present={bool(password)}",
                flush=True,
            )
            device_id = self._authenticate(client_id, username, password)
            print(f"[xiaozhi][mqtt] {peer} firma valida device_id={device_id}", flush=True)
            registered = bool(self.authenticate_device(device_id)) if self.authenticate_device else True
            if not registered:
                print(
                    f"[xiaozhi][mqtt] {peer} device non registrato: {device_id}; "
                    "CONNACK consentito per comunicare l'ID",
                    flush=True,
                )
            writer.write(_mqtt_packet(2, b"\x00\x00"))
            await writer.drain()
            print(f"[xiaozhi][mqtt] {peer} CONNACK inviato device_id={device_id}", flush=True)
            device = MqttDevice(self, reader, writer, client_id, device_id, username, registered=registered)
            for existing in list(self.devices.values()):
                if existing.device_id == device_id:
                    await existing.close()
            self.devices[device.connection_id] = device
            while not reader.at_eof():
                packet_type, payload = await self._read_packet(reader)
                print(f"[xiaozhi][mqtt] {peer} packet type={packet_type} bytes={len(payload)} device={device_id}", flush=True)
                if packet_type == 3:
                    topic, body = self._parse_publish(payload)
                    print(f"[xiaozhi][mqtt] {peer} PUBLISH topic={topic!r} bytes={len(body)}", flush=True)
                    if topic == "device-server":
                        await device.handle_json(json.loads(body))
                elif packet_type == 8:
                    packet_id = struct.unpack_from(">H", payload, 0)[0]
                    writer.write(_mqtt_packet(9, struct.pack(">HB", packet_id, 0)))
                    await writer.drain()
                elif packet_type == 12:
                    writer.write(b"\xD0\x00")
                    await writer.drain()
                elif packet_type == 14:
                    break
        except Exception as exc:
            print(f"[xiaozhi][mqtt] {peer} chiuso/errore: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if device:
                self.devices.pop(device.connection_id, None)
                await device.close()
            else:
                writer.close()
                await writer.wait_closed()
            print(f"[xiaozhi][mqtt] connessione TCP chiusa da {peer}", flush=True)

    async def _read_packet(self, reader: asyncio.StreamReader) -> tuple[int, bytes]:
        first = (await reader.readexactly(1))[0]
        remaining = 0
        multiplier = 1
        while True:
            digit = (await reader.readexactly(1))[0]
            remaining += (digit & 127) * multiplier
            if not digit & 128:
                break
            multiplier *= 128
            if multiplier > 128 * 128 * 128:
                raise ValueError("MQTT remaining length non valido")
        return first >> 4, await reader.readexactly(remaining)

    @staticmethod
    def _parse_connect(payload: bytes) -> tuple[str, str, str]:
        offset = 0
        protocol, offset = _read_utf(payload, offset)
        if protocol != "MQTT":
            raise ValueError("MQTT protocollo non supportato")
        level = payload[offset]
        flags = payload[offset + 1]
        offset += 4
        if level == 5:
            properties_length = payload[offset]
            offset += 1 + properties_length
        client_id, offset = _read_utf(payload, offset)
        username = ""
        password = ""
        if flags & 0x80:
            username, offset = _read_utf(payload, offset)
        if flags & 0x40:
            length, offset = _read_u16(payload, offset)
            password = payload[offset:offset + length].decode()
        if level not in (4, 5):
            raise ValueError("MQTT versione non supportata")
        return client_id, username, password

    @staticmethod
    def _parse_publish(payload: bytes) -> tuple[str, bytes]:
        topic, offset = _read_utf(payload, 0)
        return topic, payload[offset:]

    def _authenticate(self, client_id: str, username: str, password: str) -> str:
        parts = client_id.split("@@@")
        if len(parts) != 3:
            raise ValueError("client_id Xiaozhi non valido")
        if self.signature_key:
            expected = base64.b64encode(hmac.new(self.signature_key.encode(), f"{client_id}|{username}".encode(), hashlib.sha256).digest()).decode()
            if not hmac.compare_digest(expected, password):
                raise ValueError("firma MQTT non valida")
        mac = parts[1].replace("_", ":").lower()
        if len(mac) != 17 or mac.count(":") != 5:
            raise ValueError("device MAC non valido")
        return mac


class _UdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, broker: XiaozhiMqttBroker):
        self.broker = broker

    def datagram_received(self, data: bytes, addr) -> None:
        if len(data) < HEADER_SIZE:
            return
        try:
            kind, _, length, connection_id, timestamp, sequence = struct.unpack_from(">BBHIII", data, 0)
            if kind != 1 or len(data) != HEADER_SIZE + length:
                return
            device = self.broker.devices.get(connection_id)
            if not device or not device.bridge or not device.udp_key:
                return
            if sequence <= device.last_device_sequence:
                return
            device.last_device_sequence = sequence
            device.udp_remote = addr
            payload = _crypt(data[HEADER_SIZE:HEADER_SIZE + length], device.udp_key, data[:HEADER_SIZE])
            frame = b"\x00" * 8 + struct.pack(">II", timestamp, len(payload)) + payload
            asyncio.create_task(device.bridge.send(frame))
        except Exception:
            return
