#!/usr/bin/env python3
"""Loopback-only local transcription service for pi-dictate.

The process intentionally keeps the selected ASR model resident after its first
request. It accepts raw 16-bit little-endian PCM so the TypeScript extension can
keep its existing `rec` pipeline without format conversion dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import threading
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from backends import DEFAULT_BREEZE_MODEL, DEFAULT_QWEN_MODEL, Transcriber, create_transcriber

MAX_AUDIO_BYTES = 64 * 1024 * 1024


class ServiceState:
    def __init__(self, backend: str, model: Optional[str]) -> None:
        self.backend = backend
        self.model = model
        self.transcriber: Optional[Transcriber] = None
        self.load_error: Optional[str] = None
        self.load_lock = threading.Lock()
        self.inference_lock = threading.Lock()

    @property
    def effective_model(self) -> str:
        if self.model:
            return self.model
        return DEFAULT_BREEZE_MODEL if self.backend == "breeze" else DEFAULT_QWEN_MODEL

    def get_transcriber(self) -> Transcriber:
        with self.load_lock:
            if self.transcriber is not None:
                return self.transcriber
            if self.load_error is not None:
                raise RuntimeError(self.load_error)
            try:
                self.transcriber = create_transcriber(self.backend, self.model)
            except Exception as exc:
                self.load_error = str(exc)
                raise RuntimeError(self.load_error) from exc
            return self.transcriber


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix="pi-dictate-", suffix=".wav")
    os.close(fd)
    wav_path = Path(raw_path)
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return wav_path


class Handler(BaseHTTPRequestHandler):
    server: "DictateServer"

    def log_message(self, fmt: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        state = self.server.state
        self.send_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "backend": state.backend,
                "model": state.effective_model,
                "loaded": state.transcriber is not None,
                "load_error": state.load_error,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/v1/transcriptions":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if self.headers.get_content_type() != "application/octet-stream":
            self.send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "expected application/octet-stream"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if not 0 < content_length <= MAX_AUDIO_BYTES:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid or oversized audio body"})
            return
        if self.headers.get("X-Audio-Format") != "s16le":
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "X-Audio-Format must be s16le"})
            return
        try:
            sample_rate = int(self.headers.get("X-Sample-Rate", "16000"))
            channels = int(self.headers.get("X-Channels", "1"))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid audio headers"})
            return
        if sample_rate != 16000 or channels != 1:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "only 16 kHz mono audio is supported"})
            return

        pcm = self.rfile.read(content_length)
        wav_path: Optional[Path] = None
        try:
            wav_path = pcm_to_wav(pcm, sample_rate, channels)
            state = self.server.state
            with state.inference_lock:
                transcriber = state.get_transcriber()
                text = transcriber.transcribe(wav_path)
            self.send_json(HTTPStatus.OK, {"text": text, "model": transcriber.name})
        except Exception as exc:
            logging.exception("transcription failed")
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
        finally:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)


class DictateServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: ServiceState) -> None:
        super().__init__(address, Handler)
        self.state = state


def main() -> None:
    parser = argparse.ArgumentParser(description="Local STT service for pi-dictate")
    parser.add_argument("--host", default="127.0.0.1", help="loopback address only (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--backend",
        choices=("qwen-mlx", "breeze"),
        default="qwen-mlx",
        help="qwen-mlx for lower latency, breeze for Taiwan Mandarin/code-switch quality",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("for privacy this service may bind only to a loopback address")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = DictateServer((args.host, args.port), ServiceState(args.backend, args.model))
    logging.info("listening at http://%s:%d (backend %s)", args.host, args.port, args.backend)
    server.serve_forever()


if __name__ == "__main__":
    main()
