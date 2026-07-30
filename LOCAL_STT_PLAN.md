# Local STT implementation plan

## Goal

Allow this Pi/omp dictation extension to choose between the existing Deepgram
streaming backend and a privacy-preserving STT service running on the same
machine. The initial local model is Qwen3-ASR-0.6B. Breeze-ASR-25 (Twister) is
added as a benchmark candidate for Taiwanese Mandarin and Mandarin-English
code-switching, not assumed to be the default model.

## Scope and acceptance criteria

1. `DICTATE_BACKEND=deepgram|local` selects the recognizer. The default remains
   `deepgram`, so existing installations do not change behaviour.
2. In `local` mode, recording must not require `DEEPGRAM_API_KEY` and audio must
   only be sent to a loopback (`127.0.0.1`/`localhost`) service.
3. The extension continues to use `rec` for 16 kHz mono PCM and retains the
   microphone-level meter, focus-aware delivery, cancel behaviour, and final
   one-shot editor insertion.
4. `local-stt/server.py` is a long-lived, loopback-only HTTP service. It loads
   the selected Qwen or Breeze model once, exposes a health endpoint, and
   accepts raw signed-16-bit PCM.
5. Benchmarking must use the same input clips and normalized output for Qwen,
   Breeze-ASR-25, and optionally Deepgram. It must report recognition accuracy
   and stop-to-text latency without requiring any cloud upload.

## Architecture

```text
Pi extension
  rec (16 kHz, mono, s16le) ──> level meter
                              ├─ Deepgram WebSocket (existing backend)
                              └─ local HTTP POST on stop (local backend)
                                      │ raw PCM, never leaves loopback
                                      ▼
                              local-stt/server.py
                              ├─ Qwen3-ASR-0.6B via MLX (default)
                              └─ pluggable benchmark runners
                                     ├─ Qwen
                                     └─ Breeze-ASR-25
```

The first local version is deliberately **final-only**. The extension currently
does not render revisable partial text, so it gains privacy without changing its
editing UX. A later streaming mode can use the same service boundary and publish
partial text into the status row only.

## Implementation steps

1. Add documented environment settings and a `RecognizerBackend` branch in
   `index.ts`. Preserve the Deepgram path unchanged where possible.
2. While local recording is active, retain PCM chunks. Once `rec` closes, POST
   them to `LOCAL_STT_URL/v1/transcriptions` with explicit audio-format headers.
   Abort the request on cancel and never fall back to a remote service silently.
3. Add a Python local service that accepts only `application/octet-stream`,
   converts PCM to a temporary WAV file, and serializes requests with a model
   lock. The default Qwen adapter uses `mlx-qwen3-asr` and retains a loaded
   `Session` for repeated dictations.
4. Add `local-stt/requirements-qwen.txt`, a launch script, and setup guidance.
   The service binds to `127.0.0.1`, not all interfaces.
5. Add benchmark runners and a manifest-driven CLI. Benchmark clips and their
   ground truth stay outside git by default; the committed manifest is a
   template. Measure character error rate for Chinese/mixed text, normalized
   English-token error rate, and latency. Record machine/model/version settings.
6. Verify TypeScript compilation and Python syntax. Model-download/inference
   verification is a separate local step because model packages and weights are
   intentionally not bundled with this extension.

## Model decisions

- **Default local adapter: Qwen3-ASR-0.6B through MLX.** This is the smallest
  Qwen release and has an Apple-Silicon-oriented runtime. It is the practical
  path for this Mac, but the MLX implementation is a separate dependency.
- **Quality profile: Breeze-ASR-25 / Twister.** It is fine-tuned from
  Whisper-large-v2 for Taiwanese Mandarin and Mandarin-English code-switching.
  It runs in its own PyTorch/MPS environment and is selected with
  `local-stt/start.sh breeze`.
- **Not in scope yet: partial transcript UI.** Qwen's official incremental
  vLLM route is CUDA-oriented. It can later be exposed as `/v1/stream/*` without
  changing the extension's backend selection or final-delivery behaviour.

## Risks and mitigations

- Model startup is slow: keep the service persistent and lazy-load once.
- Large or concurrent recordings: serialize inference and return HTTP 503 if
  busy rather than competing for unified memory.
- Apple-Silicon package compatibility can change: `/health` reports an explicit
  dependency/model-load failure; the extension surfaces it to the user.
- Benchmark scores are only comparable with the same clips and normalizer:
  version the manifest and write the exact command/machine information into each
  JSON result.
