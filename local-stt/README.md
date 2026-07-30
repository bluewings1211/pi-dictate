# Local STT service

This is a persistent, loopback-only companion service for the Pi extension. It
keeps Qwen3-ASR-0.6B loaded between dictations; raw microphone audio never
leaves `127.0.0.1`.

## Run a local model on Apple Silicon

Use isolated Python environments for the two model families, then start one
persistent service. The Pi extension does not need to change when switching
models; stop the running service and launch the other profile.

```bash
cd local-stt

# Qwen: faster / lower memory
./start.sh qwen-mlx

# Breeze-ASR-25: higher accuracy for Taiwan Mandarin + Chinese-English mixing
./start.sh breeze
```

`start.sh qwen-mlx` uses `.venv`; `start.sh breeze` uses `.venv-breeze`. If an
environment does not exist yet, create it and install the matching requirements:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-qwen.txt

python3.12 -m venv .venv-breeze
.venv-breeze/bin/pip install -r requirements-breeze.txt
```

Qwen defaults to `models/Qwen3-ASR-0.6B`; Breeze defaults to
`models/Breeze-ASR-25`. Pass `--model <path-or-HF-id>` to `server.py` to
override either model. Confirm the active backend:

```bash
curl http://127.0.0.1:8765/health
```

Start Pi with:

```bash
export DICTATE_BACKEND=local
export LOCAL_STT_URL=http://127.0.0.1:8765
pi
```

The default remains `DICTATE_BACKEND=deepgram`. Local mode has no fallback to
Deepgram and does not need `DEEPGRAM_API_KEY`.

## Benchmark Qwen vs. Breeze-ASR-25

Create a fixed, public Taiwan Mandarin-English sample set from the ML2021 test
split, then use the same manifest for every model:

```bash
# In the Breeze environment; downloads only the selected clips.
python prepare_ml2021_benchmark.py --limit 8

# Qwen
.venv/bin/python benchmark.py benchmark-data/ml2021-mixed-8/manifest.json \
  --backend qwen-mlx --output benchmark-data/ml2021-mixed-8/qwen.json

# Breeze-ASR-25 / Twister (separate environment/dependencies)
pip install -r requirements-breeze.txt
python benchmark.py benchmark-data/ml2021-mixed-8/manifest.json \
  --backend breeze --output benchmark-data/ml2021-mixed-8/breeze.json
```

Each report contains every transcript, per-clip character error rate, inference
latency, model name, Python version, and machine identifier. Do not compare raw
CER across different manifests. The supplied normalizer is deliberately simple;
keep it unchanged for a given experiment.

The benchmark uses `local-stt/models/Breeze-ASR-25` by default; pass `--model`
or set `BREEZE_ASR_MODEL` to use a different checkpoint.

## Limits

- The live server supports Qwen MLX only in this revision. Breeze is benchmark
  only until its Transformers/MPS route is measured for latency, memory use, and
  transcription quality on the target Mac.
- The service accepts only 16 kHz mono signed-16-bit PCM from the extension and
  binds only to loopback addresses.
- Local transcription is final-only. The extension preserves its current UX:
  recording meter while speaking, then one final insertion after stop.
