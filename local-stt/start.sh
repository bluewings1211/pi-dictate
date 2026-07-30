#!/usr/bin/env bash
# Start exactly one persistent local ASR model. The Pi extension always talks
# to the same localhost API; switch models by restarting this launcher.
set -euo pipefail

service_dir="$(cd "$(dirname "$0")" && pwd)"
backend="${1:-qwen-mlx}"
port="${LOCAL_STT_PORT:-8765}"

case "$backend" in
  qwen-mlx)
    python_bin="$service_dir/.venv/bin/python"
    ;;
  breeze)
    python_bin="$service_dir/.venv-breeze/bin/python"
    ;;
  *)
    echo "Usage: $0 [qwen-mlx|breeze]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$python_bin" ]]; then
  echo "Missing runtime: $python_bin" >&2
  exit 1
fi

exec "$python_bin" "$service_dir/server.py" --backend "$backend" --port "$port"
