#!/usr/bin/env bash
# Start exactly one persistent local ASR model. The Pi extension always talks
# to the same localhost API; switch models by restarting this launcher.
#
# Idempotent: if a local-stt server is already answering on LOCAL_STT_PORT,
# this exits without launching a second process. A server already running a
# different backend is an error — stop it first to switch models.
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

# Source of truth for "already running" is our /health endpoint, which
# answers with the active backend. curl exit 0 + a parseable backend means a
# local-stt server owns the port; exit 7 (connection refused) means none is
# running and we proceed. A port that answers without a recognizable health
# payload is treated as foreign and we refuse to double-bind.
health="http://127.0.0.1:${port}/health"
if resp="$(curl --silent --max-time 2 "$health" 2>/dev/null)"; then
  running="$(printf '%s' "$resp" | sed -n 's/.*"backend"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  if [[ -n "$running" && "$running" == "$backend" ]]; then
    echo "local-stt already running (backend=$backend port=$port); nothing to start."
    exit 0
  elif [[ -n "$running" ]]; then
    echo "local-stt already running on port $port with backend '$running' (requested: '$backend')." >&2
    echo "Model switching works by restarting the launcher; stop the current server first (e.g. pkill -f 'server.py.*--backend')." >&2
    exit 1
  else
    echo "Port $port answers, but not with a local-stt health payload; refusing to start a second server." >&2
    exit 1
  fi
fi

# Launch detached with output redirected to a log file so server logs never
# paint over the caller's terminal, and return only once /health confirms the
# service is serving — matching the prompt-exit behavior of the skip path.
log_file="$service_dir/server.log"
nohup "$python_bin" "$service_dir/server.py" \
  --backend "$backend" --port "$port" >>"$log_file" 2>&1 &

for _ in $(seq 1 30); do
  if resp="$(curl --silent --max-time 1 "$health" 2>/dev/null)"; then
    running="$(printf '%s' "$resp" | sed -n 's/.*"backend"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
    if [[ "$running" == "$backend" ]]; then
      echo "local-stt started (backend=$backend port=$port); log: $log_file"
      exit 0
    fi
  fi
  sleep 0.2
done
echo "local-stt did not become healthy on port $port; see $log_file:" >&2
tail -n 20 "$log_file" >&2 || true
exit 1
