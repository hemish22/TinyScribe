#!/usr/bin/env bash
# Start / stop one llama-server per GPU (DGX) or a single Metal server (Mac).
#   bash scripts/serve_pool.sh start MODEL.gguf MMPROJ.gguf [GPUS=4,5,6,7] [SLOTS=2]
#   bash scripts/serve_pool.sh stop
# On the DGX each server listens on 127.0.0.1:(8100 + GPU index), so pools on different GPUs never
# clash; on the Mac on 8081. PIDs go to runs/servers.pids (runs/servers_$POOL.pids when POOL is set)
# so `stop` kills only what this pool started. Prints the --servers string for eval/run_eval.py.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=third_party/llama.cpp/build/bin/llama-server
[ -x "$BIN" ] || BIN="$(command -v llama-server)"
PIDS="runs/servers${POOL:+_$POOL}.pids"
mkdir -p runs/logs

case "${1:-}" in
  start)
    MODEL="$2"; MMPROJ="$3"; GPUS="${4:-4,5,6,7}"; SLOTS="${5:-2}"
    CTX=$((8192 * SLOTS))
    [ "$(uname)" = "Darwin" ] && GPUS="metal"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    urls=()
    for g in ${GPUS//,/ }; do
      env_gpu=(); port=8081
      [ "$g" != "metal" ] && { env_gpu=(CUDA_VISIBLE_DEVICES="$g"); port=$((8100 + g)); }
      env ${env_gpu[@]+"${env_gpu[@]}"} "$BIN" -m "$MODEL" --mmproj "$MMPROJ" -ngl 99 --jinja \
        -c "$CTX" -np "$SLOTS" --host 127.0.0.1 --port "$port" \
        > "runs/logs/server_$port.log" 2>&1 &
      echo $! >> "$PIDS"
      urls+=("http://127.0.0.1:$port")
    done
    (IFS=,; echo "servers: ${urls[*]}")
    ;;
  stop)
    [ -f "$PIDS" ] || { echo "no servers recorded"; exit 0; }
    while read -r pid; do kill "$pid" 2>/dev/null && echo "stopped $pid" || true; done < "$PIDS"
    rm -f "$PIDS"
    ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
