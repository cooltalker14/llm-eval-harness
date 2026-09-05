#!/bin/bash
#SBATCH --job-name=llm-eval
#SBATCH --account=def-hsajjad
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

# Serve a model, run the eval dataset against it, shut down.
#
#   sbatch scripts/eval_job.sh bf16
#   sbatch scripts/eval_job.sh awq
#
# 40 items at temperature 0 takes about a minute once the server is up;
# nearly all the wall time is vLLM startup.

set -euo pipefail

TAG="${1:-bf16}"
CACHE=/home/swap1411/.cache/huggingface/hub

case "$TAG" in
  bf16)
    MODEL_PATH="$CACHE/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    QUANT_ARGS=""
    ;;
  awq)
    MODEL_PATH="$CACHE/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641"
    QUANT_ARGS="--quantization awq_marlin"
    ;;
  *)
    echo "unknown tag '$TAG' (expected: bf16 | awq)" >&2
    exit 1
    ;;
esac

# Pick a free port instead of hardcoding 8000: the eval must talk to OUR
# server, not to whatever else happens to be listening.
PORT=$(python - <<'EOF'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0))
print(s.getsockname()[1]); s.close()
EOF
)

module load python/3.11 cuda/12.2 gcc opencv
source /home/swap1411/projects/def-hsajjad/swap1411/llm-serving-bench/venvs/bin/activate

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs runs cache

echo "[job] node=$(hostname) tag=$TAG port=$PORT"
echo "[job] model=$MODEL_PATH"
python -c "import torch, vllm, scipy; print('torch', torch.__version__, '| vllm', vllm.__version__, '| scipy', scipy.__version__)"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "[job] FATAL: no config.json at $MODEL_PATH" >&2
  exit 1
fi

vllm serve "$MODEL_PATH" \
  --served-model-name qwen7b \
  --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  $QUANT_ARGS &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "[job] waiting for server..."
READY=0
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    READY=1
    echo "[job] server ready after ~$((i*10))s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[job] FATAL: vLLM exited during startup" >&2
    exit 1
  fi
  sleep 10
done
[ "$READY" -eq 1 ] || { echo "[job] FATAL: server not ready after 15 min" >&2; exit 1; }

# Confirm we are talking to vLLM and not something else on the port.
if ! curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q '"qwen7b"'; then
  echo "[job] FATAL: port $PORT is not serving qwen7b" >&2
  exit 1
fi

echo ""
echo "=== GENERATING (tag=$TAG) ==="
PYTHONPATH=src python -m evalkit.run \
  --dataset data/incidents.jsonl \
  --base-url "http://127.0.0.1:$PORT" \
  --model qwen7b \
  --tag "$TAG"

# A run that errored on every item produces a file that looks valid but
# contains nothing. Fail loudly instead of leaving that for the comparison.
ERRORS=$(python -c "
import json
print(json.load(open('runs/$TAG.json'))['n_errors'])
")
if [ "$ERRORS" -gt 0 ]; then
  echo "[job] WARNING: $ERRORS items failed to generate" >&2
  python -c "
import json
for g in json.load(open('runs/$TAG.json'))['generations']:
    if g['error']:
        print('  ', g['item_id'], g['error'][:120]); break
"
  [ "$ERRORS" -eq 40 ] && { echo "[job] FATAL: every item failed" >&2; exit 1; }
fi

echo "[job] done — runs/$TAG.json"