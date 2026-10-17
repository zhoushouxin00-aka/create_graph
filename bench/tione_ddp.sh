#!/usr/bin/env bash
# TI-ONE multi-node entry for bench/ddp_parity.py (PET and NequIP, NCCL).
# TI-ONE's DDP mode sets MASTER_ADDR / MASTER_PORT, RANK (node index) and
# WORLD_SIZE (number of nodes). The job mounts the same CFS, so the repo's
# venv runs as is.
set -euo pipefail

ROOT=/home/tione/notebook/home/zhoushouxin/mlip/create_graph
NODE_RANK="${RANK:-${INDEX:-0}}"
NODES="${WORLD_SIZE:-1}"
GPUS="${GPU_NUM_PER_NODE:-8}"
PORT="${MASTER_PORT:-29561}"
ADDR="${MASTER_ADDR:-127.0.0.1}"
OUT="$ROOT/bench/results/ddp/tione_${NODES}x${GPUS}"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$ROOT/src:$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/cg_inductor_node${NODE_RANK}"

mkdir -p "$OUT"
cd "$ROOT"
echo "node ${NODE_RANK}/${NODES} gpus=${GPUS} master=${ADDR}:${PORT} host=$(hostname)" \
  | tee "$OUT/node${NODE_RANK}.txt"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader >> "$OUT/node${NODE_RANK}.txt"

offset=0
for model in pet nequip; do
  "$ROOT/.venv/bin/torchrun" \
    --nnodes "$NODES" --node_rank "$NODE_RANK" \
    --master_addr "$ADDR" --master_port "$((PORT + offset))" \
    --nproc_per_node "$GPUS" \
    bench/ddp_parity.py --model "$model" --steps 4 --json "$OUT/${model}.json" \
    2>&1 | tee "$OUT/${model}_node${NODE_RANK}.log"
  offset=$((offset + 1))
done
echo "done node ${NODE_RANK}"
