#!/usr/bin/env bash
# One short step of every benchmark workload (small sizes), one GPU per model,
# before a long run: GPUS="0 1 2 3 5" bash bench/smoke_runner.sh
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MLIP=$(cd "$ROOT/.." && pwd)
OUT=${OUT:-$ROOT/bench/results/smoke_runner}
GPUS=(${GPUS:-0 1 2 3 4})
CG_PY=$ROOT/.venv/bin/python
FC_PY=$MLIP/fairchem/.venv/bin/python
rm -rf "$OUT" && mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled

each() {  # each <gpu> <python> <fairchem checkout or -> <model> <size> <atoms> <methods...>
  local gpu=$1 py=$2 tree=$3 model=$4 size=$5 atoms=$6
  shift 6
  local path="$ROOT/src:$ROOT" root=""
  [[ $tree != - ]] && path="$MLIP/$tree/src:$path" && root="$MLIP/$tree"
  for method in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu FAIRCHEM_ROOT=$root PYTHONPATH=$path \
      TORCHINDUCTOR_CACHE_DIR=/tmp/cg_smoke_$gpu \
      "$py" -m cg_bench.runner --model "$model" --size "$size" --method "$method" \
      --atoms "$atoms" --warmup 1 --steps 1 --tf32 --json "$OUT/${model}_${method}.json" \
      >"$OUT/${model}_${size}_${method}.log" 2>&1
    echo "$model $size $method exit=$? $(tail -c 300 "$OUT/${model}_${size}_${method}.log" | tr '\n' ' ')"
  done
}

each "${GPUS[0]}" "$CG_PY" - pet default 512 native_eager cg_eager cg cg_budget cg_ckpt &
each "${GPUS[1]}" "$CG_PY" - nequip S 512 nequip_eager nequip_compile cg_eager cg cg_ckpt &
each "${GPUS[2]}" "$FC_PY" fairchem-explicit allscaip h128_l2 256 \
  cg_eager explicit_compiled explicit_budget explicit_ckpt cg cg_ckpt &
each "${GPUS[3]}" "$FC_PY" fairchem-create_graph allscaip h128_l2 256 2f2b 2f2b_ckpt cg_budget &
each "${GPUS[4]}" "$FC_PY" fairchem-create_graph uma s 300 cg_eager cg cg_ckpt &
wait
