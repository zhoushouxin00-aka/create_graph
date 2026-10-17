#!/usr/bin/env bash
# Benchmark shards: every (model, size, method) in its own process on the GPU
# named by CUDA_VISIBLE_DEVICES, results in $OUT/<shard>/. Within a shard every
# method sees the same batches (same seeds). A failure or OOM does not stop the
# shard.
#
#   CUDA_VISIBLE_DEVICES=0 bash bench/run_matrix.sh pet
#   shards: allscaip_explicit allscaip_explicit_large allscaip_2f2b allscaip_sizes
#           pet nequip uma
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MLIP=$(cd "$ROOT/.." && pwd)
shard=${1:?shard}
OUT=${OUT:-$ROOT/bench/results/matrix}/$shard
STEPS=${STEPS:-10}
mkdir -p "$OUT"
CG_PY=${CG_PY:-$ROOT/.venv/bin/python}
FAIRCHEM_PY=${FAIRCHEM_PY:-$MLIP/fairchem/.venv/bin/python}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/cg_inductor_$(id -u)_${CUDA_VISIBLE_DEVICES:-0}}

run() {  # run <python> <pythonpath> <model> <size> <method> <atoms> [extra runner flags]
  local py=$1 path=$2 model=$3 size=$4 method=$5 atoms=$6
  shift 6
  local name="${model}_${size}_${method}_${atoms}"
  mkdir -p "$OUT"
  echo "=== $name $(date '+%F %T')"
  if PYTHONPATH="$path" "$py" -m cg_bench.runner --model "$model" --size "$size" \
      --method "$method" --atoms "$atoms" --warmup 3 --steps "$STEPS" --tf32 \
      --json "$OUT/$name.json" "$@" >"$OUT/$name.log" 2>&1; then
    echo "ok $name"
  else
    echo "FAILED $name" | tee -a "$OUT/failures.txt"
  fi
}

# fairchem configs come from the checkout whose code runs
explicit_tree() {
  export FAIRCHEM_ROOT=$MLIP/fairchem-explicit
  path="$MLIP/fairchem-explicit/src:$ROOT/src:$ROOT"
}
create_graph_tree() {
  export FAIRCHEM_ROOT=$MLIP/fairchem-create_graph
  path="$MLIP/fairchem-create_graph/src:$ROOT/src:$ROOT"
}

case "$shard" in
  allscaip_explicit)  # 85M at its training batch, against the hand-written explicit VJP
    explicit_tree
    for method in cg_eager explicit_compiled explicit_budget explicit_ckpt cg cg_budget cg_ckpt; do
      run "$FAIRCHEM_PY" "$path" allscaip h640_l10 "$method" 1024 --max-atoms --max-limit 16384
    done
    for budget in 0.3 0.7; do
      for method in explicit_budget cg_budget; do
        OUT="$OUT/budget$budget" run "$FAIRCHEM_PY" "$path" allscaip h640_l10 "$method" 1024 --budget "$budget"
      done
    done
    ;;
  allscaip_explicit_large)  # 85M above 1024 atoms: only the memory knobs fit
    explicit_tree
    for method in explicit_budget cg_budget explicit_ckpt cg_ckpt; do
      run "$FAIRCHEM_PY" "$path" allscaip h640_l10 "$method" 1536
    done
    for atoms in 2048 3072; do
      for method in explicit_ckpt cg_ckpt; do
        run "$FAIRCHEM_PY" "$path" allscaip h640_l10 "$method" "$atoms"
      done
    done
    ;;
  allscaip_2f2b)  # 85M against the hand-written 2F2B of feature/2f2b
    create_graph_tree
    for method in cg_eager 2f2b 2f2b_budget 2f2b_ckpt cg cg_budget cg_ckpt; do
      run "$FAIRCHEM_PY" "$path" allscaip h640_l10 "$method" 1024 --max-atoms --max-limit 16384
    done
    ;;
  allscaip_sizes)  # smaller AllScAIP models
    create_graph_tree
    for size in h128_l2 h256_l4 h512_l6; do
      for method in cg_eager 2f2b cg; do
        run "$FAIRCHEM_PY" "$path" allscaip "$size" "$method" 1024 --max-atoms --max-limit 16384
      done
    done
    ;;
  pet)
    for method in native_eager cg_eager cg cg_budget cg_ckpt; do
      run "$CG_PY" "$ROOT/src:$ROOT" pet default "$method" 4096 --max-atoms --max-limit 131072
      run "$CG_PY" "$ROOT/src:$ROOT" pet large "$method" 2048 --max-atoms --max-limit 131072
    done
    ;;
  nequip)
    for size in S M L; do
      atoms=4096
      [[ $size == L ]] && atoms=2048
      for method in nequip_eager nequip_compile cg_eager cg cg_budget cg_ckpt; do
        run "$CG_PY" "$ROOT/src:$ROOT" nequip "$size" "$method" "$atoms" --max-atoms --max-limit 131072
      done
    done
    ;;
  uma)
    create_graph_tree
    for size in s m; do
      for method in cg_eager cg cg_budget cg_ckpt; do
        run "$FAIRCHEM_PY" "$path" uma "$size" "$method" 300
      done
    done
    ;;
  *)
    echo "unknown shard $shard" >&2
    exit 2
    ;;
esac
echo "shard $shard done $(date '+%F %T')"
