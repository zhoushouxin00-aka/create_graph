#!/usr/bin/env bash
# nsys profiles of a few steady-state training steps per method, and their
# kernel summaries (to explain where create_graph is slower than a baseline):
#
#   CUDA_VISIBLE_DEVICES=3 bash bench/profile_pair.sh <out_dir> <python> <fairchem tree or -> \
#       <model> <size> <atoms> <method>...
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MLIP=$(dirname "$ROOT")
out=$1 py=$2 tree=$3 model=$4 size=$5 atoms=$6
shift 6
mkdir -p "$out"
path="$ROOT/src:$ROOT"
root=""
[[ $tree != - ]] && path="$MLIP/$tree/src:$path" && root="$MLIP/$tree"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
for method in "$@"; do
  name="${model}_${size}_${method}_${atoms}"
  FAIRCHEM_ROOT=$root PYTHONPATH=$path TORCHINDUCTOR_CACHE_DIR=/tmp/cg_profile_$name \
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true -o "$out/$name" \
    "$py" -m cg_bench.runner --model "$model" --size "$size" --method "$method" \
    --atoms "$atoms" --warmup 3 --steps 2 --profile-steps 3 --tf32 \
    >"$out/$name.log" 2>&1
  echo "$name exit=$?"
  nsys stats -r cuda_gpu_kern_sum,nvtx_sum --format csv --force-export=true \
    -o "$out/$name" "$out/$name.nsys-rep" >/dev/null 2>&1
done
