#!/usr/bin/env bash
# End-to-end fairchem training, AllScAIP md_NeNo (85M), real OMol25:
#   1 GPU, 30 steps each: create_graph / 2f2b / generated (feature/create_graph
#   worktree) and explicit (feature/explicit_create_graph worktree), at 1024
#   atoms (and 1536 with the memory budgets); atoms/s from the trainer's log
#   (print_every=1), peak memory from nvidia-smi every 2 s.
#   2 GPUs, 4 steps: generated and create_graph with the trainer's gradient
#   checksums (debug_checksums_save_path), to compare ranks and modes.
#
#   GPUS="6 7" bash bench/fairchem_e2e.sh
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MLIP=$(cd "$ROOT/.." && pwd)
OUT=${OUT:-$ROOT/bench/results/e2e}
RUNS=${RUNS:-/tmp/cg_e2e_runs}  # fairchem run dirs (checkpoints): outside the repo
GPUS=(${GPUS:-0 1})
PY=$MLIP/fairchem/.venv/bin/python
CG=$MLIP/fairchem-create_graph
EX=$MLIP/fairchem-explicit
PATHS=(
  "cluster.omol_all_path=$MLIP/datasets/omol25"
  "cluster.omol_4M_train_path=$MLIP/datasets/omol25/train_4M"
  "cluster.data_root_dir=$MLIP/datasets/omol25"
)
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled

cli() {  # cli <worktree> <yaml> <name> <devices> <ranks> [overrides...]
  local tree=$1 yaml=$2 name=$3 devices=$4 ranks=$5
  shift 5
  [[ -n ${ONLY:-} && " $ONLY " != *" $name "* ]] && return  # ONLY="a b": rerun some
  local dir="$OUT/$name"
  rm -rf "$dir" "$RUNS/$name" && mkdir -p "$dir" "$RUNS/$name"
  echo "=== $name $(date '+%F %T')"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$devices" -l 2 \
    >"$dir/memory.csv" 2>/dev/null &
  local sampler=$!
  (
    cd "$tree" &&
      CUDA_VISIBLE_DEVICES=$devices PYTHONPATH="$tree/src:$ROOT/src" \
        TORCHINDUCTOR_CACHE_DIR="$RUNS/$name/inductor" \
        "$PY" -B -c 'from fairchem.core._cli import main; main()' -c "$tree/allscaip-scripts/$yaml" \
        "cluster.ranks_per_node=$ranks" "cluster.run_dir=$RUNS/$name" "job.run_name=$name" \
        "+job.timestamp_id=runs" "runner.train_eval_unit.print_every=1" epochs=null \
        "++runner.max_eval_steps_per_epoch=1" "$@"
  ) >"$dir/stdout.log" 2>&1
  local status=$?
  kill "$sampler" 2>/dev/null
  echo "$name exit=$status" | tee -a "$OUT/status.txt"
}

g=${GPUS[0]}
cli "$CG" omol_4m_8gpu_atoms1024_cons.yaml cons_1024 "$g" 1 steps=30
cli "$CG" omol_4m_8gpu_atoms1024_cons_2f2b.yaml 2f2b_1024 "$g" 1 steps=30
cli "$CG" omol_4m_8gpu_atoms1024_cons_generated.yaml generated_1024 "$g" 1 steps=30
cli "$EX" omol_4m_8gpu_atoms1024_cons_explicit.yaml explicit_1024 "$g" 1 steps=30 "${PATHS[@]}"
cli "$CG" omol_4m_8gpu_atoms1536_cons_2f2b.yaml 2f2b_1536 "$g" 1 steps=30
cli "$CG" omol_4m_8gpu_atoms1536_cons_generated.yaml generated_1536 "$g" 1 steps=30
cli "$EX" omol_4m_8gpu_atoms1536_cons_explicit.yaml explicit_1536 "$g" 1 steps=30 "${PATHS[@]}"
cli "$CG" omol_4m_8gpu_atoms1024_cons_generated_ckpt.yaml generated_ckpt_1024 "$g" 1 steps=30

pair="${GPUS[0]},${GPUS[1]}"
for mode in generated cons; do
  yaml=omol_4m_8gpu_atoms1024_cons.yaml
  [[ $mode == generated ]] && yaml=omol_4m_8gpu_atoms1024_cons_generated.yaml
  cli "$CG" "$yaml" "ddp2_$mode" "$pair" 2 steps=4 \
    "++runner.train_eval_unit.debug_checksums_save_path=$OUT/ddp2_$mode/checksums"
done

"$PY" "$ROOT/bench/summarize_e2e.py" --dir "$OUT" | tee "$OUT/summary.md"
