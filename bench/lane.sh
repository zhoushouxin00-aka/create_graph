#!/usr/bin/env bash
# One lane of a benchmark run on the given GPUs, pinned to an equal share of
# the CPU threads next to them (cores of GPU g: the g-th slice of the physical
# cores and of their hyperthreads), so lanes do not steal CPU from each other:
#
#   bash bench/lane.sh pet 2 bench/results/<run>
#   lanes: a benchmark shard of run_matrix.sh, or several joined by "+"
#   (uma+allscaip_sizes), fairchem_tests, e2e, versions
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MLIP=$(dirname "$ROOT")
lane=${1:?lane}
gpus=${2:?gpus}
RES=${3:?results dir}
mkdir -p "$RES"

ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
allowed=($(python3 -c 'import os; print(*sorted(os.sched_getaffinity(0)))' 2>/dev/null))
total=${#allowed[@]}
cores=""
threads=8
if [[ $total -lt $ngpu ]]; then
  :  # unknown CPU set: no pinning
elif [[ $total -eq $(nproc --all) && $((total % (2 * ngpu))) -eq 0 ]]; then
  # whole machine: GPU g gets the g-th slice of the physical cores and their hyperthreads
  half=$((total / 2 / ngpu))
  for g in ${gpus//,/ }; do
    a=$((g * half))
    b=$((total / 2 + g * half))
    cores+="${a}-$((a + half - 1)),${b}-$((b + half - 1)),"
  done
  threads=$half
else
  # a container's CPU set: equal slices of the CPUs this process may use
  per=$((total / ngpu))
  for g in ${gpus//,/ }; do
    for ((i = g * per; i < (g + 1) * per; i++)); do cores+="${allowed[$i]},"; done
  done
  threads=$((per / 2 > 0 ? per / 2 : 1))
fi
cores=${cores%,}
export TORCHINDUCTOR_COMPILE_THREADS=$threads
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled

case "$lane" in
  allscaip* | pet* | nequip* | uma*)
    shards=""
    for shard in ${lane//+/ }; do
      shards+="bash $ROOT/bench/run_matrix.sh $shard; "
    done
    cmd=(env OUT="$RES/matrix" bash -c "$shards")
    ;;
  fairchem_tests)
    cmd=(bash -c "
      cd $ROOT && FAIRCHEM_ROOT=$MLIP/fairchem-create_graph \
      PYTHONPATH=$MLIP/fairchem-create_graph/src:$ROOT/src:$ROOT $MLIP/fairchem/.venv/bin/python \
        -m pytest -p no:cacheprovider -q -rs --durations=0 tests/test_models_fairchem.py;
      cd $MLIP/fairchem-create_graph && PYTHONPATH=src:$ROOT/src $MLIP/fairchem/.venv/bin/python \
        -m pytest -p no:cacheprovider -q -rs -c packages/fairchem-core/pyproject.toml --noconftest \
        --maxfail=100 tests/core/models/allscaip/test_allscaip_generated.py \
        tests/core/models/allscaip/test_allscaip_2f2b.py")
    ;;
  e2e)
    cmd=(env GPUS="${gpus//,/ }" OUT="$RES/e2e" bash "$ROOT/bench/fairchem_e2e.sh")
    ;;
  versions)
    cmd=(env OUT="$RES/versions" bash "$ROOT/bench/version_matrix.sh" test)
    ;;
  *)
    echo "unknown lane $lane" >&2
    exit 2
    ;;
esac

pin=()
[[ -n $cores ]] && pin=(taskset -c "$cores")
echo "$lane gpus=$gpus cores=${cores:-all} start=$(date '+%F %T')" >>"$RES/status.txt"
CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=/tmp/cg_inductor_$lane \
  "${pin[@]}" "${cmd[@]}" >"$RES/$lane.log" 2>&1
echo "$lane exit=$? end=$(date '+%F %T')" >>"$RES/status.txt"
