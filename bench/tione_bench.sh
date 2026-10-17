#!/usr/bin/env bash
# A whole benchmark run on one dedicated 8-GPU node (TI-ONE, or this machine
# when it is idle), one lane per GPU, each pinned to the CPU cores next to it:
#   0 AllScAIP vs explicit (1024 atoms)  1 AllScAIP vs 2F2B
#   2 PET                                3 NequIP
#   4 UMA, then smaller AllScAIP sizes   5 AllScAIP vs explicit above 1024 atoms
#   6 torch version matrix               7 fairchem CLI end-to-end (DDP on 7,6)
# plus fairchem_tests (create_graph repo and the feature/create_graph branch)
# when selected. The job mounts the same CFS, so the venvs run as they are.
# LANES selects a subset, e.g. LANES="pet nequip".
set -uo pipefail

ROOT=/home/tione/notebook/home/zhoushouxin/mlip/create_graph
MLIP=$(dirname "$ROOT")
STAMP=${STAMP:-$(date +%Y%m%d-%H%M)}
RES=${RES:-$ROOT/bench/results/run-$STAMP}
mkdir -p "$RES"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd "$ROOT"

declare -A GPU=(
  [allscaip_explicit]=0 [allscaip_2f2b]=1 [pet]=2 [nequip]=3 [uma+allscaip_sizes]=4
  [allscaip_explicit_large]=5 [versions]=6 [e2e]=7,6 [fairchem_tests]=5
)
LANES=${LANES:-"allscaip_explicit allscaip_2f2b pet nequip uma+allscaip_sizes allscaip_explicit_large versions e2e"}

{
  echo "host=$(hostname) start=$(date '+%F %T') lanes=$LANES"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
  git -C "$MLIP/fairchem-create_graph" log --oneline -1
  git -C "$MLIP/fairchem-explicit" log --oneline -1
} >>"$RES/node.txt" 2>&1

# GPU memory and utilization every 30 s: a record that nothing else shared the node
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 30 \
  >>"$RES/gpu_usage.csv" 2>/dev/null &
sampler=$!

pids=()
for lane in $LANES; do
  bash bench/lane.sh "$lane" "${GPU[$lane]}" "$RES" &
  pids+=($!)
done
wait "${pids[@]}"
kill "$sampler" 2>/dev/null

"$ROOT/.venv/bin/python" bench/summarize.py --dir "$RES/matrix" >"$RES/summary.md" 2>&1
echo "done $(date '+%F %T')" >>"$RES/node.txt"
