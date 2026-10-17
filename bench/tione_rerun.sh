#!/usr/bin/env bash
# Entry of a TI-ONE task that reruns some lanes one after the other on GPU 0:
#   bash bench/tione_rerun.sh "uma" uma
ROOT=/home/tione/notebook/home/zhoushouxin/mlip/create_graph
RES=$ROOT/bench/results/run-tione-${2:-rerun}-$(date +%Y%m%d-%H%M)
mkdir -p "$RES"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
for lane in $1; do
  bash "$ROOT/bench/lane.sh" "$lane" 0 "$RES"
done
"$ROOT/.venv/bin/python" "$ROOT/bench/summarize.py" --dir "$RES/matrix" >"$RES/summary.md" 2>&1
