#!/usr/bin/env bash
# Entry of the TI-ONE task for the final benchmark run on one dedicated 8-GPU
# node: every benchmark lane and the fairchem CLI end-to-end runs.
ROOT=/home/tione/notebook/home/zhoushouxin/mlip/create_graph
RES=$ROOT/bench/results/run-tione-$(date +%Y%m%d-%H%M) \
LANES="allscaip_explicit allscaip_2f2b pet nequip uma+allscaip_sizes allscaip_explicit_large e2e" \
  exec bash "$ROOT/bench/tione_bench.sh"
