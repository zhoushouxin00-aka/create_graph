#!/usr/bin/env bash
# torch version matrix. One venv per torch release under .venvs/ with the real
# PET (metatrain) and NequIP stacks; create_graph is installed --no-deps, so
# torch 2.10 (below the supported range) exercises the eager fallback.
#
#   bash bench/version_matrix.sh setup                 # CPU is enough
#   CUDA_VISIBLE_DEVICES=7 bash bench/version_matrix.sh test
#
# The main .venv (torch 2.13.0) is the fifth entry of the matrix.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VERSIONS=${VERSIONS:-"2.10.0 2.11.0 2.12.1 2.14.1"}
INDEX=http://mirrors.tencentyun.com/pypi/simple
OUT=${OUT:-$ROOT/bench/results/versions}
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}

venv_of() { [[ $1 == 2.13.0 ]] && echo "$ROOT/.venv" || echo "$ROOT/.venvs/torch-$1"; }

setup() {
  for v in $VERSIONS; do
    local venv
    venv=$(venv_of "$v")
    echo "=== torch $v -> $venv"
    "$UV" venv --python 3.12 --allow-existing "$venv" &&
      "$UV" pip install --python "$venv/bin/python" --index-url "$INDEX" \
        --allow-insecure-host mirrors.tencentyun.com \
        "torch==$v" "metatrain==2026.4.1" "nequip==0.19.1" ase "pytest>=8.3" &&
      "$UV" pip install --python "$venv/bin/python" --index-url "$INDEX" \
        --allow-insecure-host mirrors.tencentyun.com --no-deps -e "$ROOT" &&
      "$venv/bin/python" -c "import torch, metatrain, nequip, create_graph as cg; \
print(torch.__version__, torch.version.cuda, 'in-graph grad:', cg.supports_in_graph_grad())" ||
      echo "setup FAILED for torch $v"
  done
}

run_tests() {
  mkdir -p "$OUT"
  cd "$ROOT"
  for v in $VERSIONS 2.13.0; do
    local venv log
    venv=$(venv_of "$v")
    log="$OUT/torch-$v.log"
    [[ -x $venv/bin/python ]] || { echo "no venv for $v" | tee "$log"; continue; }
    echo "=== torch $v $(date '+%F %T')"
    # Every mechanism test, and the real-model accuracy, memory-knob, fused-op
    # and dynamic-shape tests on the smaller sizes; epochs and the largest
    # sizes run once, on the main venv (test_models_*.py).
    TORCHINDUCTOR_CACHE_DIR="/tmp/cg_inductor_versions_$v" PYTHONDONTWRITEBYTECODE=1 \
      "$venv/bin/python" -m pytest -p no:cacheprovider -q -rs \
      tests/test_torch_surface.py tests/test_semantics.py tests/test_custom_ops.py \
      tests/test_ddp.py tests/test_models_pet.py tests/test_models_nequip.py \
      -k "not epoch" \
      --deselect "tests/test_models_pet.py::test_float32_as_close_as_eager_create_graph[large]" \
      --deselect "tests/test_models_nequip.py::test_float32_as_close_as_eager_create_graph[M]" \
      --deselect "tests/test_models_nequip.py::test_float32_as_close_as_eager_create_graph[L]" \
      >"$log" 2>&1
    echo "torch $v exit=$? $(tail -1 "$log")" | tee -a "$OUT/summary.txt"
  done
}

case "${1:?setup|test}" in
  setup) setup ;;
  test) run_tests ;;
  *) echo "setup|test" >&2; exit 2 ;;
esac
