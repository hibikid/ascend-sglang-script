#!/usr/bin/env bash
# Single-NPU address demo. Optional build/install, two rounds, persistent logs.
set -eo pipefail
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
KERNEL_DIR=${KERNEL_DIR:-/home/cryang/sgl-kernel-npu}
RUN_DIR=/home/cryang/log/mempool-opt/address-kernel-$(date +%Y%m%d-%H%M%S)
BUILD=0
PREVIEW=0
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sglang-dir) SGLANG_DIR="$2"; shift 2 ;;
        --kernel-dir) KERNEL_DIR="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --build-kernel) BUILD=1; shift ;;
        --help|-h)
            printf '%s\n' 'Wrapper: --sglang-dir PATH --kernel-dir PATH --run-dir PATH --build-kernel'
            PREVIEW=1; ARGS+=("$1"); shift ;;
        --describe) PREVIEW=1; ARGS+=("$1"); shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
if [[ "$PREVIEW" == 1 ]]; then
    exec python3 "$SGLANG_DIR/ascend-mempool-test/scripts/bench_address_kernel.py" \
        --kernel-dir "$KERNEL_DIR" --output-dir "$RUN_DIR/results" "${ARGS[@]}"
fi
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
exec > >(tee "$RUN_DIR/runner.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "$RUN_DIR/exit-code"' EXIT
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
if [[ "$BUILD" == 1 ]]; then
    cd "$KERNEL_DIR"
    git rev-parse HEAD > "$RUN_DIR/kernel-build-sha.txt"
    bash build.sh -a kernels
    wheel=$(python3 -c 'from pathlib import Path; print(max(Path("output").glob("sgl_kernel_npu*.whl"), key=lambda p: p.stat().st_mtime))')
    python3 -m pip install --force-reinstall --no-deps "$wheel"
fi
cd "$SGLANG_DIR"
python3 ascend-mempool-test/scripts/bench_address_kernel.py \
    --kernel-dir "$KERNEL_DIR" --output-dir "$RUN_DIR/results" "${ARGS[@]}"
