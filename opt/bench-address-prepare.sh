#!/usr/bin/env bash
# 独立地址准备微测：一张NPU，无模型/BM，生产实现不切换。
set -eo pipefail
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
ARGS=()
PREVIEW=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sglang-dir) SGLANG_DIR="$2"; shift 2 ;;
        --help|-h|--describe) PREVIEW=1; ARGS+=("$1"); shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
cd "$SGLANG_DIR"
if [[ "$PREVIEW" == 0 ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
exec python3 ascend-mempool-test/scripts/bench_address_prepare.py "${ARGS[@]}"
