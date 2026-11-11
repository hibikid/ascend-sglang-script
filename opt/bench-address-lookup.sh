#!/usr/bin/env bash
# 独立微测：不启动模型，不连接 BM，不修改服务算子。
set -eo pipefail
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sglang-dir) SGLANG_DIR="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
cd "$SGLANG_DIR"
if [[ " ${ARGS[*]} " != *" --help "* && " ${ARGS[*]} " != *" -h "* ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
exec python3 ascend-mempool-test/scripts/bench_address_lookup.py "${ARGS[@]}"
