#!/usr/bin/env bash
# 一台主机一条命令；统计开/关各启动一次，重放同一负载并清理本轮进程。
set -eo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sglang-dir) SGLANG_DIR="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
cd "$SGLANG_DIR"
exec python3 ascend-mempool-test/scripts/run_source_evidence.py \
    --script-dir "$SCRIPT_DIR" \
    --config "$SCRIPT_DIR/glm52mempool-opt.json" \
    "${ARGS[@]}"
