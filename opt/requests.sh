#!/usr/bin/env bash
set -eo pipefail

SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
CONFIG_FILE=/home/cryang/log/mempool-opt/opt-001/config.json
INPUT_LEN=4096
OUTPUT_LEN=128
CONCURRENCY=1
COUNT=1
MODE=cold
WORKLOAD=""
OUTPUT_DIR="/home/cryang/log/mempool-opt/requests/$(date +%Y%m%d-%H%M%S)"
TIMEOUT=1800
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --count) COUNT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --workload) WORKLOAD="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --sglang-dir) SGLANG_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash opt/requests.sh [options]
  --config FILE         Router/P/D 生成的最终 config.json
  --input-len 4096      含模板的实际 token 数，必须为 page-size 的倍数
  --output-len 128      强制生成 token 数；默认服务配置上限 512
  --concurrency 1       客户端并发；实际活跃请求数见日志
  --count 1             测量请求总数
  --mode cold|reuse     cold 每请求独立首个 token 页；reuse 先发种子再复用同一输入
  --workload FILE       重放已保存的 workload.json；长度、数量和 mode 需匹配
  --output-dir DIR      新目录，默认 /home/cryang/log/mempool-opt/requests/<时间>
  --timeout 1800        等待服务就绪及单条请求的超时秒数
  --sglang-dir DIR      默认 /home/cryang/sglang
  --dry-run             校验参数并打印计划，不加载 tokenizer、不发请求
结果保存 workload.json、每请求响应、Router 状态和 report.json；失败退出非零。
HELP
            exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

CONFIG_FILE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_FILE")
OUTPUT_DIR=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_DIR")
WORKLOAD_ARGS=()
if [[ -n "$WORKLOAD" ]]; then
    WORKLOAD=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$WORKLOAD")
    WORKLOAD_ARGS=(--workload "$WORKLOAD")
fi
DRY_ARGS=()
if [[ "$DRY_RUN" == 1 ]]; then DRY_ARGS=(--dry-run); fi
cd "$SGLANG_DIR"
python3 ascend-mempool-test/scripts/opt_requests.py \
    --config "$CONFIG_FILE" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --concurrency "$CONCURRENCY" \
    --count "$COUNT" \
    --mode "$MODE" \
    --output-dir "$OUTPUT_DIR" \
    --timeout "$TIMEOUT" \
    "${WORKLOAD_ARGS[@]}" \
    "${DRY_ARGS[@]}"
