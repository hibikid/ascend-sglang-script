#!/usr/bin/env bash
set -eo pipefail

# 在被采集的 P 或 D 本机运行；URL 指向该 worker，不是 Router。
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
URL=http://10.120.72.32:8001
RANKS=0
SECONDS_TO_RECORD=2
OUTPUT_DIR="/home/cryang/log/mempool-opt/traces/$(date +%Y%m%d-%H%M%S)"
TIMEOUT=1800
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url) URL="$2"; shift 2 ;;
        --ranks) RANKS="$2"; shift 2 ;;
        --seconds) SECONDS_TO_RECORD="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --sglang-dir) SGLANG_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash opt/profiler.sh [options]
  --url http://10.120.72.32:8001  D worker；P 使用 http://10.120.72.31:8000
  --ranks 0,1                   本机 TP rank，支持 all；默认只采 rank 0
  --seconds 2                   start API 返回后等待的秒数，不含 stop 导出时间
  --output-dir DIR              新目录，默认 /home/cryang/log/mempool-opt/traces/<时间>
  --timeout 1800                每次 profile API 的超时秒数
  --sglang-dir DIR              默认 /home/cryang/sglang
  --dry-run                     只打印采集计划
先运行 requests.sh，确认有负载后，在本机另开终端执行本脚本。
需要本轮支持 profile_ranks 的 SGLang；CPU/NPU 同采，stack/shapes/merge 关闭。
HELP
            exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

OUTPUT_DIR=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_DIR")
cd "$SGLANG_DIR"
DRY_ARGS=()
if [[ "$DRY_RUN" == 1 ]]; then DRY_ARGS=(--dry-run); fi
python3 ascend-mempool-test/scripts/opt_profile.py \
    --url "$URL" \
    --ranks "$RANKS" \
    --seconds "$SECONDS_TO_RECORD" \
    --output-dir "$OUTPUT_DIR" \
    --timeout "$TIMEOUT" \
    "${DRY_ARGS[@]}"
