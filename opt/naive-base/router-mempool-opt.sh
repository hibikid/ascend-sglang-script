#!/usr/bin/env bash
set -eo pipefail

# 工具从当前性能分支加载，模型服务从旧版独立 checkout 加载。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
NAIVE_SGLANG_DIR=${NAIVE_SGLANG_DIR:-/home/cryang/sglang-naive-profile}
CONFIG_FILE=""
RUN_ID=naive-001
LOG_DIR=""
ROLE=router
ROUTER_HOST=""
ROUTER_PORT=6699
TIMEOUT=1800
P_IP=""
D_IP=""
MODEL_PATH=""
INTERFACE=""
PAGE_SIZE=""
PROMPT_CAPACITY=""
DECODE_CAPACITY=""
MEMPOOL_CONSOLE=${SGLANG_MEMPOOL_CONSOLE:-0}
DRY_RUN=0
LAUNCH_ARGS=("$@")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) ROUTER_HOST="$2"; shift 2 ;;
        --port) ROUTER_PORT="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --p-ip) P_IP="$2"; shift 2 ;;
        --d-ip) D_IP="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --interface) INTERFACE="$2"; shift 2 ;;
        --page-size) PAGE_SIZE="$2"; shift 2 ;;
        --prompt-capacity) PROMPT_CAPACITY="$2"; shift 2 ;;
        --decode-capacity) DECODE_CAPACITY="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --sglang-dir) SGLANG_DIR="$2"; shift 2 ;;
        --naive-sglang-dir) NAIVE_SGLANG_DIR="$2"; shift 2 ;;
        --mempool-console) MEMPOOL_CONSOLE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash opt/naive-base/router-mempool-opt.sh [options]
  --host IP --port 6699       默认绑定P_IP，供D上的请求客户端访问
  --timeout 1800             后端请求超时秒数
  --run-id naive-001          默认日志 /home/cryang/log/mempool-opt/naive-base/<run-id>
  --log-dir DIR              显式运行目录
  --config FILE              JSON；launch默认读该运行目录的config.json
  --p-ip IP --d-ip IP --model-path PATH --interface bond4
  --page-size 64             原生HBM页大小
  --prompt-capacity 8192     每个P slot预留token数
  --decode-capacity 8704     每个D slot预留token数；不是输出请求长度
  --sglang-dir DIR           当前工具仓库，默认/home/cryang/sglang
  --naive-sglang-dir DIR     旧版checkout，默认/home/cryang/sglang-naive-profile
  --mempool-console 0|1      默认0；完整日志始终保存
  --dry-run                  不建checkout、不启动服务、不加载模型
先运行prepare.sh，P/D与本Router各占一个终端；本入口使用旧版原生PD MiniLB。
HELP
            exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done
if [[ "$MEMPOOL_CONSOLE" != 0 && "$MEMPOOL_CONSOLE" != 1 ]]; then
    echo "--mempool-console / SGLANG_MEMPOOL_CONSOLE must be 0 or 1" >&2; exit 2
fi
if [[ -z "$LOG_DIR" ]]; then LOG_DIR="/home/cryang/log/mempool-opt/naive-base/$RUN_ID"; fi
RUN_DIR=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$LOG_DIR")
SGLANG_DIR=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$SGLANG_DIR")
NAIVE_SGLANG_DIR=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$NAIVE_SGLANG_DIR")
TOOLS_DIR="$SGLANG_DIR/ascend-mempool-test/scripts"
if [[ -z "$CONFIG_FILE" ]]; then
    if [[ -f "$RUN_DIR/config.json" ]]; then CONFIG_FILE="$RUN_DIR/config.json";
    else CONFIG_FILE="$SCRIPT_DIR/glm52mempool-opt.json"; fi
fi
CONFIG_FILE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_FILE")
ENV_FILE=$(mktemp)
trap 'rm -f "$ENV_FILE"' EXIT
DRY_ARGS=()
if [[ "$DRY_RUN" == 1 ]]; then DRY_ARGS=(--dry-run); fi
python3 "$TOOLS_DIR/opt_naive_config.py" launch-env \
    --config "$CONFIG_FILE" \
    --run-dir "$RUN_DIR" \
    --naive-sglang-dir "$NAIVE_SGLANG_DIR" \
    --role "$ROLE" \
    --env-file "$ENV_FILE" \
    --p-ip "$P_IP" \
    --d-ip "$D_IP" \
    --model-path "$MODEL_PATH" \
    --interface "$INTERFACE" \
    --page-size "$PAGE_SIZE" \
    --prompt-capacity "$PROMPT_CAPACITY" \
    --decode-capacity "$DECODE_CAPACITY" \
    "${DRY_ARGS[@]}"
source "$ENV_FILE"

if [[ -z "$ROUTER_HOST" ]]; then ROUTER_HOST="$P_IP"; fi
export SGLANG_MEMPOOL_CONSOLE="$MEMPOOL_CONSOLE"
export PYTHONPATH="$NAIVE_SGLANG_DIR/sgl-model-gateway/bindings/python/src:$NAIVE_SGLANG_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "$@"
        printf '| tee %q | python3 -u %q --console %q\n' "$RUN_DIR/router.log" "$TOOLS_DIR/mempool_console.py" "$MEMPOOL_CONSOLE"
    else
        {
            git -C "$NAIVE_SGLANG_DIR" rev-parse HEAD
            printf 'mempool_console=%s\n' "$MEMPOOL_CONSOLE"
            printf '%q ' "$@"
            printf '\n'
        } > "$RUN_DIR/router-launch.txt"
        python3 "$TOOLS_DIR/opt_naive_config.py" identity \
            --config "$CONFIG_FILE" --run-dir "$RUN_DIR" \
            --naive-sglang-dir "$NAIVE_SGLANG_DIR" --role router
        "$@" 2>&1 | tee "$RUN_DIR/router.log" | python3 -u "$TOOLS_DIR/mempool_console.py" --console "$MEMPOOL_CONSOLE"
    fi
}
run python3 -u -m sglang_router.launch_router \
    --pd-disaggregation \
    --policy random \
    --prefill "http://$P_IP:$P_PORT" "$BOOTSTRAP_PORT" \
    --decode "http://$D_IP:$D_PORT" \
    --host "$ROUTER_HOST" \
    --port "$ROUTER_PORT" \
    --request-timeout-secs "$TIMEOUT" \
    --mini-lb
