#!/usr/bin/env bash
set -eo pipefail

# 默认值在这里；空的覆盖项表示读取同目录 JSON。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
CONFIG_FILE="$SCRIPT_DIR/glm52mempool-opt.json"
RUN_ID=opt-001
LOG_DIR=""
ROLE=router
P_IP=""             # 默认 10.120.72.31
D_IP=""             # 默认 10.120.72.32
MODEL_PATH=""       # 默认 /data_lib/data/models/GLM-5.2-w4a8
INTERFACE=""        # 默认 bond4
P_ALIVE_PAGE=""     # 默认 0；dual_write 不使用 P alive main
P_INDEX_PAGE=""     # 默认 4096
D_ALIVE_PAGE=""     # 默认 4096
PREFIX_PAGE=""      # 默认 192；后续两张 baseline 固定此容量
PAGE_SIZE=""        # 默认 64 token/page
DRY_RUN=0
LAUNCH_ARGS=("$@")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --p-ip) P_IP="$2"; shift 2 ;;
        --d-ip) D_IP="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --interface) INTERFACE="$2"; shift 2 ;;
        --palive-page) P_ALIVE_PAGE="$2"; shift 2 ;;
        --pindex-page) P_INDEX_PAGE="$2"; shift 2 ;;
        --dalive-page) D_ALIVE_PAGE="$2"; shift 2 ;;
        --prefix-page) PREFIX_PAGE="$2"; shift 2 ;;
        --page-size) PAGE_SIZE="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --sglang-dir) SGLANG_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash opt/router-mempool-opt.sh [options]
  --p-ip IP              --d-ip IP
  --model-path PATH      --interface bond4
  --palive-page 0        --pindex-page 4096
  --dalive-page 4096     --prefix-page 192     --page-size 64
  --config FILE          其他容量、端口、NUMA 参数在 JSON 中修改
  --run-id opt-001       Router/P/D 三者必须一致
  --log-dir DIR          默认 /home/cryang/log/mempool-opt/<run-id>
  --sglang-dir DIR       默认 /home/cryang/sglang
  --dry-run              打印参数；不启动服务、不加载模型
HELP
            exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="/home/cryang/log/mempool-opt/$RUN_ID"
fi
# 配置路径先转为绝对路径，避免 cd 后改变相对路径含义。
CONFIG_FILE=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_FILE")
RUN_DIR=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$LOG_DIR")
cd "$SGLANG_DIR"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 只把配置解析放在 Python；模型启动命令完整保留在下面。
ENV_FILE=$(mktemp)
trap 'rm -f "$ENV_FILE"' EXIT
DRY_ARGS=()
if [[ "$DRY_RUN" == 1 ]]; then DRY_ARGS=(--dry-run); fi
python3 ascend-mempool-test/scripts/opt_config.py \
    --config "$CONFIG_FILE" \
    --role "$ROLE" \
    --run-id "$RUN_ID" \
    --run-dir "$RUN_DIR" \
    --env-file "$ENV_FILE" \
    --p-ip "$P_IP" \
    --d-ip "$D_IP" \
    --model-path "$MODEL_PATH" \
    --interface "$INTERFACE" \
    --palive-page "$P_ALIVE_PAGE" \
    --pindex-page "$P_INDEX_PAGE" \
    --dalive-page "$D_ALIVE_PAGE" \
    --prefix-page "$PREFIX_PAGE" \
    --page-size "$PAGE_SIZE" \
    "${DRY_ARGS[@]}"
source "$ENV_FILE"

if [[ "$DRY_RUN" == 0 ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PATH="/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH"
    mkdir -p "$RUN_DIR"
    { git rev-parse HEAD; printf '%q ' "$0" "${LAUNCH_ARGS[@]}"; printf '\n'; } > "$RUN_DIR/$ROLE-launch.txt"
fi

# dry-run 和真正启动共用下面同一条命令。
run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@" 2>&1 | tee "$RUN_DIR/$ROLE.log"
    fi
}

# 测量版 Main Router，保留既有分配/前缀复用协议，记录每请求 TTFT/TPOT。
# 启动顺序：Router -> P -> D；P 等待 BM join 时就可以启动 D。
run python3 -u ascend-mempool-test/scripts/measure_mempool_router.py \
    --config "$CONFIG_FILE" \
    --run-id "$RUN_ID"
