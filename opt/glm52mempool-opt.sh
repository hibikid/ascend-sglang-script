#!/usr/bin/env bash
set -eo pipefail

# 默认值在这里；空的覆盖项表示读取同目录 JSON。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
CONFIG_FILE="$SCRIPT_DIR/glm52mempool-opt.json"
RUN_ID=opt-001
LOG_DIR=""
ROLE=prefill
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
        --role) ROLE="$2"; shift 2 ;;
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
Usage: bash opt/glm52mempool-opt.sh --role prefill|decode [options]
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
if [[ "$ROLE" != prefill && "$ROLE" != decode ]]; then
    echo "--role must be prefill or decode" >&2
    exit 2
fi
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

# 公共环境：保留现有 CANN PYTHONPATH；原生 fake warmup 保持开启。
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export USE_VLLM_CUSTOM_ALLREDUCE=1
export ASCEND_MF_STORE_URL="tcp://$P_IP:24670"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1
export HCCL_BUFFSIZE=1024
export HCCL_SOCKET_IFNAME="$INTERFACE"
export GLOO_SOCKET_IFNAME="$INTERFACE"
unset ASCEND_LAUNCH_BLOCKING ASCEND_MF_TRANSFER_PROTOCOL SGLANG_ENABLE_SPEC_V2

# Mempool：Router 和 worker 都读取本轮生成的同一份最终配置。
export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
export SGLANG_NPU_ENABLE_MEMPOOL=1
export SGLANG_NPU_MEMPOOL_ROUTER_CONFIG="$CONFIG_FILE"
export SGLANG_NPU_MEMPOOL_ROUTER_RUN_ID="$RUN_ID"
export SGLANG_NPU_MEMPOOL_P_ALIVE_MAIN_PAGES="$P_ALIVE_PAGE"
export SGLANG_NPU_MEMPOOL_P_ALIVE_INDEX_PAGES="$P_INDEX_PAGE"
export SGLANG_NPU_MEMPOOL_D_ALIVE_MAIN_PAGES="$D_ALIVE_PAGE"
export SGLANG_NPU_MEMPOOL_P_PREFIX_PAGES="$PREFIX_PAGE"
export SGLANG_NPU_MEMPOOL_SHARE=0
export SGLANG_NPU_MEMPOOL_DIAGNOSTICS=0
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_USE_FIA_NZ=0
export SGLANG_NPU_SPARSE_KV_ENABLE_LRU=0
export SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR=2
export SGLANG_NPU_SPARSE_KV_PROBATION_AGE=4
unset SGLANG_NPU_MEMPOOL_TEST_CAPACITY SGLANG_NPU_MEMPOOL_EVIDENCE_DIR
unset MEMPOOL_ACCEPTANCE_CONCURRENCY MEMPOOL_RANK_NUMA_MAP

NUMA_ARGS=()
if [[ -n "$RANK_NUMA_MAP" ]]; then
    unset SGLANG_NPU_MEMPOOL_LOCAL_NUMA_NODE
    NUMA_ARGS=(--mempool-rank-numa-map "$RANK_NUMA_MAP")
else
    export SGLANG_NPU_MEMPOOL_LOCAL_NUMA_NODE="$NUMA_NODES"
fi

# 轻量计时与手动 profile 窗口；ranks/时长在 profiler.sh 中指定。
export SGLANG_PROFILE_V2=0
export SGLANG_PROFILE_WITH_STACK=0
export SGLANG_PROFILE_RECORD_SHAPES=0
export SGLANG_NPU_MEMPOOL_PERF_STEP_TIMING=1

if [[ "$ROLE" == prefill ]]; then
    export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
    export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
    export TASK_QUEUE_ENABLE=1

    run python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tp 16 \
        --base-gpu-id 0 \
        --gpu-id-step 1 \
        --trust-remote-code \
        --attention-backend ascend \
        --device npu \
        --quantization modelslim \
        --watchdog-timeout 9000 \
        --host "$P_IP" \
        --port "$P_PORT" \
        --mem-fraction-static 0.75 \
        --context-length "$CONTEXT_LENGTH" \
        --max-total-tokens "$MAX_TOTAL_TOKENS" \
        --page-size "$PAGE_SIZE" \
        --disable-radix-cache \
        --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
        --max-prefill-tokens "$PROMPT_CAPACITY" \
        --enable-dp-attention \
        --dp-size 1 \
        --enable-dp-lm-head \
        --max-running-requests "$MAX_RUNNING_REQUESTS" \
        --prefill-max-requests 1 \
        --disaggregation-transfer-backend ascend \
        --disaggregation-mode prefill \
        --disable-cuda-graph \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --nnodes 1 \
        --node-rank 0 \
        --moe-dense-tp-size 1 \
        --moe-a2a-backend deepep \
        --deepep-mode normal \
        --disable-shared-experts-fusion \
        --load-balance-method round_robin \
        --dtype bfloat16 \
        --kv-cache-dtype bfloat16 \
        --dist-init-addr "$P_IP:10000" \
        --mempool-prefill-host "$P_IP" \
        --mempool-bootstrap-port "$BOOTSTRAP_PORT" \
        --mempool-base-port 19000 \
        --mempool-pool-id 104 \
        --mempool-nic "tcp://$P_IP:25670" \
        --mempool-prefill-capacity "$PROMPT_CAPACITY" \
        --mempool-decode-capacity "$DECODE_STORAGE_CAPACITY" \
        --mempool-timeout "$STARTUP_TIMEOUT" \
        "${NUMA_ARGS[@]}"
else
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_NPU_USE_MULTI_STREAM=1
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
    export TASK_QUEUE_ENABLE=0

    run python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tp 16 \
        --base-gpu-id 0 \
        --gpu-id-step 1 \
        --trust-remote-code \
        --attention-backend ascend \
        --device npu \
        --quantization modelslim \
        --watchdog-timeout 9000 \
        --host "$D_IP" \
        --port "$D_PORT" \
        --mem-fraction-static 0.75 \
        --context-length "$CONTEXT_LENGTH" \
        --max-total-tokens "$MAX_TOTAL_TOKENS" \
        --page-size "$PAGE_SIZE" \
        --disable-radix-cache \
        --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
        --max-prefill-tokens "$PROMPT_CAPACITY" \
        --enable-dp-attention \
        --dp-size 1 \
        --enable-dp-lm-head \
        --max-running-requests "$MAX_RUNNING_REQUESTS" \
        --prefill-max-requests 1 \
        --disaggregation-transfer-backend ascend \
        --disaggregation-mode decode \
        --ep-size 16 \
        --disaggregation-decode-extra-slots 0 \
        --cuda-graph-bs-decode 16 \
        --nnodes 1 \
        --node-rank 0 \
        --moe-dense-tp-size 1 \
        --moe-a2a-backend deepep \
        --deepep-mode low_latency \
        --disable-shared-experts-fusion \
        --load-balance-method round_robin \
        --dtype bfloat16 \
        --kv-cache-dtype bfloat16 \
        --dist-init-addr "$D_IP:10000" \
        --mempool-prefill-host "$P_IP" \
        --mempool-bootstrap-port "$BOOTSTRAP_PORT" \
        --mempool-base-port 19000 \
        --mempool-pool-id 104 \
        --mempool-nic "tcp://$D_IP:25670" \
        --mempool-prefill-capacity "$PROMPT_CAPACITY" \
        --mempool-decode-capacity "$DECODE_STORAGE_CAPACITY" \
        --mempool-timeout "$STARTUP_TIMEOUT" \
        "${NUMA_ARGS[@]}"
fi
