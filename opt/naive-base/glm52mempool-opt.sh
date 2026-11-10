#!/usr/bin/env bash
set -eo pipefail

# 工具从当前性能分支加载，模型服务从旧版独立 checkout 加载。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
NAIVE_SGLANG_DIR=${NAIVE_SGLANG_DIR:-/home/cryang/sglang-naive-profile}
CONFIG_FILE=""
RUN_ID=naive-001
LOG_DIR=""
ROLE=prefill
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
        --role) ROLE="$2"; shift 2 ;;
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
Usage: bash opt/naive-base/glm52mempool-opt.sh [options]
  --role prefill|decode       服务角色（prepare不启动服务）
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
先在两台各执行prepare.sh。旧版不启动Main Router；请求工具直接派发到P/D。
HELP
            exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done
if [[ "$ROLE" != prefill && "$ROLE" != decode ]]; then
    echo "--role must be prefill or decode" >&2; exit 2
fi
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

printf 'mempool console=%s; full log: %s/%s.log\n' "$MEMPOOL_CONSOLE" "$RUN_DIR" "$ROLE"
export SGLANG_MEMPOOL_CONSOLE="$MEMPOOL_CONSOLE"
if [[ "$DRY_RUN" == 0 ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PATH="/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH"
    cd "$NAIVE_SGLANG_DIR"
fi
export PYTHONPATH="$NAIVE_SGLANG_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset ASCEND_LAUNCH_BLOCKING ASCEND_MF_TRANSFER_PROTOCOL SGLANG_ENABLE_SPEC_V2
unset MEMPOOL_ACCEPTANCE_CONCURRENCY MEMPOOL_RANK_NUMA_MAP
# 清除当前版本的Router/物理预算/观测环境，以下只设置旧版使用的开关。
for name in $(compgen -v SGLANG_NPU_MEMPOOL_); do unset "$name"; done
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
export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
export SGLANG_NPU_ENABLE_MEMPOOL=1
export SGLANG_NPU_MEMPOOL_SHARE=0
export SGLANG_NPU_MEMPOOL_LOCAL_NUMA_NODE="$NUMA_NODES"
export SGLANG_NPU_MEMPOOL_DIAGNOSTICS=0
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_USE_FIA_NZ=0
export SGLANG_NPU_SPARSE_KV_ENABLE_LRU=0
export SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR=2
export SGLANG_NPU_SPARSE_KV_PROBATION_AGE=4
export SGLANG_NPU_MEMPOOL_PROFILE_RAW=1
export SGLANG_PROFILE_V2=0
export SGLANG_PROFILE_WITH_STACK=0
export SGLANG_PROFILE_RECORD_SHAPES=0

run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'cd %q\n' "$NAIVE_SGLANG_DIR"
        printf '%q ' "$@"
        printf '| tee %q | python3 -u %q --console %q\n' "$RUN_DIR/$ROLE.log" "$TOOLS_DIR/mempool_console.py" "$MEMPOOL_CONSOLE"
    else
        {
            printf 'mempool_console=%s\n' "$MEMPOOL_CONSOLE"
            git rev-parse HEAD
            git diff --stat
            printf '%q ' "$@"
            printf '\n'
        } > "$RUN_DIR/$ROLE-launch.txt"
        python3 "$TOOLS_DIR/opt_naive_config.py" identity \
            --config "$CONFIG_FILE" --run-dir "$RUN_DIR" \
            --naive-sglang-dir "$NAIVE_SGLANG_DIR" --role "$ROLE"
        "$@" 2>&1 | tee "$RUN_DIR/$ROLE.log" | python3 -u "$TOOLS_DIR/mempool_console.py" --console "$MEMPOOL_CONSOLE"
    fi
}

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
        --mempool-decode-capacity "$DECODE_CAPACITY" \
        --mempool-timeout "$STARTUP_TIMEOUT"
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
        --mempool-decode-capacity "$DECODE_CAPACITY" \
        --mempool-timeout "$STARTUP_TIMEOUT"
fi
