#!/usr/bin/env bash
# GLM-5.1 / TP16 / 1P1D mempool shadow gate，基于同目录 glm51dis.sh。
# 原 sparse KV offload、main-KV/Index K/metadata 传输继续运行，额外双写 mempool。
# 启动顺序：P -> 随后启动 D（不要等 P ready）-> 两侧 ready 后 router -> test -> check。
# 真实 NPU 验收仍由用户执行；本脚本不把服务输出通过当作 mempool KV readback 通过。
set -eo pipefail

# 两台机器应使用相同版本的 sglang，以及相同 P_IP/D_IP、容量、pool/store 参数。
# 所有配置都支持环境变量覆盖，例如：
# P_IP=10.0.0.31 D_IP=10.0.0.32 HCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
#   bash pd-disaggregation/glm51mempool.sh p
SGLANG_DIR=${SGLANG_DIR:-/home/cryang/sglang}
MODEL_PATH=${MODEL_PATH:-/data_lib/data/models/GLM-5.1-w4a8}
PYTHON_BIN=${PYTHON_BIN:-python3}
P_IP=${P_IP:-10.120.72.31}
D_IP=${D_IP:-10.120.72.32}
P_PORT=${P_PORT:-8000}
D_PORT=${D_PORT:-8001}
P_BOOTSTRAP_PORT=${P_BOOTSTRAP_PORT:-8995}
ROUTER_HOST=${ROUTER_HOST:-127.0.0.1}
ROUTER_PORT=${ROUTER_PORT:-6699}
ROUTER_URL=${ROUTER_URL:-http://127.0.0.1:${ROUTER_PORT}}
LOG_DIR=${LOG_DIR:-/tmp/mempool-02-service}

# 较原样例缩小 token 预算；实际 DRAM 同时容纳原 hostSHM 和新增 mempool。
CONTEXT_LENGTH=${CONTEXT_LENGTH:-16384}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
MEMPOOL_SP=${MEMPOOL_SP:-8192}
MEMPOOL_SD=${MEMPOOL_SD:-8192}
MEMPOOL_BASE_PORT=${MEMPOOL_BASE_PORT:-19000}
MEMPOOL_NIC_PORT=${MEMPOOL_NIC_PORT:-25670}
MEMPOOL_POOL_ID=${MEMPOOL_POOL_ID:-104}
MEMPOOL_TIMEOUT=${MEMPOOL_TIMEOUT:-600}
# 服务地址与 MF 网卡地址不同时，分别覆盖以下两个值。
P_MEMPOOL_IP=${P_MEMPOOL_IP:-$P_IP}
D_MEMPOOL_IP=${D_MEMPOOL_IP:-$D_IP}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
ATB_ENV=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
# wheel 已能正常加载时可留空；需要 source MF 时填写其实际 set_env.sh 路径。
MF_ENV=${MF_ENV:-}

usage() {
    # 在 Mac 上查看说明不会 source Ascend 环境或启动模型。
    cat <<'EOF'
用法：bash pd-disaggregation/glm51mempool.sh {p|d|router|test|check|help} [附加参数]

在 ascend-sglang-script 库内执行，分别使用不同终端：
  1. P 机器：bash pd-disaggregation/glm51mempool.sh p
  2. D 机器：bash pd-disaggregation/glm51mempool.sh d
     P 在 BM join 等待 D 是正常的；不要等 P ready 才启动 D。
  3. 两侧服务 ready 后，在 router 机器：
     bash pd-disaggregation/glm51mempool.sh router
  4. router ready 后，在 router 机器：
     bash pd-disaggregation/glm51mempool.sh test
  5. 等两侧完成 RELEASE_ACK，把 P/D 日志放在同一台机器：
     bash pd-disaggregation/glm51mempool.sh check \
       --prefill-logs /tmp/mempool-02-service/p.log \
       --decode-logs /tmp/mempool-02-service/d.log

默认服务器：P=10.120.72.31:8000，D=10.120.72.32:8001，P bootstrap=8995。
默认 router：127.0.0.1:6699；远端请求请同时设置 ROUTER_HOST 和客户端 ROUTER_URL。
常用覆盖项：SGLANG_DIR、MODEL_PATH、P_IP、D_IP、HCCL_SOCKET_IFNAME、GLOO_SOCKET_IFNAME、
P_MEMPOOL_IP、D_MEMPOOL_IP、MEMPOOL_SP、MEMPOOL_SD、CONTEXT_LENGTH、LOG_DIR。

保留原 TransferEngine store 默认 P_IP:24670。
mempool store 使用 P_IP:19000..19015；NIC 预留每侧25670..25701。
16 对 BM 各有 local rank 0/1，由服务按 TP rank 自动配对，不需要起16份脚本。
P 使用 eager；D 使用 cuda-graph-bs-decode=16；两侧 runtime 均先映射 BM 再 capture。
保持 MLAPO=0、不启用 draft/prefix reuse；本轮仍为 shadow 双写。

日志和结果默认写入 /tmp/mempool-02-service：
  p.log / d.log / router.log / requests.json / lifecycle.json
每次启动会覆盖同名日志；要保留上一轮，可换一个 LOG_DIR。
test 预期 REQUESTS_PASSED；check 预期 SHADOW_LIFECYCLE_PASSED (no KV readback)。
还需人工检查 requests.json 的文本；此处不执行 top-k readback 或 AIME 精度测试。

p/d 附加参数会追加到 launch_server；例如：
  bash pd-disaggregation/glm51mempool.sh p --max-total-tokens 16384
test/check 附加参数会追加到 verify_shadow_service.py 的对应子命令。
EOF
}

setup_python() {
    # router/验证只需要现有 Python 环境，不加载 NPU 动态库。
    if [[ ! -d "$SGLANG_DIR/python/sglang" ]]; then
        printf '找不到 SGLang checkout：%s；请设置 SGLANG_DIR。\n' "$SGLANG_DIR" >&2
        exit 1
    fi
    cd "$SGLANG_DIR"
    export PYTHONPATH="$SGLANG_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
    mkdir -p "$LOG_DIR"
}

setup_npu() {
    # 沿用已跑通脚本的 CANN/ATB、绑核与 allocator 配置。
    source "$CANN_ENV"
    source "$ATB_ENV"
    if [[ -n "$MF_ENV" ]]; then
        source "$MF_ENV"
    fi
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PATH="/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH"
    export SGLANG_SET_CPU_AFFINITY=${SGLANG_SET_CPU_AFFINITY:-1}
    export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
    export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
    export USE_VLLM_CUSTOM_ALLREDUCE=${USE_VLLM_CUSTOM_ALLREDUCE:-1}
    export ASCEND_MF_STORE_URL=${ASCEND_MF_STORE_URL:-tcp://${P_IP}:24670}
    # 原样例最终 unset 该变量；需要其他已验证协议时可通过环境显式设置。
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-600}
    export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
    export SGLANG_NPU_ENABLE_MEMPOOL=1
    export SGLANG_NPU_USE_MLAPO=0
    export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-bond4}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond4}
    export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
    export TRANSFORMERS_VERBOSITY=error
    export SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0
    export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
    export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=0
    unset CUDA_COREDUMP_FILE CUDA_COREDUMP_PIPE ASCEND_LAUNCH_BLOCKING
}

launch_worker() {
    # 一条命令启动本侧16个TP workers；角色相关参数仅在这里分开。
    local role=$1
    shift
    local local_ip local_mf_ip service_port
    local -a role_args common_args command
    if [[ "$role" == p ]]; then
        local_ip=$P_IP
        local_mf_ip=$P_MEMPOOL_IP
        service_port=$P_PORT
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=${DEEPEP_NORMAL_LONG_SEQ_ROUND:-72}
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-1024}
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=${DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ:-1}
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}
        export TASK_QUEUE_ENABLE=1
        role_args=(--disaggregation-mode prefill --disable-cuda-graph
            --disaggregation-bootstrap-port "$P_BOOTSTRAP_PORT" --deepep-mode normal)
    else
        local_ip=$D_IP
        local_mf_ip=$D_MEMPOOL_IP
        service_port=$D_PORT
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_NPU_USE_MULTI_STREAM=1
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
        export TASK_QUEUE_ENABLE=0
        role_args=(--disaggregation-mode decode --ep-size 16
            --disaggregation-decode-extra-slots 0 --deepep-mode low_latency
            --cuda-graph-bs-decode 16)
    fi
    common_args=(
        --model-path "$MODEL_PATH" --tp 16 --base-gpu-id 0 --gpu-id-step 1
        --trust-remote-code --attention-backend ascend --device npu
        --quantization modelslim --dtype bfloat16 --watchdog-timeout 9000
        --host "$local_ip" --port "$service_port"
        --mem-fraction-static "$MEM_FRACTION_STATIC"
        --context-length "$CONTEXT_LENGTH" --max-prefill-tokens "$MAX_PREFILL_TOKENS"
        --disable-radix-cache --chunked-prefill-size -1
        --enable-dp-attention --dp-size 1 --enable-dp-lm-head
        --max-running-requests 16 --prefill-max-requests 1
        --disaggregation-transfer-backend ascend --nnodes 1 --node-rank 0
        --moe-dense-tp-size 1 --moe-a2a-backend deepep
        --disable-shared-experts-fusion --load-balance-method round_robin
        --dist-init-addr "${local_ip}:10000"
        --mempool-prefill-host "$P_IP" --mempool-bootstrap-port "$P_BOOTSTRAP_PORT"
        --mempool-base-port "$MEMPOOL_BASE_PORT" --mempool-pool-id "$MEMPOOL_POOL_ID"
        --mempool-nic "tcp://${local_mf_ip}:${MEMPOOL_NIC_PORT}"
        --mempool-prefill-capacity "$MEMPOOL_SP" --mempool-decode-capacity "$MEMPOOL_SD"
        --mempool-timeout "$MEMPOOL_TIMEOUT"
    )
    command=("$PYTHON_BIN" -m sglang.launch_server "${common_args[@]}" "${role_args[@]}" "$@")
    {
        git rev-parse HEAD
        git diff --stat
        printf '[%s] HCCL=%s GLOO=%s MF_STORE=%s\n' "$role" "$HCCL_SOCKET_IFNAME" "$GLOO_SOCKET_IFNAME" "$ASCEND_MF_STORE_URL"
        printf 'Command: '
        printf '%q ' "${command[@]}"
        printf '\n'
        "${command[@]}"
    } 2>&1 | tee "$LOG_DIR/$role.log"
}

# Explicit role selection avoids a hard-coded LOCAL_HOST1 silently launching the wrong side.
action=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
    help|-h|--help) usage ;;
    p|d)
        setup_python
        setup_npu
        launch_worker "$action" "$@"
        ;;
    router)
        setup_python
        "$PYTHON_BIN" -m sglang_router.launch_router \
            --pd-disaggregation --policy round_robin \
            --prefill "http://${P_IP}:${P_PORT}" "$P_BOOTSTRAP_PORT" \
            --decode "http://${D_IP}:${D_PORT}" \
            --host "$ROUTER_HOST" --port "$ROUTER_PORT" --mini-lb "$@" \
            2>&1 | tee "$LOG_DIR/router.log"
        ;;
    test)
        setup_python
        "$PYTHON_BIN" ascend-mempool-test/scripts/verify_shadow_service.py requests \
            --url "$ROUTER_URL" --decode-tokens 32 --timeout 900 \
            --output "$LOG_DIR/requests.json" "$@"
        ;;
    check)
        setup_python
        "$PYTHON_BIN" ascend-mempool-test/scripts/verify_shadow_service.py check-logs \
            --prefill-logs "$LOG_DIR/p.log" --decode-logs "$LOG_DIR/d.log" \
            --requests 3 --output "$LOG_DIR/lifecycle.json" "$@"
        ;;
    *) usage >&2; exit 2 ;;
esac
