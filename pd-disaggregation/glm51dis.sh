#!/usr/bin/env bash

# Two-node PD disaggregation for DeepSeek V3.2 W8A8 on Ascend NPU.
# The same script runs on both nodes: the node matching P_IP starts the prefill
# engine, the node matching D_IP starts the decode engine. The router lives in
# router.sh.

# Host-level tuning. These commands usually require root and are better run once
# during machine preparation rather than on every server restart.
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# Use the source checkout instead of an installed sglang package.
SGLANG_DIR=/home/cryang/sglang
cd "${SGLANG_DIR}"
export PYTHONPATH=${PWD}/python:${PYTHONPATH:-}

# Avoid inheriting proxy settings for local model serving and intra-node calls.
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY

# Disable synchronous Ascend debug mode for normal performance runs.
unset ASCEND_LAUNCH_BLOCKING

# Ascend runtime and ATB environment.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
export PATH=/usr/local/Ascend/9.0.0/compiler/bishengir/bin:$PATH

# W8A8 ModelSlim quantized DeepSeek V3.2 path.
MODEL_PATH=/mnt/raid/user/data/models/GLM-5.1-w4a8

# Cluster layout. LOCAL_HOST is this node's own IP; it must match one of the
# arrays below for this script to launch an engine.
P_IP=('10.120.72.23')
D_IP=('10.120.72.25')
LOCAL_HOST='10.120.72.23'

# -----------------------------------------------------------------------------
# Performance tuning
# -----------------------------------------------------------------------------
# Pin SGLang CPU workers, reduce NPU memory fragmentation, and increase available
# NPU streams.
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# -----------------------------------------------------------------------------
# KV transfer between prefill and decode
# -----------------------------------------------------------------------------
# ASCEND_MF_STORE_URL is the rendezvous endpoint for the transfer metadata
# store; it must point at the first prefill node. device_rdma is required on A2
# hardware; on A3 the default protocol is used and this stays commented.
export ASCEND_MF_STORE_URL="tcp://${P_IP[0]}:24670"
#export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"

# -----------------------------------------------------------------------------
# Communication
# -----------------------------------------------------------------------------
# Real NIC for cross-node HCCL/Gloo traffic. On these hosts the 10.120.72.x
# address lives on bond4 (enp196s0f0/f1 are bond slaves with no IP of their
# own), and Gloo needs the interface that owns the IP.
export HCCL_SOCKET_IFNAME=bond4
export GLOO_SOCKET_IFNAME=bond4

# -----------------------------------------------------------------------------
# Sparse KV/offload
# -----------------------------------------------------------------------------
# Core switch for sparsekv experiments.
export SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD=0

# -----------------------------------------------------------------------------
# Debug and profiling
# -----------------------------------------------------------------------------
# CUDA coredump switches are not expected to affect NPU execution, but are kept
# unchanged for now in case the local fork checks them in shared crash handling.
export SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=0

# -----------------------------------------------------------------------------
# Prefill engine
# -----------------------------------------------------------------------------
for i in "${!P_IP[@]}"; do
    if [[ "${LOCAL_HOST}" == "${P_IP[$i]}" ]]; then
        echo "launching prefill on ${P_IP[$i]}"

        # INT8 dispatch for the W8A8 MoE path, and the task queue mode used by
        # the Ascend prefill best practice.
        export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
        export TASK_QUEUE_ENABLE=2
        export HCCL_BUFFSIZE=900

        sglang serve --model-path ${MODEL_PATH} \
        --tp 16 \
        --base-gpu-id 0 \
        --gpu-id-step 1 \
        --trust-remote-code \
        --attention-backend ascend \
        --device npu \
        --watchdog-timeout 9000 \
        --host ${P_IP[$i]} --port 8000 \
        --mem-fraction-static 0.80 \
        --context-length 64000 \
        --disable-radix-cache --chunked-prefill-size -1 --max-prefill-tokens 64000 \
        --max-running-requests 16 \
        --prefill-max-requests 1 \
        --quantization modelslim \
        --disaggregation-transfer-backend ascend \
        --disaggregation-mode prefill \
        --disable-cuda-graph \
        --nnodes 1 --node-rank 0 \
        --disaggregation-bootstrap-port 8995 \
        --moe-dense-tp-size 1 \
        --reasoning-parser glm45 \
        --tool-call-parser glm47 \
        --moe-a2a-backend deepep \
        --deepep-mode auto \
        --dtype bfloat16 \
        --dist-init-addr ${P_IP[0]}:10000
        exit 0
    fi
done

# -----------------------------------------------------------------------------
# Decode engine
# -----------------------------------------------------------------------------
for i in "${!D_IP[@]}"; do
    if [[ "${LOCAL_HOST}" == "${D_IP[$i]}" ]]; then
        echo "launching decode on ${D_IP[$i]}"

        # Decode-side scheduling and MoE dispatch tuning from the Ascend best
        # practice. TASK_QUEUE_ENABLE=0 and skipping the scheduler all-gather
        # reduce decode-path overhead.
        export TASK_QUEUE_ENABLE=0
        export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
        export HCCL_BUFFSIZE=900
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

        # Overlap scheduling for speculative/MTP paths. SPEC_V2 is removed in
        # current SGLang (V2 worker is the default); only the plan-stream and
        # reflow switches remain, kept disabled until spec flags are enabled.
        #export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        #export SGLANG_SPEC_ENABLE_OVERLAP_REFLOW=1

        sglang serve --model-path ${MODEL_PATH} \
        --tp 16 \
        --dp 1 \
        --base-gpu-id 0 \
        --gpu-id-step 1 \
        --trust-remote-code \
        --attention-backend ascend \
        --device npu \
        --watchdog-timeout 9000 \
        --host ${D_IP[$i]} --port 8001 \
        --mem-fraction-static 0.80 \
        --context-length 64000 \
        --disable-radix-cache \
        --chunked-prefill-size -1 --max-prefill-tokens 64000 \
        --max-running-requests 16 \
        --prefill-max-requests 1 \
        --cuda-graph-bs-decode 16 \
        --quantization modelslim \
        --disaggregation-transfer-backend ascend \
        --disaggregation-mode decode \
        --disaggregation-decode-extra-slots 0 \
        --nnodes 1 --node-rank 0 \
        --prefill-round-robin-balance \
        --reasoning-parser glm45 \
        --tool-call-parser glm47 \
        --moe-a2a-backend deepep \
        --deepep-mode auto \
        --dtype bfloat16 \
        --dist-init-addr ${D_IP[0]}:10000
        exit 0
    fi
done

echo "LOCAL_HOST ${LOCAL_HOST} matches neither P_IP nor D_IP; nothing launched."
exit 1

# -----------------------------------------------------------------------------
# Optional NEXTN speculative decoding (decode engine)
# -----------------------------------------------------------------------------
# From the Ascend dsv32 PD-disaggregation best practice. When enabling, also
# enable SGLANG_ENABLE_OVERLAP_PLAN_STREAM / SGLANG_SPEC_ENABLE_OVERLAP_REFLOW
# above and retune SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.
# --speculative-algorithm NEXTN \
# --speculative-num-steps 3 \
# --speculative-eagle-topk 1 \
# --speculative-num-draft-tokens 4

# Shorter NEXTN candidate.
# --speculative-algorithm NEXTN --speculative-num-steps 1 \
# --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
