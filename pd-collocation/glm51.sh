#!/usr/bin/env bash

# Single-node PD mixed deployment for GLM-5.1 W4A8 on Ascend NPU.

# Optional stale shared-memory cleanup before a fresh run. Do not enable while
# other jobs are running on the same host.
# ipcs -m | awk '$1=="0x00000000" && $6==0 {print $2}' | xargs -r -n1 ipcrm -m
# rm -rf /dev/shm/*

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
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

# W4A8 ModelSlim quantized GLM-5.1 path.
MODEL_PATH=/mnt/raid/user/data/models/GLM-5.1-w4a8

# -----------------------------------------------------------------------------
# Performance tuning
# -----------------------------------------------------------------------------
# Pin SGLang CPU workers, reduce NPU memory fragmentation, and increase available
# NPU streams. Prefill delay tuning is passed as launch arguments below because
# the old scheduler environment variables are deprecated.
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
#export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

# -----------------------------------------------------------------------------
# Communication
# -----------------------------------------------------------------------------
# Single-node TP uses loopback for process-group communication. Change these to
# a real NIC for multi-node serving.
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_BUFFSIZE=900

# -----------------------------------------------------------------------------
# Quantization and DeepEP/MoE
# -----------------------------------------------------------------------------
# DEEP_NORMAL_MODE_USE_INT8_QUANT is needed for the W4A8 dispatch path in this
# Ascend fork. DeepEP token limits only matter when the MoE/DeepEP backend reads
# them; retune the dispatch-token cap when changing concurrency or spec length.
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export SGLANG_DISABLE_DSA_INDEXER_FUSION=1
#export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
#export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

# Enable NPU dual-stream MoE execution for shared experts and routed experts.
# Leave disabled unless benchmark confirms stability and throughput gain.
#export SGLANG_NPU_USE_MULTI_STREAM=1

# -----------------------------------------------------------------------------
# Speculative decoding and overlap scheduling
# -----------------------------------------------------------------------------
# SGLANG_ENABLE_SPEC_V2 has been removed in current SGLang: speculative decoding
# uses the V2 worker by default. Plan-stream/reflow overlap is useful for GLM-5.1
# NEXTN runs, but keep it disabled until the speculative CLI flags below are used.
#export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
#export SGLANG_SPEC_ENABLE_OVERLAP_REFLOW=1

# -----------------------------------------------------------------------------
# MLA attention preprocessing and FIA NZ layout
# -----------------------------------------------------------------------------
# MLAPO fuses MLA preprocessing for GLM/DeepSeek DSA-family models. FIA NZ changes
# KV cache layout for FIA and must be enabled together with MLAPO.
#export SGLANG_NPU_USE_MLAPO=1
#export SGLANG_USE_FIA_NZ=1

# -----------------------------------------------------------------------------
# Sparse KV/offload
# -----------------------------------------------------------------------------
# Core switch for sparsekv experiments. The local SGLang hook supports this for
# DSA models such as DeepSeek V3.2 and GLM-5.
export SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD=1

# -----------------------------------------------------------------------------
# Debug and profiling
# -----------------------------------------------------------------------------
# Optional NPU profiling; keep disabled for normal serving.
#export SGLANG_NPU_PROFILING=1
#export SGLANG_NPU_PROFILING_BS=14
#export SGLANG_NPU_PROFILING_STEP=10

# CUDA coredump switches are not expected to affect NPU execution, but are kept
# unchanged for now in case the local fork checks them in shared crash handling.
export SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=0

# NOTE: dp-size=1 does not provide data-parallel scaling; keep DP attention only
# if this Ascend path requires the DP code path.
sglang serve --model-path ${MODEL_PATH} \
--tp 16 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--quantization modelslim \
--watchdog-timeout 9000 \
--host 127.0.0.1 --port 6699 \
--mem-fraction-static 0.75 \ 
--max-running-requests 16 \
--enable-prefill-delayer --prefill-delayer-max-delay-passes 100 \
--context-length 16384 --disable-radix-cache --chunked-prefill-size 4096 \
--enable-dp-attention --dp-size 1 --enable-dp-lm-head \
--cuda-graph-bs-decode 16 \
--reasoning-parser glm45 \
--tool-call-parser glm47 \
--moe-a2a-backend deepep \
--cuda-graph-backend-prefill disabled \
--deepep-mode auto \
--dtype bfloat16

# -----------------------------------------------------------------------------
# Optional GLM-5.1 speculative/DeepEP variants
# -----------------------------------------------------------------------------
# Candidate from GLM-5.1 data. If enabled, also consider enabling the overlap
# envs above and retuning SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.
# --prefill-max-requests 1 \
# --served-model-name glm-5 \
# --speculative-draft-model-quantization unquant \
# --load-balance-method round_robin \
# --speculative-algorithm NEXTN \
# --speculative-num-steps 2 \
# --speculative-eagle-topk 1 \
# --speculative-num-draft-tokens 3