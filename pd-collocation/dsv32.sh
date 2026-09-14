#!/usr/bin/env bash

# Single-node PD mixed deployment for DeepSeek V3.2 on Ascend NPU.

# Host-level tuning. These commands usually require root and are better run once
# during machine preparation rather than on every server restart.
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# Use the source checkout instead of an installed sglang package.
SGLANG_DIR=/home/cryang_wx1511021/sglang
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

# W8A8 ModelSlim quantized DeepSeek V3.2 path.
MODEL_PATH=/home/cryang_wx1511021/DeepSeek-V3.2-Exp-w8a8

# -----------------------------------------------------------------------------
# Performance tuning
# -----------------------------------------------------------------------------
# Pin SGLang CPU workers, reduce NPU memory fragmentation, and increase available
# NPU streams. The scheduler knobs are local Ascend/SGLang-fork tuning items;
# verify they are still read by the target checkout when changing branches.
export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
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
# DEEP_NORMAL_MODE_USE_INT8_QUANT is needed for the W8A8 dispatch path. DeepEP
# token limits and long-sequence options only matter when the MoE/DeepEP backend
# reads them; keep long-sequence candidates disabled until validated by benchmark.
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
#export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
#export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

# Enable NPU dual-stream MoE execution for shared experts and routed experts.
# Leave disabled unless benchmark confirms stability and throughput gain.
#export SGLANG_NPU_USE_MULTI_STREAM=1

# -----------------------------------------------------------------------------
# Speculative decoding and overlap scheduling
# -----------------------------------------------------------------------------
# SpecV2 enables the overlap scheduler for speculative/MTP-style paths. The plan
# stream overlaps planning work with execution when the corresponding model path
# and launch arguments activate speculative decoding.
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# -----------------------------------------------------------------------------
# MLA attention preprocessing and FIA NZ layout
# -----------------------------------------------------------------------------
# MLAPO fuses MLA preprocessing for DeepSeek-style MLA models. FIA NZ changes KV
# cache layout for FIA and must be enabled together with MLAPO.
#export SGLANG_NPU_USE_MLAPO=1
#export SGLANG_USE_FIA_NZ=1

# -----------------------------------------------------------------------------
# Sparse KV/offload
# -----------------------------------------------------------------------------
# Core switch for sparsekv experiments.
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
# if this Ascend path requires the DP code path. For DeepSeek-V3.2-Exp, also
# verify whether the upstream tool parser should be deepseekv31 or deepseekv32.
python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
--tp 16 \
--trust-remote-code \
--attention-backend ascend \
--device npu \
--quantization modelslim \
--watchdog-timeout 9000 \
--host 127.0.0.1 --port 6699 \
--mem-fraction-static 0.8 \
--max-running-requests 16 \
--context-length 65536  --disable-radix-cache --chunked-prefill-size 4096 \
--enable-dp-attention --dp-size 1 --enable-dp-lm-head \
--cuda-graph-bs 16 \
--reasoning-parser deepseek-v3 \
--tool-call-parser deepseekv32 \
--dtype bfloat16 # 2>&1 | tee /home/cryang_wx1511021/testlauch.log
