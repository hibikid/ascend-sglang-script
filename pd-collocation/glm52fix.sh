#!/usr/bin/env bash
set -euo pipefail

# Single-node collocation deployment for GLM-5.2 W8A8 on Ascend NPU.

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# Keep your local source checkout.
SGLANG_DIR=/home/cryang/sglang
cd "${SGLANG_DIR}"
export PYTHONPATH=${PWD}/python:${PYTHONPATH:-}

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PATH=/usr/local/Ascend/9.0.0/compiler/bishengir/bin:$PATH

# Keep your data path.
MODEL_PATH=/mnt/raid/user/data/models/glm-q

export SGLANG_SET_CPU_AFFINITY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1

# Match baseline: do not enable sparse KV/offload or forced DSA indexer mode here.
unset SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD
unset SGLANG_DISABLE_DSA_INDEXER_FUSION

export SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=0

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name glm-5.2 \
  --device npu \
  --attention-backend ascend \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --quantization modelslim \
  --dtype bfloat16 \
  --base-gpu-id 0 \
  --tp-size 16 \
  --nnodes 1 \
  --node-rank 0 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 131072 \
  --context-length 131072 \
  --cuda-graph-bs 16 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --weight-loader-prefetch-checkpoints