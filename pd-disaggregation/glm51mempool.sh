#!/usr/bin/env bash
set -o pipefail

# 与 glm51dis.sh 一样：修改下面 P_IP、D_IP、LOCAL_HOST1、MODEL_PATH 和各段网卡名。
# 两侧运行同一版本 sglang（启动诊断版本：2261e35683），并使用相同的 mempool 容量/端口。
# 启动顺序：先 P，随后 D；P 等待 BM join 时就启动 D，不要等 P ready。
# 两台机器各自执行：bash pd-disaggregation/glm51mempool.sh
# 两侧服务 ready 后，另开终端启动下方 router，再执行请求测试和日志检查。
# 原 TransferEngine store 为 P:24670；mempool store 为 P:19000..19015。
# mempool NIC 预留每侧25670..25701；若 MF 网卡 IP 不同，在对应 --mempool-nic 处修改。
# 小容量对照：context=1024、S_P/S_D=512；整体 DRAM 仍同时容纳原 hostSHM 与 mempool。
# 按当前78层/16 slots/576维BF16估算：BM每rank 1GiB，每机16GiB；D原hostSHM约23.40GiB。
# 本轮通过后，可保留context=1024，仅把两侧S_P/S_D恢复4096，另存日志比较BM容量影响。

# cpu高性能
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  sysctl -w vm.swappiness=0
  sysctl -w kernel.numa_balancing=0
  sysctl -w kernel.sched_migration_cost_ns=50000

  # 绑核
  export SGLANG_SET_CPU_AFFINITY=1

  # 设置PYTHONPATH
  cd /home/cryang/sglang || exit 1
  export PYTHONPATH=${PWD}/python:$PYTHONPATH

  unset https_proxy
  unset http_proxy
  unset HTTPS_PROXY
  unset HTTP_PROXY
  unset ASCEND_LAUNCH_BLOCKING

  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh

  export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
  export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

  # 内存碎片
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export STREAMS_PER_DEVICE=32

  # pd传输, IP设置为p节点首节点
  export USE_VLLM_CUSTOM_ALLREDUCE=1
  export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
  unset ASCEND_MF_TRANSFER_PROTOCOL
  export ASCEND_MF_STORE_URL="tcp://10.120.72.31:24670"
  export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

  export TRANSFORMERS_VERBOSITY=error

  # 关闭coredump
  export SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0
  export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
  export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=0
  unset CUDA_COREDUMP_FILE
  unset CUDA_COREDUMP_PIPE

  # mempool shadow：保留原 sparse KV PD 路径，同时双写 mempool
  export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
  export SGLANG_NPU_ENABLE_MEMPOOL=1
  # BM 启动诊断：MF INFO、阶段耗时、每 15 秒等待状态与内存/栈快照。
  export SGLANG_NPU_MEMPOOL_DIAGNOSTICS=1
  export SGLANG_NPU_USE_MLAPO=0
  export PYTHONUNBUFFERED=1

  # p节点IP
  P_IP=('10.120.72.31')
  # D节点IP D节点首节点IP
  D_IP=('10.120.72.32')

  MODEL_PATH=/data_lib/data/models/GLM-5.1-w4a8

  # P 机器填 P_IP，D 机器填 D_IP；网卡名在下面各自的启动段修改
  LOCAL_HOST1='10.120.72.31'
  echo "${LOCAL_HOST1}"

  # 日志每轮覆盖；需要保留历史时修改 LOG_DIR
  LOG_DIR=/tmp/mempool-02-service-small
  mkdir -p "${LOG_DIR}"
  git rev-parse HEAD
  git diff --stat

  # prefill（eager；mempool 参数在本段命令末尾）
  for i in "${!P_IP[@]}";
  do
      if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" ]];
      then
          echo "${P_IP[$i]}"

          export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
          export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
          export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
          export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
          export HCCL_BUFFSIZE=1024
          export TASK_QUEUE_ENABLE=1
          export HCCL_SOCKET_IFNAME=bond4
          export GLOO_SOCKET_IFNAME=bond4

          python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
          --tp 16 \
          --base-gpu-id 0 \
          --gpu-id-step 1 \
          --trust-remote-code \
          --attention-backend ascend \
          --device npu \
          --quantization modelslim \
          --watchdog-timeout 9000 \
          --host ${P_IP[$i]} --port 8000 \
          --mem-fraction-static 0.75 \
          --context-length 1024 \
          --disable-radix-cache \
          --chunked-prefill-size -1 \
          --max-prefill-tokens 4096 \
          --enable-dp-attention \
          --dp-size 1 \
          --enable-dp-lm-head \
          --max-running-requests 16 \
          --prefill-max-requests 1 \
          --disaggregation-transfer-backend ascend \
          --disaggregation-mode prefill \
          --disable-cuda-graph \
          --nnodes 1 --node-rank 0 \
          --disaggregation-bootstrap-port 8995 \
          --moe-dense-tp-size 1 \
          --moe-a2a-backend deepep \
          --deepep-mode normal \
          --disable-shared-experts-fusion \
          --load-balance-method round_robin \
          --dtype bfloat16 \
          --dist-init-addr ${P_IP[0]}:10000 \
          --mempool-prefill-host ${P_IP[0]} \
          --mempool-bootstrap-port 8995 \
          --mempool-base-port 19000 \
          --mempool-pool-id 104 \
          --mempool-nic tcp://${P_IP[$i]}:25670 \
          --mempool-prefill-capacity 512 \
          --mempool-decode-capacity 512 \
          --mempool-timeout 600 \
          2>&1 | tee "${LOG_DIR}/p.log"
          exit $?
      fi
  done

  # decode（Graph BS=16；mempool 参数在本段命令末尾）
  for i in "${!D_IP[@]}";
  do
      if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" ]];
      then
          echo "${D_IP[$i]}"

          export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
          export SGLANG_ENABLE_SPEC_V2=1
          export SGLANG_NPU_USE_MULTI_STREAM=1
          export HCCL_BUFFSIZE=1024
          export TASK_QUEUE_ENABLE=0
          export HCCL_SOCKET_IFNAME=bond4
          export GLOO_SOCKET_IFNAME=bond4
          export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32

          python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
          --tp 16 \
          --ep-size 16 \
          --base-gpu-id 0 \
          --gpu-id-step 1 \
          --trust-remote-code \
          --attention-backend ascend \
          --device npu \
          --quantization modelslim \
          --watchdog-timeout 9000 \
          --host ${D_IP[$i]} --port 8001 \
          --mem-fraction-static 0.75 \
          --context-length 1024 \
          --disable-radix-cache \
          --chunked-prefill-size -1 \
          --max-prefill-tokens 4096 \
          --enable-dp-attention \
          --dp-size 1 \
          --enable-dp-lm-head \
          --max-running-requests 16 \
          --prefill-max-requests 1 \
          --disaggregation-transfer-backend ascend \
          --disaggregation-mode decode \
          --disaggregation-decode-extra-slots 0 \
          --nnodes 1 --node-rank 0 \
          --moe-a2a-backend deepep \
          --deepep-mode low_latency \
          --moe-dense-tp-size 1 \
          --disable-shared-experts-fusion \
          --load-balance-method round_robin \
          --dtype bfloat16 \
          --cuda-graph-bs-decode 16 \
          --dist-init-addr ${D_IP[0]}:10000 \
          --mempool-prefill-host ${P_IP[0]} \
          --mempool-bootstrap-port 8995 \
          --mempool-base-port 19000 \
          --mempool-pool-id 104 \
          --mempool-nic tcp://${D_IP[$i]}:25670 \
          --mempool-prefill-capacity 512 \
          --mempool-decode-capacity 512 \
          --mempool-timeout 600 \
          2>&1 | tee "${LOG_DIR}/d.log"
          exit $?
      fi
  done

  echo "LOCAL_HOST1=${LOCAL_HOST1} 未匹配 P_IP/D_IP，请修改为本机地址。" >&2
  exit 1

# ==================== 两侧 ready 后，在其他终端手动执行 ====================
# router（按实际 P/D IP 修改；8995 对应 P 的 bootstrap port）：
# python3 -m sglang_router.launch_router \
#     --pd-disaggregation --policy round_robin \
#     --prefill http://10.120.72.31:8000 8995 \
#     --decode http://10.120.72.32:8001 \
#     --host 127.0.0.1 --port 6699 --mini-lb
#
# 请求测试（在 sglang 仓库根目录执行，router 地址按实际修改）：
# cd /home/cryang/sglang
# python3 ascend-mempool-test/scripts/verify_shadow_service.py requests \
#     --url http://127.0.0.1:6699 --decode-tokens 32 --timeout 900 \
#     --output /tmp/mempool-02-service-small/requests.json
#
# 等待 RELEASE_ACK，把 P/D 的 p.log、d.log 放到同一台机器，再检查：
# python3 ascend-mempool-test/scripts/verify_shadow_service.py check-logs \
#     --prefill-logs /tmp/mempool-02-service-small/p.log \
#     --decode-logs /tmp/mempool-02-service-small/d.log \
#     --requests 3 --output /tmp/mempool-02-service-small/lifecycle.json
#
# 通过判据：REQUESTS_PASSED、SHADOW_LIFECYCLE_PASSED (no KV readback)。
# 同时检查 requests.json 中的生成文本。本轮只验收 shadow 服务，尚不做 top-k readback。
