# cpu高性能
  echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  sysctl -w vm.swappiness=0
  sysctl -w kernel.numa_balancing=0
  sysctl -w kernel.sched_migration_cost_ns=50000

  # 绑核
  export SGLANG_SET_CPU_AFFINITY=1

  # 设置PYTHONPATH
  cd /home/cryang_wx1511021/sglang
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
  export ASCEND_MF_STORE_URL="tcp://61.28.30.27:24670"
  export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

  export TRANSFORMERS_VERBOSITY=error

  # 关闭coredump
  export SGLANG_CUDA_COREDUMP_BEFORE_CRASH=0
  export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
  export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=0
  unset CUDA_COREDUMP_FILE
  unset CUDA_COREDUMP_PIPE

  # 先跑普通PD正确性，暂不开SparseKV
  export SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD=0

  # p节点IP
  P_IP=('61.28.30.27')
  # D节点IP D节点首节点IP
  D_IP=('61.28.30.28')

  MODEL_PATH=/home/cryang_wx1511021/GLM-5.1-w4a8

  LOCAL_HOST1='61.28.30.27'
  echo "${LOCAL_HOST1}"

  # prefill
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
          export HCCL_SOCKET_IFNAME=enp196s0f0
          export GLOO_SOCKET_IFNAME=enp196s0f0

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
          --context-length 16384 \
          --disable-radix-cache \
          --chunked-prefill-size -1 \
          --max-prefill-tokens 16384 \
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
          --dist-init-addr ${P_IP[0]}:10000
          break
      fi
  done

  # decode
  for i in "${!D_IP[@]}";
  do
      if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" ]];
      then
          echo "${D_IP[$i]}"

          export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
          export SGLANG_ENABLE_SPEC_V2=1
          export SGLANG_NPU_USE_MULTI_STREAM=1
          export HCCL_BUFFSIZE=650
          export TASK_QUEUE_ENABLE=0
          export HCCL_SOCKET_IFNAME=enp196s0f0
          export GLOO_SOCKET_IFNAME=enp196s0f0
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
          --context-length 16384 \
          --disable-radix-cache \
          --chunked-prefill-size -1 \
          --max-prefill-tokens 16384 \
          --enable-dp-attention \
          --dp-size 1 \
          --enable-dp-lm-head \
          --max-running-requests 16 \
          --prefill-max-requests 1 \
          --disaggregation-transfer-backend ascend \
          --disaggregation-mode decode \
          --disaggregation-decode-extra-slots 0 \
          --nnodes 1 --node-rank 0 \
          --prefill-round-robin-balance \
          --moe-a2a-backend deepep \
          --deepep-mode low_latency \
          --moe-dense-tp-size 1 \
          --disable-shared-experts-fusion \
          --load-balance-method round_robin \
          --dtype bfloat16 \
          --disable-cuda-graph \
          --dist-init-addr ${D_IP[0]}:10000
          break
      fi
  done

  exit 1