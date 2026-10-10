python3 -m sglang.benchmark.serving \
    --backend sglang \
    --base-url http://127.0.0.1:6699 \
    --ready-check-timeout-sec 0 \
    --dataset-name random-ids \
    --tokenize-prompt \
    --random-input-len 130000\
    --random-output-len 1024\
    --random-range-ratio 1 \
    --num-prompts 3 \
    --max-concurrency 2 \
    --request-rate inf \
    --model /data_lib/data/models/GLM-5.1-w4a8
    # --model /home/cryang_wx1511021/GLM-5.1-w4a8
    # --model /home/caofei/DeepSeek-V3.2-Exp-w8a8

sgl-eval run aime26 \
    --base-url http://localhost:6699/v1 \
    --num-examples 8 \
    --n-repeats 1 \
    --max-tokens 28672 \
    --temperature 0 \
    --thinking \
    --num-threads 8