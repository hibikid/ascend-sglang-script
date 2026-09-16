python3 -m sglang.benchmark.serving \
    --backend sglang \
    --base-url http://127.0.0.1:6699 \
    --ready-check-timeout-sec 0 \
    --dataset-name random-ids \
    --tokenize-prompt \
    --random-input-len 50000\
    --random-output-len 1024\
    --random-range-ratio 1 \
    --num-prompts 32 \
    --max-concurrency 16 \
    --request-rate inf \
    --model /mnt/raid/user/data/models/GLM-5.1-w4a8
    # --model /home/cryang_wx1511021/GLM-5.1-w4a8
    # --model /home/caofei/DeepSeek-V3.2-Exp-w8a8