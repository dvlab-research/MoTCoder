#!/bin/bash

# 定义多个模型目录
model_list=(
    models/MoTCoder-7B-v1.5
)
echo "$model_id/results/chat-template-queries.jsonl"

for model_id in "${model_list[@]}"; do
    echo "Processing $model_id..."

    python /mnt/nas-alinlp/ljy/MoTCoder/eval/vllm_gen.py \
        --model_id "$model_id" \
        --data_path /mnt/nas-alinlp/ljy/MoTCoder/data/prompts/queries.jsonl \
        --save_path "$model_id/results/chat-template-queries.jsonl" \
        --key prompt \
        --batch_size 1000 \
        --apply_chat_template

    python /mnt/nas-alinlp/ljy/MoTCoder/eval/eval.py \
        --data_path "$model_id/results/chat-template-queries.jsonl" \
        --save_path "$model_id/results/chat-template-queries-metrics.jsonl"  \
        --results_path "$model_id/results/chat-template-queries-results.jsonl" \

done
