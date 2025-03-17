#!/bin/bash

model_list=(
    models/MoTCoder-7B-v1.5
)
name=cc-chat-template-queries
fix_name=cc-chat-template-queries-reflect_n5
cd /mnt/nas-alinlp/ljy/MoTCoder/
for model_id in "${model_list[@]}"; do
    results_path=$model_id/results/$name-results.jsonl
    python /mnt/nas-alinlp/ljy/MoTCoder/eval/eval.py \
        --data_path "$model_id/results/$name.jsonl" \
        --save_path "$model_id/results/$name-metrics.jsonl"  \
        --results_path $results_path \
        --levels test valid

    python /mnt/nas-alinlp/ljy/MoTCoder/eval/vllm_fix.py \
        --model_id "$model_id" \
        --data_path $results_path \
        --save_path "$model_id/results/$fix_name.jsonl" \
        --key prompt \
        --batch_size 1000 \
        --apply_chat_template \
        --n 5

    python /mnt/nas-alinlp/ljy/MoTCoder/eval/eval.py \
        --data_path "$model_id/results/$fix_name.jsonl" \
        --save_path "$model_id/results/$fix_name-metrics.jsonl" \
        --k_list 1 5 \
        --levels test valid
done

