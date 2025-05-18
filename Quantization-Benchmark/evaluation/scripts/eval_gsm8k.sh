#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -x

dataset_name=gsm8k

mkdir -p ./results/$dataset_name
mkdir -p ./logs/$dataset_name

models=(
    "Qwen2.5-7B-Instruct-gptq-W4A16-gsm8k"
    "Qwen2.5-7B-Instruct-obs-W4A16-gsm8k"
    "Qwen2.5-7B-Instruct-shapley-W4A16-gsm8k"
    "Qwen2.5-7B-Instruct-gptq-W8A8-gsm8k"
    "Qwen2.5-7B-Instruct-obs-W8A8-gsm8k"
    "Qwen2.5-7B-Instruct-shapley-W8A8-gsm8k"
    "Qwen2.5-7B-Instruct-gptq-FP8-gsm8k"
    "Qwen2.5-7B-Instruct-obs-FP8-gsm8k"
    "Qwen2.5-7B-Instruct-shapley-W8A8-gsm8k"
    "Qwen2.5-14B-Instruct-gptq-W4A16-gsm8k"
    "Qwen2.5-14B-Instruct-obs-W4A16-gsm8k"
    "Qwen2.5-14B-Instruct-shapley-W4A16-gsm8k"
    "Qwen2.5-14B-Instruct-gptq-W8A8-gsm8k"
    "Qwen2.5-14B-Instruct-obs-W8A8-gsm8k"
    "Qwen2.5-14B-Instruct-shapley-W8A8-gsm8k"
    "Qwen2.5-14B-Instruct-gptq-FP8-gsm8k"
    "Qwen2.5-14B-Instruct-obs-FP8-gsm8k"
    "Qwen2.5-14B-Instruct-shapley-FP8-gsm8k"
)

echo "Starting generation phase for all models..."
for i in "${!models[@]}"; do
    model_name="${models[$i]}"
    gpu_id=$i
    output_path="./results/$dataset_name/$model_name.parquet"
    log_dir="./logs/$dataset_name/$model_name"
    mkdir -p "$log_dir"

    if [ -d "/data/xiaobei/gjr/$model_name" ]; then
        echo "Launching generation for $model_name on GPU $gpu_id …"
        (
          CUDA_VISIBLE_DEVICES=$gpu_id python evaluator/generation.py \
              data.output_path="$output_path" \
              data.path="./data/$dataset_name/test.parquet" \
              model.path="/data/xiaobei/gjr/$model_name" \
              > "$log_dir/generation.log" 2>&1 &&
          echo "✅ Done generation: $model_name"
        ) &   # <-- 放到后台执行
    else
        echo "Model not found: $model_name, skipping."
    fi
done

# 等所有 generation 完成
wait
echo "All generation jobs finished."
echo "----------------------------------------"

echo "Starting evaluation phase for all models..."
for i in "${!models[@]}"; do
    model_name="${models[$i]}"
    gpu_id=$i
    output_path="./results/$dataset_name/$model_name.parquet"
    log_dir="./logs/$dataset_name/$model_name"

    if [ -f "$output_path" ]; then
        echo "Launching evaluation for $model_name on GPU $gpu_id …"
        (
          CUDA_VISIBLE_DEVICES=$gpu_id python evaluator/evaluation.py \
              data.path="$output_path" \
              > "$log_dir/evaluation.log" 2>&1 &&
          echo "✅ Done evaluation: $model_name"
        ) &   # <-- 放到后台执行
    else
        echo "No output file for $model_name, skipping evaluation."
    fi
done

# 等所有 evaluation 完成
wait
echo "All evaluation jobs finished."