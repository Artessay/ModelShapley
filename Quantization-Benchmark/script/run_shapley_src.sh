#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

# Get alpha and GPU ID from command line arguments
if [ -z "$1" ]; then
    echo "Error: Alpha value not provided"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Error: GPU ID not provided"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$2
ALPHA=("$1")

MODEL_SIZES=("7B" "14B")
BASELINES=("shapley")
DATASETS=("gsm8k")
PRESET_SCHEMES=("W4A16" "W8A8" "FP8")
NUM_CALIBRATION_SAMPLES=("512")

BASE_MODEL_PATH="/data/xiaobei/gjr/Qwen2.5"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="result-time.log"

run_quantization() {
    local model_size=$1
    local scheme=$2
    local dataset=$3
    local baseline=$4
    
    local model_path="${BASE_MODEL_PATH}-${model_size}-Instruct"
    local script="quantize_by_${baseline}.py"
    
    echo "Running ${script} with:"
    echo "Model: ${model_path}"
    echo "Scheme: ${scheme}"
    echo "Dataset: ${dataset}"
    echo "Num Calibration Sampls: ${num_calibration_samples}"
    echo "----------------------------------------"
    
    python ${script} \
        --model_path "${model_path}" \
        --scheme "${scheme}" \
        --dataset "${dataset}" \
        --num_calibration_samples "${num_calibration_samples}" \
        --alpha "${alpha}"
}

for model_size in "${MODEL_SIZES[@]}"; do
    for scheme in "${PRESET_SCHEMES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for baseline in "${BASELINES[@]}"; do
                for num_calibration_samples in "${NUM_CALIBRATION_SAMPLES[@]}"; do
                    for alpha in "${ALPHA[@]}"; do
                        run_quantization "${model_size}" "${scheme}" "${dataset}" "${baseline}" "${num_calibration_samples}" "${alpha}"
                    done
                done
            done
        done
    done
done
