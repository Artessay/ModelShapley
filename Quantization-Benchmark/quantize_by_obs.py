import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from modifiers import GPTQModifierWithOBSCorrection


def set_seed(seed: int = 42):
    # Python built-in random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch (CPU/CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 对 cudnn 做一些限制以获得更确定的结果
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Python hash 随机化
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scheme", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    model_path = args.model_path
    scheme = args.scheme
    dataset = args.dataset
    dataset_config = args.dataset_config
    quant_path = f"{model_path}-obs-{scheme}-{dataset}"
    quant_path += "-fl32"
    print(quant_path)

    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Select quantization algorithm. In this case, we:
    #   * apply SmoothQuant to make the activations easier to quantize
    #   * quantize the weights to int4 with GPTQ (static per channel)
    #   * quantize the activations to int8 (dynamic per token)
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifierWithOBSCorrection(scheme=scheme, targets="Linear", ignore=["lm_head"]),
    ]

    # Apply quantization using the built in open_platypus dataset.
    #   * See examples for demos showing how to pass a custom calibration set
    oneshot(
        model=model_path,
        dataset=dataset,
        dataset_config_name=dataset_config,
        recipe=recipe,
        output_dir=quant_path,
        max_seq_length=2048,
        num_calibration_samples=512,
    )

    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time

    with open("result-time.log", "a") as f:
        f.write(f"Script: quantize_by_obs.py\n")
        f.write(f"Start time: {start_datetime}\n")
        f.write(f"End time: {end_datetime}\n")
        f.write(f"Duration: {duration:.2f} seconds\n")
        f.write("-" * 50 + "\n")
