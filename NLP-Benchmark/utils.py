import os
import torch
import random
import logging
import argparse
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(log_file, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file output
    if log_file:
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")  # use 'w' to overwrite existing file, default is 'a' (append)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

get_experiment_name = lambda model_name, dataset_name, mode, seed, activate_ratio, activate_top_percentile, is_train: (
    f"{model_name}_{dataset_name}_{mode}_{seed}" if mode == 'vanilla' else
    f"{model_name}_{dataset_name}_{mode}_{seed}_r{activate_ratio}" if activate_top_percentile else
    f"{model_name}_{dataset_name}_{mode}_{seed}_r{activate_ratio}_inverse"
) if not is_train else (
    f"{model_name}_{dataset_name}/{mode}_{seed}" if mode == 'vanilla' else
    f"{model_name}_{dataset_name}/{mode}_{seed}_r{activate_ratio}" if activate_top_percentile else
    f"{model_name}_{dataset_name}/{mode}_{seed}_r{activate_ratio}_inverse"
)


def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Continue Finetune Benchmark")

    # pretrain, finetune, continue train
    parser.add_argument('-m', '--mode', type=str, help='train mode', default='vanilla', 
                        choices=['vanilla', 'random', 'gradient', 'individual', 'cooperative', 'weight', 'shapley'])
    
    parser.add_argument('-d', '--dataset', type=str, default='gsm8k', help='Dataset name',
                        choices=['gsm8k', 'mmlu'])
    
    # choices: meta-llama/Llama-3.2-3B-Instruct, Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct
    parser.add_argument('-p', '--model', type=str, default='Qwen/Qwen2.5-3B-Instruct', help='Model name')
    
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument('--train', action='store_true', help='Train the model')

    args = parser.parse_args()
    print(args)
    
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    experiment_name = get_experiment_name(
        model_name=args.model,
        dataset_name=args.dataset,
        mode=args.mode,
        seed=args.seed,
        activate_ratio=0.1,  # Placeholder, replace with actual value
        activate_top_percentile=False,  # Placeholder, replace with actual value
        is_train=args.train
    )
    print(f"Experiment Name: {experiment_name}")