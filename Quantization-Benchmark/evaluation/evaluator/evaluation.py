"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.
"""


import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from reward_score import reward_fn

def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)

@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def evaluate(config):
    dataset = pd.read_parquet(config.data.path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)

    # Process results as they come in
    for i in tqdm(range(total)):
        data_source, score = process_item(reward_fn, data_sources[i], responses[i], reward_model_data[i])
        data_source_reward[data_source].append(score)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards).item()

    # Write results to result.log file
    with open('./result.log', 'a') as f:
        f.write("="*100 + "\n")
        f.write(f"Data path: {config.data.path}\n")
        f.write(f"Metrics: {metric_dict}\n\n")
    
    # Also print to console for visibility
    print(config.data.path)
    print(metric_dict)

if __name__ == "__main__":
    evaluate()