"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import CONFIG
from reward_score import reward_fn

from utils import get_experiment_name

def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


# Process items in parallel using ThreadPoolExecutor
def process_items_parallel(reward_fn, data_sources, responses, reward_model_data, max_workers=8):
    valid_scores = 0
    total_score = 0.0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
        future_to_idx = {
            executor.submit(
                process_item, 
                reward_fn, 
                data_sources[i], 
                responses[i], 
                reward_model_data[i]
            ): i 
            for i in range(len(data_sources))
        }
        
        # Use tqdm to display progress bar for completed tasks
        futures_iter = tqdm(
            as_completed(future_to_idx), 
            total=len(data_sources), 
            ncols=80, 
            desc="Processing items"
        )
        
        for future in futures_iter:
            try:
                # Get the result from the completed future
                _, score = future.result()
                total_score += score
                valid_scores += 1
            except Exception as e:
                # Handle exceptions for individual tasks
                print(f"Error processing item -: {e}")
                
    # Calculate accuracy, handle case where all tasks failed
    return total_score / valid_scores if valid_scores > 0 else 0.0

def evaluate(args):
    seed = args.seed
    mode = args.mode
    is_train = args.train
    model_name = args.model
    dataset_name = args.dataset

    config = CONFIG.get(dataset_name)
    activate_ratio = config['activate_ratio']
    activate_top_percentile = config['activate_top_percentile']
    
    experiment_name = get_experiment_name(model_name, dataset_name, mode, seed, 
                                          activate_ratio, activate_top_percentile, is_train)

    output_path = f"outputs/{experiment_name}.parquet"
    dataset = pd.read_parquet(output_path)

    responses = dataset["responses"]
    data_sources = dataset["data_source"]
    reward_model_data = dataset["reward_model"]

    # Process results as they come in
    # accuracy = process_items_parallel(reward_fn, data_sources, responses, reward_model_data)
    rewards = []
    for i in tqdm(range(len(dataset)), ncols=80, desc="Processing items"):
        _, score = process_item(reward_fn, data_sources[i], responses[i], reward_model_data[i])
        rewards.append(score)
    accuracy = np.mean(rewards).item()
    accuracy = round(100 * accuracy, 2)
    print(f"Accuracy: {accuracy}")

    result_path = f"outputs/{experiment_name}.json"
    with open(result_path, "w") as f:
        config["accuracy"] = accuracy
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    evaluate(args)