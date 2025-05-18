import os
import torch

import neural_function
from config import CONFIG
from neural_function import compute_and_cache_metrics
from data_loader import get_data_loader
from model_loader import load_model_and_tokenizer
from utils import seed_everything

metric_fn_dict = {
    "vanilla": neural_function.calculate_vanilla,
    "gradient": neural_function.calculate_gradient,
    "individual": neural_function.calculate_individual_importance,
    "cooperative": neural_function.calculate_cooperative_interactions,
    "shapley": neural_function.calculate_shapley_value,
}


def save_metrics_by_metric_name(neuron_importance_dict: dict, output_dir: str):
    """
    Save computed metrics into separate files based on metric names.

    Args:
        neuron_importance_dict (dict): Nested dictionary of computed metrics.
        output_dir (str): Directory path to save the metric files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, metric_data in neuron_importance_dict.items():
        file_path = os.path.join(output_dir, f"{metric_name}.pt")
        try:
            # move data to cpu
            for key, value in metric_data.items():
                metric_data[key] = value.cpu()
            # save data
            torch.save(metric_data, file_path)
            print(f"Successfully saved {metric_name} metrics to: {file_path}")
        except Exception as e:
            print(f"Error saving {metric_name} metrics: {e}")

def load_metric(metric_name: str, output_dir: str) -> dict:
    """
    Load metrics from a saved file.

    Args:
        metric_name (str): Name of the metric to load.
        output_dir (str): Directory where metric files are stored.

    Returns:
        dict: Loaded metric data for the specified metric name.
    """
    file_path = os.path.join(output_dir, f"{metric_name}.pt")
    try:
        return torch.load(file_path)
    except Exception as e:
        print(f"Error loading {metric_name} metrics: {e}")
        return {}


def cache_neuron_metrics(args):
    seed = args.seed
    model_name = args.model
    dataset_name = args.dataset
    seed_everything(seed)

    config = CONFIG.get(dataset_name)
    model_path = f"/data/{model_name}"
    model, tokenizer = load_model_and_tokenizer(model_path)

    _, val_loader, _ = get_data_loader(
        dataset_name, tokenizer, config=config)

    # parameter settings
    output_dir = f"./metrics/{model_name}/{dataset_name}"

    # Compute metrics
    neuron_importance_dict = compute_and_cache_metrics(model, val_loader, metric_fn_dict)
    
    # Save metrics
    save_metrics_by_metric_name(neuron_importance_dict, output_dir)


if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    cache_neuron_metrics(args)