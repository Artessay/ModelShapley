import neural_function
from config import CONFIG
from cache_neuron_metrics import load_metric
from model_loader import load_model_and_tokenizer
from utils import seed_everything, get_experiment_name


def deactivate_neurons(args):
    seed = args.seed
    mode = args.mode
    is_train = args.train
    model_name = args.model
    dataset_name = args.dataset
    seed_everything(seed)

    config = CONFIG.get(dataset_name)
    model_path = f"/data/{model_name}"
    model, tokenizer = load_model_and_tokenizer(model_path)

    # parameter settings
    config['activate_ratio'] = 0.95
    activate_ratio = config['activate_ratio']
    activate_top_percentile = config['activate_top_percentile']
    experiment_name = get_experiment_name(model_name, dataset_name, mode, seed, 
                                          activate_ratio, activate_top_percentile, is_train)
    
    if mode == 'random':
        neural_function.activate_neuron_random(model, activate_ratio)
    else:
        # Load a specific metric (uncomment after saving)
        output_dir = f"./metrics/{model_name}/{dataset_name}"
        if mode == 'shapley':
            pass
            individual_metric = load_metric("individual", output_dir)
            cooperative_metric = load_metric("cooperative", output_dir)
            metric_data = neural_function.calculate_shapley(individual_metric, cooperative_metric)
        else:
            metric_data = load_metric(mode, output_dir)
        neural_function.activate_neuron_based_on_importance(model, activate_ratio, activate_top_percentile, metric_data)

    # save model and tokenizer
    save_path = f"/data/{experiment_name}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    deactivate_neurons(args)