import neural_hook
import neural_function
from config import CONFIG
from data_loader import get_data_loader
from cache_neuron_metrics import load_metric
from neuron_trainer import NeuronTrainer
from model_loader import load_model_and_tokenizer
from utils import seed_everything, get_experiment_name


def main(args):
    seed = args.seed
    mode = args.mode
    model_name = args.model
    dataset_name = args.dataset
    seed_everything(seed)

    config = CONFIG.get(dataset_name)
    model_path = f"/data/{model_name}"
    model, tokenizer = load_model_and_tokenizer(model_path)

    train_loader, val_loader, _ = get_data_loader(
        dataset_name, tokenizer, config=config)

    # parameter settings
    activate_ratio = config['activate_ratio']
    activate_top_percentile = config['activate_top_percentile']
    
    experiment_name = get_experiment_name(model_name, dataset_name, mode, seed, 
                                          activate_ratio, activate_top_percentile, is_train=True)

    if mode == 'vanilla':
        metric_data = {}
    elif mode == 'random':
        metric_data = neural_function.assign_random_importance(model)
    else:
        # Load a specific metric (uncomment after saving)
        output_dir = f"./metrics/{model_name}/{dataset_name}"
        metric_data = load_metric(mode, output_dir)
    neuron_mask_dict = neural_hook.create_neuron_mask_dict(model, activate_ratio, activate_top_percentile, metric_data)

    config['neuron_mask_dict'] = neuron_mask_dict
    config['output_dir'] = f"/data/{experiment_name}"
    config['log_path'] = f"./logs/{experiment_name}.log"
    trainer = NeuronTrainer(model, tokenizer, train_loader, val_loader, config)
    trainer.train()

if __name__ == "__main__":
    from utils import get_args
    args = get_args()
    main(args)