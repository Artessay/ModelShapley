import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

def activate_neuron_random(model: PreTrainedModel, activate_ratio: float):
    """randomly select some neurals, and froze other neurals"""
    neuron_groups = {}
    for name, param in model.named_parameters():
        if "model.layers" in name and "mlp" in name:
            # Get the neuron identifier by removing the .weight or .bias suffix
            neuron_key = name.rsplit('.', 1)[0]
            neuron_value = name.rsplit('.', 1)[1]
            assert neuron_value in ['weight', 'bias']

            if neuron_key not in neuron_groups:
                neuron_groups[neuron_key] = {}
            neuron_groups[neuron_key][neuron_value] = param

    # Set the requires_grad attribute of parameters
    for neuron_key, param_dict in neuron_groups.items():
        weight_param: torch.nn.Parameter = param_dict['weight']
        num_neurons = weight_param.shape[0]
        num_activate = int(num_neurons * activate_ratio)

        # Randomly select neuron groups to activate
        activate_indices = random.sample(range(num_neurons), k=num_activate)

        # Create a mask to indicate activate neurons
        mask = torch.zeros(num_neurons, dtype=torch.bool)
        mask[activate_indices] = True

        # Set non - activate neurons' weight and bias to 0
        weight_param.data[~mask] = 0

        bias_param: torch.nn.Parameter = param_dict.get('bias', None)
        if bias_param is not None:
            assert weight_param.shape[0] == bias_param.shape[0]
            bias_param.data[~mask] = 0



@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def compute_loss_and_backward(model: PreTrainedModel, model_inputs: dict):
    """
    Compute the loss and perform backward pass for the model.
    Args:
        model (PreTrainedModel): The Vision Transformer model.
        model_inputs (dict): The model inputs.
    Returns:
        None: The function modifies the requires_grad attribute of model parameters in-place.
    """
    input_ids = model_inputs["input_ids"].cuda()
    loss_mask = model_inputs.pop("loss_mask")[:, :-1].reshape(-1).cuda()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    # Forward pass
    outputs = model(**model_inputs)
    logits = outputs.logits
    labels = input_ids[:, 1:].contiguous()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels.contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss * loss_mask.to(loss.device)

    valid_token_num = torch.sum(loss_mask)
    loss = torch.sum(loss) / valid_token_num # (valid_token_num + 1e-8)
    
    # Backward pass to compute gradients
    loss.backward()


def param_name_check(name: str) -> bool:
    """
    Check if the parameter name is valid for Shapley value calculation.

    Args:
        name (str): The name of the parameter.

    Returns:
        bool: True if the parameter name is valid, False otherwise.
    """
    target_module = ['up_proj', 'gate_proj'] # down_proj
    return "model.layers" in name and any(module in name for module in target_module) and "weight" in name


def param_cache_check(name: str, param: torch.nn.Parameter) -> bool:
    """
    Check if the parameter name is valid for Shapley value calculation.

    Args:
        name (str): The name of the parameter.

    Returns:
        bool: True if the parameter name is valid, False otherwise.
    """
    return "model.layers" in name and param.ndim == 2


@torch.no_grad()
def calculate_vanilla(param: torch.nn.Parameter) -> torch.Tensor:
    """
    Calculate the gradient of a given parameter.
    Args:
        param (torch.nn.Parameter): The parameter for which to calculate the gradient.
    Returns:
        torch.Tensor: The gradient of the parameter.
    """
    assert param.grad is not None
    assert param.ndim == 2, "Gradient calculation now is only supported for 2D parameters."

    return param.grad


@torch.no_grad()
def calculate_random_importance(param: torch.nn.Parameter) -> torch.Tensor:
    """
    Randomply give an importance score of a given parameter.
    Args:
        param (torch.nn.Parameter): The parameter for which to calculate the gradient.
    Returns:
        torch.Tensor: The importance score of the parameter.
    """
    # Get the size of the first dimension of the parameter
    first_dim_size = param.shape[0]
    
    # Generate random importance scores between 0 and 1
    importance_scores = torch.rand(first_dim_size, device=param.device)
    
    return importance_scores


@torch.no_grad()
def calculate_gradient(param: torch.nn.Parameter) -> torch.Tensor:
    """
    Calculate the gradient of a given parameter.
    Args:
        param (torch.nn.Parameter): The parameter for which to calculate the gradient.
    Returns:
        torch.Tensor: The gradient of the parameter.
    """
    assert param.grad is not None
    assert param.ndim == 2, "Gradient calculation now is only supported for 2D parameters."

    # Compute the gradient
    param_gradient = torch.sum(param.grad, dim=1)
    return param_gradient


@torch.no_grad()
def calculate_individual_importance(param: torch.nn.Parameter) -> torch.Tensor:
    """
    Calculate the gradient trace of a given parameter.
    Args:
        param (torch.nn.Parameter): The parameter for which to calculate the gradient trace.
    Returns:
        torch.Tensor: The gradient trace of the parameter.
    """
    assert param.grad is not None
    assert param.ndim == 2, "Gradient trace calculation now is only supported for 2D parameters."
    
    # Compute the gradient trace
    individual_importance = torch.sum(param.grad * param, dim=1)  # shape: (param.shape[0],)
    
    return individual_importance


@torch.no_grad()
def calculate_cooperative_interactions(param: torch.nn.Parameter) -> torch.Tensor:
    """
    Calculate the gradient trace of a given parameter.
    Args:
        param (torch.nn.Parameter): The parameter for which to calculate the gradient trace.
    Returns:
        torch.Tensor: The gradient trace of the parameter.
    """
    assert param.grad is not None
    assert param.ndim == 2, "Gradient trace calculation now is only supported for 2D parameters."
    
    # Approximate Hessian matrix by Fisher information matrix
    hessian_matrix = torch.matmul(param.grad, param.grad.T) # shape: (param.shape[0], param.shape[0])

    # Compute the Shapley value
    cooperative_interactions = torch.sum(param * torch.matmul(hessian_matrix, param), dim=1)  # shape: (param.shape[0],)

    return cooperative_interactions

@torch.no_grad()
def calculate_shapley_value(param: torch.nn.Parameter) -> torch.Tensor:
    """
    Calculate the Shapley value for a given parameter.

    Args:
        param (torch.nn.Parameter): The parameter for which to calculate the Shapley value.

    Returns:
        float: The Shapley value of the parameter.
    """
    
    # Compute the shapley value for the parameter
    assert param.grad is not None
    assert param.ndim == 2, "Shapley value calculation now is only supported for 2D parameters."

    # Compute the Shapley value
    individual_importance = calculate_individual_importance(param)
    cooperative_interactions = calculate_cooperative_interactions(param)

    # The Shapley value is the sum of individual importance and cooperative interactions
    shapley_value = individual_importance + 0.5 * cooperative_interactions
    return shapley_value

def compute_and_cache_metrics(model: PreTrainedModel, val_loader: DataLoader, metric_fn_dict: dict):
    """
    Activate neurons in the model based on the specific metric.

    Args:
        model (PreTrainedModel): The Vision Transformer model.
        activate_ratio (float): The ratio of neurons to be activated.
        val_loader (DataLoader): The validation data loader.
        metric_fn (Callable): The metric function to be used for neuron activation.

    Returns:
        None: The function modifies the requires_grad attribute of model parameters in-place.
    """
    
    model.eval()

    # Store the value of each weight
    neuron_importance_dict = defaultdict(dict)

    for model_inputs in tqdm(val_loader, ncols=80, desc="Identify"):
        
        # Zero the gradients
        model.zero_grad()

        # Compute the loss and perform backward pass
        compute_loss_and_backward(model, model_inputs)

        # Compute value only for weights
        for name, param in model.named_parameters():
            if param_cache_check(name, param):
                # Compute the value
                for metric_name, metric_fn in metric_fn_dict.items():
                    param_value = metric_fn(param).detach()
                    if name not in neuron_importance_dict[metric_name]:
                        neuron_importance_dict[metric_name][name] = param_value
                    else:
                        neuron_importance_dict[metric_name][name] += param_value

    return neuron_importance_dict



def activate_neuron_based_on_importance(model: PreTrainedModel, activate_ratio: float, activate_top_percentile: bool, weight_neuron_value: dict):
    """
    Activate neurons in the model based on the specific metric.

    Args:
        model (PreTrainedModel): The Vision Transformer model.
        activate_ratio (float): The ratio of neurons to be activated.
        activate_top_percentile (bool): Whether to use top ratio important neurons or bottom ratio.
        weight_neuron_value (dict): The value of each weight.

    Returns:
        None: The function modifies the requires_grad attribute of model parameters in-place.
    """
    
    model.eval()

    # Activate neurons based on the metric value
    for name, param in model.named_parameters():
        if "model.layers" in name and "mlp" in name:
            if "weight" in name:
                param_value = weight_neuron_value[name]
            elif "bias" in name:
                # Find the corresponding weight name
                weight_name = name.rsplit('.', 1)[0] + '.weight'
                param_value = weight_neuron_value[weight_name]

            param_value = torch.abs(param_value)
            num_neuron = param_value.shape[0]
            num_to_activate = int(num_neuron * activate_ratio)
            
            # select the top num_to_activate neurons
            sorted_indices = torch.argsort(param_value, descending=activate_top_percentile)
            top_indices = sorted_indices[:num_to_activate]
            mask = torch.zeros(num_neuron, dtype=torch.bool, device=param.data.device)
            mask[top_indices] = True

            # Expand mask to match the weight shape
            if len(param.shape) == 2: # weight matrix
                expanded_mask = mask.unsqueeze(1).expand_as(param)
            elif len(param.shape) == 1: # bias vector
                expanded_mask = mask
            else:
                raise ValueError(f"Parameter {name} with shape {param.shape} is not supported")
            
            param.data *= expanded_mask.float()

def assign_random_importance(model: PreTrainedModel):
    weight_neuron_value = {}

    # Compute value only for weights
    for name, param in model.named_parameters():
        if param_cache_check(name, param):
            # Compute the value
            param_value = calculate_random_importance(param)
            weight_neuron_value[name] = param_value

    return weight_neuron_value

def calculate_shapley(individual_metric, cooperative_metric):
    """
    Calculate the Shapley value for a given parameter.

    Args:
        individual_metric (dict): The individual metric for each parameter.
        cooperative_metric (dict): The cooperative metric for each parameter.

    Returns:
        dict: The Shapley value of the parameter.
    """
    
    # Compute the shapley value for the parameter
    shapley_value = {}
    for name in individual_metric.keys():
        shapley_value[name] = individual_metric[name] + 0.0 * cooperative_metric[name]
    
    return shapley_value