import torch
from transformers import PreTrainedModel

from neural_function import param_name_check

def create_neuron_mask_dict(model: PreTrainedModel, activate_ratio: float, activate_top_percentile: bool, weight_neuron_value: dict):
    neuron_mask_dict = {}

    # Activate neurons based on the metric value
    for name, param in model.named_parameters():
        if param_name_check(name):
            param_value = weight_neuron_value[name]
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
            
            neuron_mask_dict[name] = expanded_mask

    return neuron_mask_dict

# ========= Register/Update Hook and Clear Optimizer Buffers =========
def set_neuron_freeze_hook(model: PreTrainedModel, neuron_mask_dict: dict):
    # Define and register new hook (zero out gradients for frozen neurons) for each parameter
    for name, param in model.named_parameters():
        if param.requires_grad and name in neuron_mask_dict:
            mask: torch.Tensor = neuron_mask_dict[name]
            
            # Create hook function to zero out gradients for frozen neurons
            def create_hook(param_mask: torch.Tensor):
                def hook_fn(grad):
                    return grad * param_mask.to(grad.dtype)
                return hook_fn
            param.register_hook(create_hook(mask))
    
                
def clear_optimizer_buffers(model: PreTrainedModel, optimizer: torch.optim.Optimizer, neuron_mask_dict: dict):
    # Clear Adam's momentum buffers for frozen neurons
    for name, param in model.named_parameters():
        if param.requires_grad and name in neuron_mask_dict and param in optimizer.state:
            mask: torch.Tensor = neuron_mask_dict[name]
            state = optimizer.state[param]                  # Exists only after first step
            if "exp_avg" in state:                          # Adam / AdamW
                state["exp_avg"].mul_(mask.float())
            if "exp_avg_sq" in state:                       # Adam / AdamW
                state["exp_avg_sq"].mul_(mask.float())
            if "momentum_buffer" in state:                  # SGD with momentum
                state["momentum_buffer"].mul_(mask.float())
