from typing import Tuple

import torch
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
)


# Manually define the get_execution_device function
def get_execution_device(model: torch.nn.Module) -> torch.device:
    """
    Returns the appropriate device (GPU if available, otherwise CPU) for the model.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Shapley correction function
# def apply_OBS_correction(theta: torch.Tensor, H: torch.Tensor, alpha=0.1) -> torch.Tensor:
#     """
#     Apply Shapley correction to the Hessian matrix.

#     :param theta: The weights (parameters) of the module
#     :param H: The Hessian matrix
#     :param alpha: The correction factor (default is 0.1)
#     :return: Corrected Hessian matrix
#     """
#     eps = 1e-6
#     # print("🔥  Shapley correction triggered")
#     # Transpose theta for easier calculations
#     H_diag = torch.diag(H)  # Extract the diagonal of the Hessian

#     return torch.diag_embed(H_diag)


def apply_OBS_correction(theta: torch.Tensor, H: torch.Tensor, alpha=0.1) -> torch.Tensor:
    """
    Apply Shapley correction to the Hessian matrix.

    :param theta: The weights (parameters) of the module
    :param H: The Hessian matrix
    :param alpha: The correction factor (default is 0.1)
    :return: Corrected Hessian matrix
    """
    eps = 1e-6
    # 强制所有计算都用 float32
    theta = theta.to(torch.float32)
    H = H.to(torch.float32)

    H_diag = torch.diag(H)  # Extract the diagonal of the Hessian

    return torch.diag_embed(H_diag)


# Modified GPTQModifier to include Shapley correction
class GPTQModifierWithOBSCorrection(GPTQModifier):
    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        """
        Quantize a module's weight according to the GPTQ algorithm with Shapley correction.

        :param module: The module being quantized
        :param args: Input arguments for the module forward pass
        """
        inp = args[0]

        # Initialize Hessian if not present
        if module not in self._num_samples:
            init_device = "cpu" if self.offload_hessians else get_execution_device(module)
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = 0

        # Accumulate Hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

        # Apply Shapley correction to the Hessian matrix
        corrected_hessian = apply_OBS_correction(module.weight, self._hessians[module], alpha=0.1)

        # Update the Hessian with the corrected version
        self._hessians[module] = corrected_hessian
