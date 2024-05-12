import pytest
import jax
from jax import random
import numpy as np
import jax.numpy as jnp
from jax_kan.kan import KANLinear as FlaxKANLinear
from .pytorch_kan import KANLinear as TorchKANLinear
from flax.core.frozen_dict import freeze, unfreeze
import torch

@pytest.fixture
def flax_module():
    key = random.PRNGKey(0)  # Using a fixed key for reproducibility
    in_features = 10
    out_features = 5
    module = FlaxKANLinear(in_features, out_features, key)
    dummy_input = jax.random.uniform(key, (13, in_features))
    variables = module.init(key, dummy_input)
    return module, variables

@pytest.fixture
def torch_module():
    return TorchKANLinear(in_features=10, out_features=5)

def test_parameter_shapes_grid_values(flax_module, torch_module):
    _, flax_variables = flax_module

    # Check shapes for all weights and buffers
    param_checks = {
        'base_weight': ('params', torch_module.base_weight),
        'spline_weight': ('params', torch_module.spline_weight),
        'grid': ('buffers', torch_module.grid)
    }

    if torch_module.enable_standalone_scale_spline:
        param_checks['spline_scaler'] = ('params', torch_module.spline_scaler)

    for name, (flax_category, torch_tensor) in param_checks.items():
        print(f"Checking {name}")
        flax_shape = flax_variables[flax_category][name].shape
        torch_shape = torch_tensor.shape
        assert flax_shape == torch_shape, f"Shape mismatch for {name}: Flax {flax_shape}, Torch {torch_shape}"
        if name == 'grid':
            np.testing.assert_allclose(np.array(flax_variables[flax_category][name]), torch_tensor.detach().numpy(), atol=1e-5)


def test_forward(torch_module, flax_module):
    flax_model, flax_params = flax_module
    # Copy parameters from PyTorch to Flax
    for name, param in torch_module.named_parameters():
        numpy_param = param.detach().numpy()
        # if 'weight' in name:
        #     numpy_param = numpy_param.T  # Transpose the weight to match Flax's convention
        flax_params = unfreeze(flax_params)
        flax_params['params'][name] = numpy_param
        flax_params = freeze(flax_params)

    # Test case 1: Correct input shape
    input_tensor = jnp.ones((3, 10))  # batch size of 3, in_features of 10
    flax_output = flax_model.apply(flax_params, input_tensor)

    # Convert JAX input tensor to a numpy array first, then to a Torch tensor
    numpy_input_tensor = np.array(input_tensor)  # Ensure a copy is created from JAX to numpy
    torch_input_tensor = torch.tensor(numpy_input_tensor, dtype=torch.float32)
    torch_output = torch_module(torch_input_tensor).detach().numpy()

    # Check output shape
    expected_output_shape = (3, 5)  # batch size of 3, out_features of 5
    assert flax_output.shape == expected_output_shape, f"Output shape was {flax_output.shape}, but expected {expected_output_shape}"

    # Check that the outputs are close
    assert np.allclose(flax_output, torch_output, atol=1e-5), "Outputs of Torch and Flax models are not close enough!"

    # Test case 2: Incorrect input shape
    wrong_input_tensor = jnp.ones((3, 8))  # Incorrect second dimension
    with pytest.raises(AssertionError):
        flax_model.apply(flax_params, wrong_input_tensor)