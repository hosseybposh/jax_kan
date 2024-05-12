import pytest
import jax
from jax import random
import numpy as np
from jax_kan.kan import KANLinear as FlaxKANLinear
from .pytorch_kan import KANLinear as TorchKANLinear

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

def test_parameter_shapes(flax_module, torch_module):
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