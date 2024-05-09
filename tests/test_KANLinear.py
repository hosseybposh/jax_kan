import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from jax_kan.kan import KANLinear as JxKANLinear  # Your Jax implementation
from .pytorch_kan import KANLinear as PyKANLinear  # Your local PyTorch implementation

def test_initialization():
    # Create instances of both models
    key = jax.random.PRNGKey(0)
    py_model = PyKANLinear(10, 5)
    jx_model = JxKANLinear(10, 5, key=key)

    # Test grid initialization
    assert py_model.grid.shape == jx_model.grid.shape, "Grid shapes do not match"
    assert py_model.grid.dtype == torch.from_numpy(np.array(jx_model.grid)).dtype, "Grid data types do not match"

    # Test base_weight initialization
    assert py_model.base_weight.shape == jx_model.base_weight.shape, "Base weight shapes do not match"
    assert py_model.base_weight.dtype == torch.from_numpy(np.array(jx_model.base_weight)).dtype, "Base weight data types do not match"

    # Test spline_weight initialization
    assert py_model.spline_weight.shape == jx_model.spline_weight.shape, "Spline weight shapes do not match"
    assert py_model.spline_weight.dtype == torch.from_numpy(np.array(jx_model.spline_weight)).dtype, "Spline weight data types do not match"

    # Test spline_scaler initialization if enabled
    if py_model.enable_standalone_scale_spline:
        assert hasattr(jx_model, 'spline_scaler'), "Jax model missing spline_scaler attribute"
        assert py_model.spline_scaler.shape == jx_model.spline_scaler.shape, "Spline scaler shapes do not match"
        assert py_model.spline_scaler.dtype == torch.from_numpy(np.array(jx_model.spline_scaler)).dtype, "Spline scaler data types do not match"

    # Additional tests for any other parameters or buffers initialized in the setup
    # ...

# Add additional pytest markers if needed to parameterize or configure the test environment
