import pytest
import jax
from jax import random
import numpy as np
import jax.numpy as jnp
from jax_kan.kan import KANLinear as FlaxKANLinear
from .pytorch_kan import KANLinear as TorchKANLinear
from flax.core.frozen_dict import freeze, unfreeze
import torch

batch_size = 31
in_features = 23
out_features = 11

@pytest.fixture
def flax_module():
    key = random.PRNGKey(0)  # Using a fixed key for reproducibility
    module = FlaxKANLinear(in_features, out_features, key)
    dummy_input = jax.random.uniform(key, (batch_size, in_features))
    variables = module.init(key, dummy_input)
    return module, variables

@pytest.fixture
def torch_module():
    return TorchKANLinear(in_features=in_features, out_features=out_features)

# def test_parameter_shapes_grid_values(flax_module, torch_module):
#     _, flax_variables = flax_module

#     # Check shapes for all weights and buffers
#     param_checks = {
#         'base_weight': ('params', torch_module.base_weight),
#         'spline_weight': ('params', torch_module.spline_weight),
#         'grid': ('buffers', torch_module.grid)
#     }

#     if torch_module.enable_standalone_scale_spline:
#         param_checks['spline_scaler'] = ('params', torch_module.spline_scaler)

#     for name, (flax_category, torch_tensor) in param_checks.items():
#         flax_shape = flax_variables[flax_category][name].shape
#         torch_shape = torch_tensor.shape
#         assert flax_shape == torch_shape, f"Shape mismatch for {name}: Flax {flax_shape}, Torch {torch_shape}"
#         if name == 'grid':
#             np.testing.assert_allclose(np.array(flax_variables[flax_category][name]), torch_tensor.detach().numpy(), atol=1e-5)


# def test_forward(torch_module, flax_module):
#     flax_model, flax_params = flax_module
#     # Copy parameters from PyTorch to Flax
#     for name, param in torch_module.named_parameters():
#         numpy_param = param.detach().numpy()
#         # if 'weight' in name:
#         #     numpy_param = numpy_param.T  # Transpose the weight to match Flax's convention
#         flax_params = unfreeze(flax_params)
#         flax_params['params'][name] = numpy_param
#         flax_params = freeze(flax_params)

    
#     # Test case 1: Correct input shape
#     input_tensor = jnp.ones((3, 10))  # batch size of 3, in_features of 10
#     flax_output = flax_model.apply(flax_params, input_tensor)

#     # Convert JAX input tensor to a numpy array first, then to a Torch tensor
#     numpy_input_tensor = np.array(input_tensor)  # Ensure a copy is created from JAX to numpy
#     torch_input_tensor = torch.tensor(numpy_input_tensor, dtype=torch.float32)
#     torch_output = torch_module(torch_input_tensor).detach().numpy()

#     # Check output shape
#     expected_output_shape = (3, 5)  # batch size of 3, out_features of 5
#     assert flax_output.shape == expected_output_shape, f"Output shape was {flax_output.shape}, but expected {expected_output_shape}"

#     # Check that the outputs are close
#     assert np.allclose(flax_output, torch_output, atol=1e-5), "Outputs of Torch and Flax models are not close enough!"

#     # Test case 2: Incorrect input shape
#     wrong_input_tensor = jnp.ones((3, 8))  # Incorrect second dimension
#     with pytest.raises(AssertionError):
#         flax_model.apply(flax_params, wrong_input_tensor)

# def assert_not_allclose(a, b, rtol=1e-05, atol=1e-08, err_msg=""):
#     if np.allclose(a, b, rtol=rtol, atol=atol):
#         raise AssertionError(err_msg)

# def test_update_grid_and_spline_weights(torch_module, flax_module):
#     flax_model, flax_params = flax_module
#     # Copy parameters from PyTorch to Flax
#     for name, param in torch_module.named_parameters():
#         numpy_param = param.detach().numpy()
#         flax_params = unfreeze(flax_params)
#         flax_params['params'][name] = numpy_param
#         flax_params = freeze(flax_params)

#         # Generate random input
#     in_features = flax_params['params']['base_weight'].shape[1]  # Example way to fetch in_features from model's base weight
#     batch_size = 13
#     input_tensor = torch.randn(batch_size, in_features)
    
#     # Compare values before updating
#     old_torch_grid = torch_module.grid.detach().clone()
#     old_torch_spline_weight = torch_module.spline_weight.detach().clone()
#     old_flax_grid = flax_params['buffers']['grid']
#     old_flax_spline_weight = flax_params['params']['spline_weight']
#     assert old_torch_grid.shape == old_flax_grid.shape, "Grid shapes do not match"
#     assert old_torch_spline_weight.shape == old_flax_spline_weight.shape, "Spline weight shapes do not match"
#     np.testing.assert_allclose(old_torch_grid.numpy(), old_flax_grid, atol=1e-6, rtol=1e-6, err_msg="Grid values do not match")
#     np.testing.assert_allclose(old_torch_spline_weight.numpy(), old_flax_spline_weight, atol=1e-6, rtol=1e-6, err_msg="Spline weight values do not match")

#     # Update grids and spline weights using PyTorch model
#     torch_module.update_grid(input_tensor)
#     torch_grid = torch_module.grid.detach().clone()
#     torch_spline_weight = torch_module.spline_weight.detach().clone()

#     # Update grids and spline weights using Flax model
#     _, flax_params = flax_model.apply(flax_params,
#                                       x = jnp.array(input_tensor.numpy()),
#                                       method=flax_model.update_grid,
#                                       mutable=['buffers', 'params'])

#     flax_grid = flax_params['buffers']['grid']
#     flax_spline_weight = flax_params['params']['spline_weight']

#     # Now we first test to see if the values have changed at all
#     assert_not_allclose(torch_grid.numpy(), old_torch_grid.numpy(), atol=1e-6, rtol=1e-6, err_msg="Torch didn't work")
#     assert_not_allclose(flax_spline_weight, old_flax_spline_weight, atol=1e-6, rtol=1e-6, err_msg="Flax didn't work")

#     # Comparing shapes and values
#     assert torch_grid.shape == flax_grid.shape, "Grid shapes do not match"
#     assert torch_spline_weight.shape == flax_spline_weight.shape, "Spline weight shapes do not match"
#     np.testing.assert_allclose(torch_grid.numpy(), flax_grid, atol=1e-6, rtol=1e-6, err_msg="Grid values do not match")
    # np.testing.assert_allclose(torch_spline_weight.numpy(), flax_spline_weight, atol=1e-6, rtol=1e-6, err_msg="Spline weight values do not match")

    # print("Test passed: Grid and spline weights match between PyTorch and Flax.")

def test_compare_torchjax(torch_module, flax_module):
    margin=0.01
    # Get models in place
    flax_model, flax_params = flax_module
    old_params = {}
    for name, param in torch_module.named_parameters():
        old_params[name] = param.detach().numpy()
    for name, param in torch_module.named_buffers():
        old_params[name] = param.detach().numpy()
    
    input_tensor_t = torch.randn(batch_size, in_features)
    input_tensor_f = jnp.array(input_tensor_t.numpy())

    # Compute splines with the correct grid argument
    splines_f = flax_model.b_splines(input_tensor_f, old_params['grid'], indicator='Called from test code')  # (batch, in_features, coeff)
    splines_f = jnp.transpose(splines_f, (1, 0, 2))  # (in_features, batch, coeff)
    scaled_spline_weight_f = old_params['spline_weight'] * (old_params['spline_scaler'][..., None])
    unreduced_spline_output_f = jnp.einsum('ibc,icd->ibd', splines_f, jnp.transpose(scaled_spline_weight_f, (1, 2, 0)))
    unreduced_spline_output_f = jnp.transpose(unreduced_spline_output_f, (1, 0, 2))  # (batch, in_features, out_features)
    
    splines = torch_module.b_splines(input_tensor_t)  # (batch, in, coeff)
    splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
    np.testing.assert_allclose(splines.detach().numpy(), np.array(splines_f), atol=1e-5, rtol=1e-5, err_msg="Splines do not match")

    orig_coeff = torch_module.scaled_spline_weight  # (out, in, coeff)
    orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
    unreduced_spline_output_t = torch.bmm(splines, orig_coeff)  # (in, batch, out)
    unreduced_spline_output_t = unreduced_spline_output_t.permute(1, 0, 2)  # (batch, in, out)

    np.testing.assert_allclose(unreduced_spline_output_t.detach().numpy(), np.array(unreduced_spline_output_f), atol=1e-5, rtol=1e-5, err_msg="unreduced_spline_output do not match")

    # Sorting each channel individually to collect data distribution
    x_sorted_f = jnp.sort(input_tensor_f, axis=0)
    grid_adaptive_f = x_sorted_f.at[jnp.linspace(0, batch_size - 1, flax_model.grid_size + 1, dtype=jnp.int32)].get()

    x_sorted_t = torch.sort(input_tensor_t, dim=0)[0]
    grid_adaptive_t = x_sorted_t[
        torch.linspace(
            0, batch_size - 1, torch_module.grid_size + 1, dtype=torch.int64, device=input_tensor_t.device)]
    
    np.testing.assert_allclose(grid_adaptive_t.detach().numpy(), np.array(grid_adaptive_f), atol=1e-6, rtol=1e-6, err_msg="grid_adaptive do not match")

    # Creating uniform grid steps
    uniform_step_f = (x_sorted_f[-1, :] - x_sorted_f[0, :] + 2 * margin) / flax_model.grid_size
    grid_uniform_f = (jnp.arange(flax_model.grid_size + 1)[:, None] * uniform_step_f + x_sorted_f[0, :] - margin)

    uniform_step_t = (x_sorted_t[-1] - x_sorted_t[0] + 2 * margin) / torch_module.grid_size
    grid_uniform_t = (
        torch.arange(
            torch_module.grid_size + 1, dtype=torch.float32, device=input_tensor_t.device
        ).unsqueeze(1)
        * uniform_step_t
        + x_sorted_t[0]
        - margin)
    
    np.testing.assert_allclose(grid_uniform_t.detach().numpy(), np.array(grid_uniform_f), atol=1e-6, rtol=1e-6, err_msg="grid_uniform do not match")

    # # Interpolate between adaptive and uniform grid
    grid_f = flax_model.grid_eps * grid_uniform_f + (1 - flax_model.grid_eps) * grid_adaptive_f
    grid_f = jnp.concatenate([
        grid_f[:1] - uniform_step_f * jnp.arange(flax_model.spline_order, 0, -1)[:, None],
        grid_f,
        grid_f[-1:] + uniform_step_f * jnp.arange(1, flax_model.spline_order + 1)[:, None]
    ], axis=0).T

    grid_t = torch_module.grid_eps * grid_uniform_t + (1 - torch_module.grid_eps) * grid_adaptive_t
    grid_t = torch.concatenate(
            [
                grid_t[:1]
                - uniform_step_t
                * torch.arange(torch_module.spline_order, 0, -1, device=input_tensor_t.device).unsqueeze(1),
                grid_t,
                grid_t[-1:]
                + uniform_step_t
                * torch.arange(1, torch_module.spline_order + 1, device=input_tensor_t.device).unsqueeze(1),
            ],
            dim=0,
        ).T
    
    np.testing.assert_allclose(grid_t.detach().numpy(), np.array(grid_f), atol=1e-6, rtol=1e-6, err_msg="new grid do not match")

    assert input_tensor_f.ndim == 2 and input_tensor_f.shape[1] == flax_model.in_features
    assert unreduced_spline_output_f.shape == (input_tensor_f.shape[0], flax_model.in_features, flax_model.out_features)
    torch_module.grid.copy_(grid_t)
    A_t = torch_module.b_splines(input_tensor_t).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
    B_t = unreduced_spline_output_t.transpose(0, 1)  # (in_features, batch_size, out_features)
    A_f = flax_model.b_splines(input_tensor_f, grid_f, 'second call from test code').transpose((1, 0, 2))  # (in_features, batch_size, grid_size + spline_order)
    B_f = unreduced_spline_output_f.transpose((1, 0, 2))  # (in_features, batch_size, out_features)

    np.testing.assert_allclose(A_t.detach().numpy(), np.array(A_f), atol=1e-6, rtol=1e-6, err_msg="A do not match")
    np.testing.assert_allclose(B_t.detach().numpy(), np.array(B_f), atol=1e-5, rtol=1e-5, err_msg="B do not match")

    # Define a function to solve lstsq for each feature
    def solve_feature(a, b):
        return jnp.linalg.lstsq(a, b, rcond=None)[0]  # (grid_size + spline_order, out_features)

    # Vectorize the function over the first axis (in_features)
    vectorized_solve = jax.vmap(solve_feature, in_axes=(0, 0), out_axes=0)

    # Apply the vectorized function
    coeffs_f = vectorized_solve(A_f, B_f)  # (in_features, grid_size + spline_order, out_features)
    result_f = coeffs_f.transpose((2, 0, 1))  # Transpose to (out_features, in_features, grid_size + spline_order)
    
    solution_t = torch.linalg.lstsq(
            A_t, B_t
        ).solution  # (in_features, grid_size + spline_order, out_features)
    result_t = solution_t.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)
    
    np.testing.assert_allclose(result_t.detach().numpy(), np.array(result_f), atol=1e-5, rtol=1e-5, err_msg="result do not match")