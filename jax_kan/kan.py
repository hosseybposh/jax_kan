import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import variance_scaling, kaiming_uniform
from jax import lax, ops
from typing import Any
from jax import random
import numpy as np

class KANLinear(nn.Module):
    in_features: int
    out_features: int
    key: Any
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    enable_standalone_scale_spline: bool = True
    base_activation: callable = nn.silu
    grid_eps: float = 0.02
    grid_range: tuple = (-1, 1)

    def setup(self):
        # Grid initialization
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = (
            jnp.arange(-self.spline_order, self.grid_size + self.spline_order + 1) * h
            + self.grid_range[0]
        )
        grid = jnp.tile(grid, (self.in_features, 1))
        self.grid = self.variable('buffers', 'grid', lambda: grid)

        # Base weight initialization
        key1, key2, key3 = random.split(self.key, 3)
        self.base_weight = self.variable('params', 'base_weight',
                                         kaiming_uniform(),
                                         key1,
                                         (self.out_features,
                                          self.in_features))

        # Spline scaler initialization
        if self.enable_standalone_scale_spline:
            self.spline_scaler = self.variable('params', 'spline_scaler',
                                               kaiming_uniform(),
                                               key2,
                                               (self.out_features,
                                                self.in_features))
        
        # Spline weight initialization
        self.spline_weight = self.variable('params', 'spline_weight',
                                           self.init_spline_weights,
                                           key3,
                                           (self.out_features,
                                            self.in_features,
                                            self.grid_size + self.spline_order))
        
    # TODO I wonder if the shape already calculates things correctly automatically
    def init_spline_weights(self, rng, shape):
        '''
        Initializes the spline weights for the Kan module.

        Parameters:
        - rng: The random number generator.
        - shape: The shape of the spline weights.

        Returns:
        - The initialized spline weights.

        Note:
        - This function may not work if module parameters like self.grid_size are used within the initialization.
        '''
        grid = self.variables['buffers']['grid']
        noise = (jax.random.uniform(rng, (self.grid_size + 1, self.in_features, self.out_features)) - 0.5) * self.scale_noise / self.grid_size
        scale = self.scale_spline if not self.enable_standalone_scale_spline else 1.0
        return scale * self.curve2coeff(grid.T[self.spline_order:-self.spline_order], noise, grid)
    
    def b_splines(self, x, grid):
        '''
        Compute B-spline basis functions for given input data and grid.

        Args:
            x (ndarray): Input data of shape (batch_size, num_points, in_features).
            grid (ndarray): Grid points for B-spline basis functions of shape (num_points, grid_size + spline_order + 1).

        Returns:
            ndarray: B-spline basis functions of shape (batch_size, in_features, grid_size + spline_order).

        Raises:
            AssertionError: If the input data `x` does not have the expected shape.

        '''
        assert x.ndim == 2 and x.shape[1] == self.in_features
        x = x[..., None]  # Expand dims to match grid dimensions

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
                ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.shape == (x.shape[0], self.in_features, self.grid_size + self.spline_order)
        return bases
    
    def curve2coeff(self, x, y, grid):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, in_features).
            y (jnp.ndarray): Output tensor of shape (batch_size, in_features, out_features).
            grid (jnp.ndarray): Grid tensor used in B-spline calculations.

        Returns:
            jnp.ndarray: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.ndim == 2 and x.shape[1] == self.in_features
        assert y.shape == (x.shape[0], self.in_features, self.out_features)

        A = self.b_splines(x, grid).transpose((1, 0, 2))  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose((1, 0, 2))  # (in_features, batch_size, out_features)

        # Define a function to solve lstsq for each feature
        def solve_feature(a, b):
            return jnp.linalg.lstsq(a, b, rcond=None)[0]  # (grid_size + spline_order, out_features)

        # Vectorize the function over the first axis (in_features)
        vectorized_solve = jax.vmap(solve_feature, in_axes=(0, 0), out_axes=0)

        # Apply the vectorized function
        coeffs = vectorized_solve(A, B)  # (in_features, grid_size + spline_order, out_features)
        result = coeffs.transpose((2, 0, 1))  # Transpose to (out_features, in_features, grid_size + spline_order)

        assert result.shape == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result
    
    def __call__(self, x):
        '''
        Applies the Kan function to the input tensor x.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape (batch_size, in_features).

        Returns:
            jax.numpy.ndarray: Output tensor of shape (batch_size, out_features).

        Raises:
            AssertionError: If the input tensor does not have the expected shape.

        '''
        assert x.ndim == 2 and x.shape[1] == self.in_features

        base_output = jnp.dot(self.base_activation(x), self.base_weight.value.T)
        # Calculate scaled_spline_weight within the call
        scaled_spline_weight = self.spline_weight.value * (self.spline_scaler.value[..., None] if self.enable_standalone_scale_spline else 1.0)

        # Calculate spline output using b_splines
        b_spline_out = self.b_splines(x, self.grid.value).reshape((x.shape[0], -1))
        spline_output = jnp.dot(b_spline_out, scaled_spline_weight.reshape((self.out_features, -1)).T)

        return base_output + spline_output
    
    def update_grid(self, x, compare, margin=0.01):

        # Use `lax.stop_gradient` only within the scope of this method
        x_no_grad = lax.stop_gradient(x)
        current_grid = lax.stop_gradient(self.variables['buffers']['grid'])
        spline_weight = lax.stop_gradient(self.variables['params']['spline_weight'])

        assert x_no_grad.ndim == 2 and x_no_grad.shape[1] == self.in_features, f"Input data has incorrect shape: {x_no_grad.shape} expected ({2}, {self.in_features})"
        batch_size = x_no_grad.shape[0]
        
        # Compute splines with the correct grid argument
        splines = self.b_splines(x_no_grad, current_grid)  # (batch, in_features, coeff)
        splines = jnp.transpose(splines, (1, 0, 2))  # (in_features, batch, coeff)
        scaled_spline_weight = spline_weight * (self.spline_scaler.value[..., None] if self.enable_standalone_scale_spline
            else 1.0)
        # TODO this is the problem
        unreduced_spline_output = jnp.einsum('ibc,icd->ibd', splines, jnp.transpose(scaled_spline_weight, (1, 2, 0)))
        unreduced_spline_output = jnp.transpose(unreduced_spline_output, (1, 0, 2))  # (batch, in_features, out_features)
        
        # Sorting each channel individually to collect data distribution
        x_sorted = jnp.sort(x_no_grad, axis=0)
        grid_adaptive = x_sorted.at[jnp.linspace(0, batch_size - 1, self.grid_size + 1, dtype=jnp.int32)].get()
        
        # Creating uniform grid steps
        uniform_step = (x_sorted[-1, :] - x_sorted[0, :] + 2 * margin) / self.grid_size
        grid_uniform = (jnp.arange(self.grid_size + 1)[:, None] * uniform_step + x_sorted[0, :] - margin)
        # Interpolate between adaptive and uniform grid
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = jnp.concatenate([
            grid[:1] - uniform_step * jnp.arange(self.spline_order, 0, -1)[:, None],
            grid,
            grid[-1:] + uniform_step * jnp.arange(1, self.spline_order + 1)[:, None]
        ], axis=0).T
        self.variables['buffers']['grid'] = grid
        new_weights = self.curve2coeff(x_no_grad, compare, grid) # This is the problem
        # np.testing.assert_allclose(compare, np.array(unreduced_spline_output), atol=1e-6, rtol=1e-6, err_msg="oops")
        self.variables['params']['spline_weight'] = new_weights

class KAN(nn.Module):
    layers_hidden: list
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    base_activation: callable = nn.silu  # nn.silu corresponds to torch.nn.SiLU
    grid_eps: float = 0.02
    grid_range: tuple = (-1, 1)

    def setup(self):
        self.layers = [KANLinear(in_features, out_features, self.grid_size, self.spline_order,
                                 self.scale_noise, self.scale_base, self.scale_spline, self.base_activation,
                                 self.grid_eps, self.grid_range)
                       for in_features, out_features in zip(self.layers_hidden, self.layers_hidden[1:])]

    def __call__(self, x, update_grid=False):
        updates = {}
        for i, layer in enumerate(self.layers):
            if update_grid:
                layer_updates = self.apply({'params': self.params[f'layers_{i}'], 'buffers': self.variables['buffers'][f'layers_{i}']},
                                           x, method=KANLinear.update_grid)
                updates[f'layers_{i}'] = layer_updates
            x = layer(x)
        if updates:
            self.variables = jax.tree_multimap(lambda var, update: update if update is not None else var,
                                               self.variables, updates)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # Calculate and sum regularization loss from all layers
        total_loss = 0.0
        for layer in self.layers:
            total_loss += layer.regularization_loss(regularize_activation, regularize_entropy)
        return total_loss
    
