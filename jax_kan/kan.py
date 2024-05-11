import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import variance_scaling, kaiming_uniform
from jax import lax, ops
from typing import Any

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
        self.variable('buffers', 'grid', lambda: grid)

        # Base weight initialization
        self.base_weight = self.param('base_weight',
                                      kaiming_uniform(),
                                      (self.out_features,
                                       self.in_features))
        
        # Spline scaler initialization
        if self.enable_standalone_scale_spline:
            self.spline_scaler = self.param('spline_scaler', kaiming_uniform(), (self.out_features, self.in_features))
        
        # Spline weight initialization
        self.spline_weight = self.param('spline_weight',
                                        self.init_spline_weights,
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
        x = x[:, :, None]  # Expand dims to match grid dimensions
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:-k])
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

        base_output = jnp.dot(self.base_activation(x), self.base_weight.T)
        # Calculate scaled_spline_weight within the call
        print(self.spline_weight.shape, self.spline_scaler[..., None].shape)
        scaled_spline_weight = self.spline_weight * (self.spline_scaler[..., None] if self.enable_standalone_scale_spline else 1.0)

        # Calculate spline output using b_splines
        grid = self.variables['buffers']['grid']
        print(grid.shape)
        b_spline_out = self.b_splines(x, grid).reshape((x.shape[0], -1))
        spline_output = jnp.dot(b_spline_out, scaled_spline_weight.reshape((self.out_features, -1)).T)

        return base_output + spline_output
    
    def update_grid(self, x, margin=0.01):
        '''
        Updates the grid and spline weights in the module state based on the input data.

        Parameters:
        - x: The input data of shape (batch, in_features).
        - margin: The margin added to the range of the input data for computing the uniform grid steps. Default is 0.01.

        Returns:
        - new_grid: The updated grid of shape (out_features, grid_size + 1).
        - new_spline_weight: The updated spline weights of shape (in_features, out_features, coeff).

        Raises:
        - AssertionError: If the input data has incorrect shape.

        This function updates the grid and spline weights in the module state based on the input data. It performs the following steps:
        1. Computes the splines using the current grid and the input data.
        2. Adjusts the spline weights based on the spline scaler and the standalone scale spline flag.
        3. Computes the spline output without reduction.
        4. Sorts each channel of the input data individually to collect data distribution.
        5. Computes the adaptive grid by selecting values from the sorted input data.
        6. Computes the uniform grid steps based on the range of the sorted input data and the margin.
        7. Interpolates between the adaptive and uniform grid to obtain the final grid.
        8. Transposes the grid and computes the new spline weights based on the input data, spline output, and the new grid.
        9. Returns the updated grid and spline weights.

        Note: The module state refers to the internal state of the module that holds the grid and spline weights.
        '''
        assert x.ndim == 2 and x.shape[1] == self.in_features
        batch = x.shape[0]

        # Access the current grid from state
        current_grid = self.variables['buffers']['grid']
        
        # Compute splines with the correct grid argument
        splines = self.b_splines(x, current_grid)  # (batch, in_features, coeff)
        splines = jnp.transpose(splines, (1, 0, 2))  # (in_features, batch, coeff)

        # Access and adjust the spline weights correctly
        scaled_spline_weight = self.spline_weight * (self.spline_scaler[:, None] if self.enable_standalone_scale_spline else 1.0)

        # Compute spline output without reduction
        unreduced_spline_output = jnp.einsum('ibc,icd->ibd', splines, scaled_spline_weight)
        unreduced_spline_output = jnp.transpose(unreduced_spline_output, (1, 0, 2))  # (batch, in_features, out_features)

        # Sorting each channel individually to collect data distribution
        x_sorted = jnp.sort(x, axis=0)
        grid_adaptive = x_sorted.at[jnp.linspace(0, batch - 1, self.grid_size + 1, dtype=jnp.int32)].get()

        # Creating uniform grid steps
        uniform_step = (x_sorted[-1, :] - x_sorted[0, :] + 2 * margin) / self.grid_size
        grid_uniform = (jnp.arange(self.grid_size + 1)[:, None] * uniform_step + x_sorted[0, :] - margin)

        # Interpolate between adaptive and uniform grid
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = jnp.concatenate([
            grid[:1] - uniform_step * jnp.arange(self.spline_order, 0, -1)[:, None],
            grid,
            grid[-1:] + uniform_step * jnp.arange(1, self.spline_order + 1)[:, None]
        ], axis=0)

        # Update the grid and spline weight in the module state
        new_grid = grid.T
        new_spline_weight = self.curve2coeff(x, unreduced_spline_output, new_grid)
        
        return new_grid, new_spline_weight