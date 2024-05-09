import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import variance_scaling, kaiming_uniform
from jax import lax, ops
from typing import Any
from scipy.linalg import lstsq

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
        # Setting up the grid
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = jnp.tile(jnp.arange(-self.spline_order, self.grid_size + self.spline_order + 1) * h + self.grid_range[0], (self.in_features, 1))
        self.grid = self.variable('buffers', 'grid', lambda: grid)

        # Initialize the parameters
        self.base_weight = self.param('base_weight', kaiming_uniform(jnp.sqrt(5) * self.scale_base), (self.out_features, self.in_features))
        self.spline_weight = self.param('spline_weight', nn.initializers.lecun_normal(), (self.out_features, self.in_features, self.grid_size + self.spline_order))

        # Initializing spline weight with noise and transformations
        key1, key2 = jax.random.split(self.key)
        noise = jax.random.uniform(key1, (self.out_features, self.in_features, self.grid_size + 1), minval=-0.5, maxval=0.5) * self.scale_noise / self.grid_size
        # Assuming `curve2coeff` transformation function exists
        adjusted_spline_weight = (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(grid.T[self.spline_order : -self.spline_order], noise, grid)
        self.spline_weight = adjusted_spline_weight

        if self.enable_standalone_scale_spline:
            self.spline_scaler = self.param('spline_scaler', kaiming_uniform(a=jnp.sqrt(5) * self.scale_spline), (self.out_features, self.in_features))
    
    def get_grid(self):
        """Method to safely access the grid buffer for testing or diagnostics."""
        return self.grid.value

    def b_splines(self, x, grid):
        """
        Compute the B-spline bases for the given input tensor in Jax.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape (batch_size, in_features).

        Returns:
            jax.numpy.ndarray: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.ndim == 2 and x.shape[1] == self.in_features

        # grid = self.grid.value  # Access the non-trainable grid variable
        x = x[:, :, None]  # Expand dimensions for broadcasting

        # Compute initial B-spline bases
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(x.dtype)

        # Recursive construction of B-spline bases of higher orders
        for k in range(1, self.spline_order + 1):
            t1 = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
            t2 = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]
            bases = t1 + t2

        return bases  # No need to call .contiguous() as Jax handles memory differently
    
    def curve2coeff(self, x, y, grid):
        """
        Compute the coefficients of the curve that interpolates the given points in Jax.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape (batch_size, in_features).
            y (jax.numpy.ndarray): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            jax.numpy.ndarray: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.ndim == 2 and x.shape[1] == self.in_features
        assert y.shape == (x.shape[0], self.in_features, self.out_features)

        A = self.b_splines(x, grid).transpose(1, 0, 2)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(1, 0, 2)  # (in_features, batch_size, out_features)

        # Solve least squares problem
        solution = jnp.array([lstsq(A[i], B[i])[0] for i in range(A.shape[0])])  # list comprehension to handle per-feature lstsq

        # Reordering dimensions to match output requirements
        result = solution.transpose(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        return result
    
    def scaled_spline_weight(self):
        """ Calculate scaled spline weights based on the mode. """
        return self.spline_weight * (self.spline_scaler[:, :, None] if self.enable_standalone_scale_spline else 1.0)

    def update_grid(model, x, margin=0.01, grid_eps=0.02):
        # Assuming x is already sorted or you handle sorting outside this function
        grid_adaptive = x[jnp.linspace(0, x.shape[0] - 1, model.grid_size + 1, dtype=jnp.int32)]
        uniform_step = (x[-1] - x[0] + 2 * margin) / model.grid_size
        grid_uniform = jnp.arange(model.grid_size + 1) * uniform_step + x[0] - margin

        grid = grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive
        grid = jnp.concatenate([
            grid[:1] - uniform_step * jnp.arange(model.spline_order, 0, -1),
            grid,
            grid[-1:] + uniform_step * jnp.arange(1, model.spline_order + 1)
        ])

        # You will typically need to pass this grid to your model's next initialization or state update
        return grid
    
    def regularization_loss(spline_weight, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = jnp.abs(spline_weight).mean(axis=-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -jnp.sum(p * jnp.log(p))
        return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy

    def __call__(self, x):
        """ Forward pass for KANLinear layer. """
        assert x.ndim == 2 and x.shape[1] == self.in_features

        # Applying base linear transformation with activation
        base_output = nn.Dense(features=self.out_features)(self.base_activation(x), self.base_weight)

        # Compute B-splines and reshape for linear transformation
        b_splines = self.b_splines(x, grid=self.grid.value)
        b_splines_flat = b_splines.reshape(x.shape[0], -1)

        # Scaled spline weight and linear transformation
        scaled_weight = self.scaled_spline_weight()
        scaled_weight_flat = scaled_weight.reshape(self.out_features, -1)
        spline_output = nn.Dense(features=self.out_features)(b_splines_flat, scaled_weight_flat)

        # Combining base and spline outputs
        return base_output + spline_output

# Usage in a testing scenario
def test_grid_shape():
    key = jax.random.PRNGKey(0)
    model = KANLinear(in_features=10, out_features=5, key=key)  # adjust init parameters as needed
    x_dummy = jnp.ones((1, 10))  # adjust shape as necessary

    # Initialize the model to setup buffers and parameters
    params = model.init(key, x_dummy)
    
    # Create an output using the apply method to access the grid
    output, updated_state = model.apply(params, x_dummy, method=model.get_grid)
    
    # Now 'output' contains the grid, and you can assert its properties
    assert output.shape == (10, 9), "Grid shape does not match expected values"  # example assertion

if __name__ == "__main__":
    test_grid_shape()

# class KAN(nn.Module):
#     layers_hidden: list
#     grid_size: int = 5
#     spline_order: int = 3
#     scale_noise: float = 0.1
#     scale_base: float = 1.0
#     scale_spline: float = 1.0
#     base_activation: callable = nn.silu
#     grid_eps: float = 0.02
#     grid_range: tuple = (-1, 1)

#     def setup(self):
#         # Create a list of KANLinear layers
#         self.layers = [
#             KANLinear(
#                 in_features=in_features,
#                 out_features=out_features,
#                 grid_size=self.grid_size,
#                 spline_order=self.spline_order,
#                 scale_noise=self.scale_noise,
#                 scale_base=self.scale_base,
#                 scale_spline=self.scale_spline,
#                 base_activation=self.base_activation,
#                 grid_eps=self.grid_eps,
#                 grid_range=self.grid_range,
#             ) for in_features, out_features in zip(self.layers_hidden, self.layers_hidden[1:])
#         ]

#     def __call__(self, x, update_grid=False):
#         # Process each layer
#         for layer in self.layers:
#             if update_grid:
#                 x = layer.update_grid(x)  # This would need to be adapted if you include grid updating
#             x = layer(x)
#         return x

#     def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#         # Sum up regularization losses from all layers
#         return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)