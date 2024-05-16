import jax.numpy as jnp
from flax import linen as nn
import jax

class MultiplyByCachedConstant(nn.Module):
    b: float  # Input to compute 'a'
    c: float  # Input to compute 'a'
    
    def setup(self):
        self.a = self.b * self.c  # Calculate 'a' once during setup

    def __call__(self, x):
        return x * self.a

# Example of initializing the model and using it
model = MultiplyByCachedConstant(b=2.0, c=3.0)
x = jnp.array([1, 2, 3])

# Initialize the model and create 'a'
rng = jax.random.PRNGKey(0)
params = model.init(rng, x)  # This triggers setup and calculates 'a'

# Use the model
y = model.apply(params, x)
print("Output:", y)
