import jax
import jax.numpy as jnp
from flax import linen as nn

class SimpleRNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, inputs, hidden=None):
        if hidden is None:
            hidden = self.variable('state', 'hidden', jnp.zeros, (self.hidden_size,))
        
        # Example weight for the RNN (parameter)
        W_h = self.param('W_h', nn.initializers.normal(), (self.hidden_size, self.hidden_size))
        W_x = self.param('W_x', nn.initializers.normal(), (inputs.shape[-1], self.hidden_size))
        
        new_hidden = jnp.tanh(jnp.dot(hidden.value, W_h) + jnp.dot(inputs, W_x))
        
        # Update the hidden state
        hidden.value = new_hidden
        
        return new_hidden

# Example usage
rnn = SimpleRNN(hidden_size=20)
x = jnp.ones((10,))  # Example input vector

params = rnn.init(jax.random.PRNGKey(0), x)
output, updated_variables = rnn.apply(params, x)
