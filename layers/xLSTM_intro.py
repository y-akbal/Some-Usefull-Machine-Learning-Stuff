import jax
import flax
from jax import numpy as jnp
from jax import grad, vmap, pmap
import optax
import flax.linen as nn
from typing import Union, Callable, Optional, Any
rng = jax.random.PRNGKey(0)

class LSTM(nn.Module):

    act_sigmoid: Callable[[jnp.ndarray], jnp.ndarray] = nn.sigmoid
    act_tanh: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    
    @nn.compact
    def __call__(self, x:jnp.ndarray, h_t:jnp.ndarray, c_t:jnp.ndarray, ):
        W_f = self.param("forget", lambda rng, shape: 5*jnp.ones(shape), x.shape[1:])
        W_u = self.param("update", lambda rng, shape: 5*jnp.ones(shape), x.shape[1:])
        W_o = self.param("update", lambda rng, shape: 5*jnp.ones(shape), x.shape[1:])
        
        return self.act_tanh(W@x)
    
unit = LSTM()

params = unit.init(rng, jnp.ones((10,10)))
unit.apply(params, jnp.ones((10,10)))
