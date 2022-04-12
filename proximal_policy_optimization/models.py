import functools
from typing import Any, Callable, Tuple, Union

import flax
import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp


class ActorCriticMlp(nn.Module):

    n_hidden: int
    n_actions: int

    def setup(self):
        self.common = nn.Dense(features=self.n_hidden)
        self.actor = nn.Dense(features=self.n_actions)
        self.critic = nn.Dense(1)

    def __call__(self, x):
        x = nn.relu(self.common(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


class ActorCriticCnn(nn.Module):
    n_actions: int
    n_hidden: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))
        self.conv2 = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))
        self.hidden = nn.Dense(features=self.n_hidden)
        self.actor = nn.Dense(features=self.n_actions)
        self.critic = nn.Dense(1)

    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        # Convolutions
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        # Dense
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(self.hidden(x))
        # Actor-Critic
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


@functools.partial(jax.jit, static_argnums=0)
def apply_model(
    apply_fn: Callable[..., Any],
    params: flax.core.FrozenDict,
    observation: Union[jnp.ndarray, np.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return apply_fn(params, observation)


@jax.jit
@jax.vmap
def select_log_prob(action, log_probs):
    return log_probs[action]


@functools.partial(jax.jit, static_argnums=0)
def action_value_logprob(
    apply_fn: Callable[..., Any],
    params: flax.core.FrozenDict,
    key,
    observation: Union[jnp.ndarray, np.ndarray],
):
    logits, value = apply_fn(params, observation)

    action = jax.random.categorical(key, logits)
    log_probs = jax.nn.log_softmax(logits)
    log_prob = select_log_prob(action, log_probs)

    return action, jnp.squeeze(value), log_prob
