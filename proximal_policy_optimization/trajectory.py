import functools
from typing import Any, Callable, NamedTuple, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from .env import GymEnv, GymVecEnv, env_step
from .models import action_value_logprob, apply_model


class Trajectory(NamedTuple):
    observations: jnp.ndarray
    log_probs: jnp.ndarray
    actions: jnp.ndarray
    returns: jnp.ndarray
    advantages: jnp.ndarray


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def generalized_advantage_estimation(
    rewards: np.ndarray,
    values: np.ndarray,
    terminals: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert (
        rewards.shape[0] == values.shape[0] - 1
    ), "Values must have one more element than rewards."
    assert (
        rewards.shape[0] == terminals.shape[0]
    ), "Rewards and terminals must have same shape."

    advantages = []
    advantage = 0.0

    for t in reversed(range(len(rewards))):
        # Eqn.(11) and (12) from ArXiv:1707.06347
        delta = rewards[t] + (gamma * values[t + 1] * terminals[t]) - values[t]
        advantage = delta + (gamma * lam * advantage * terminals[t])
        advantages.append(advantage)

    advantages = jnp.array(advantages[::-1])
    returns = advantages + jnp.array(values[:-1])
    return returns, advantages


def create_trajectory(
    initial_observation: jnp.ndarray,
    apply_fn: Callable[..., Any],
    params: flax.core.FrozenDict,
    env: Union[GymEnv, GymVecEnv],
    key,
    horizon: int,
    gamma: float,
    lam: float,
):

    observation = initial_observation

    # Collected quantities
    traj_observations = []
    traj_log_probs = []
    traj_values = []
    traj_rewards = []
    traj_actions = []
    traj_dones = []

    for _ in range(horizon):
        key, rng = jax.random.split(key, 2)
        action, value, log_prob = action_value_logprob(
            apply_fn, params, rng, observation
        )

        traj_actions.append(action)
        traj_values.append(np.array(value))
        traj_observations.append(observation)
        traj_log_probs.append(log_prob)

        observation, reward, done = env_step(action, env)

        traj_rewards.append(reward)
        traj_dones.append(done)

    _, next_value = apply_model(apply_fn, params, observation)
    traj_values.append(np.squeeze(np.array(next_value)))

    traj_rewards = np.array(traj_rewards)
    traj_values = np.array(traj_values)
    traj_terminals = 1 - np.array(traj_dones)

    traj_returns, traj_advantages = generalized_advantage_estimation(
        traj_rewards, traj_values, traj_terminals, gamma, lam
    )

    trajectory = Trajectory(
        observations=jnp.array(traj_observations),
        log_probs=jnp.array(traj_log_probs),
        actions=jnp.array(traj_actions),
        returns=traj_returns,
        advantages=traj_advantages,
    )

    return trajectory, observation


@functools.partial(jax.jit, static_argnums=(2, 3))
def trajectory_reshape(
    trajectory: Trajectory, key, batch_size: int, mini_batch_size: int
):
    permutation = jax.random.permutation(key, batch_size)
    # Flatten and permute
    trajectory = tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:])[permutation], trajectory
    )
    # change shape of trajectory elements to (iterations, minibatch_size)
    iterations = batch_size // mini_batch_size
    trajectory = tree_map(
        lambda x: x.reshape((iterations, mini_batch_size) + x.shape[1:]), trajectory
    )
    return trajectory
