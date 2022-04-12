import functools
import pathlib
from typing import Any, Callable, Optional, Tuple, Union

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.metrics import tensorboard
from flax.training import checkpoints
from tqdm.autonotebook import trange

from .config import PPOConfig
from .env import GymEnv, GymVecEnv, env_reset
from .models import (ActorCriticCnn, ActorCriticMlp, apply_model,
                     select_log_prob)
from .trajectory import Trajectory, create_trajectory, trajectory_reshape


class PPOTrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: flax.core.FrozenDict

    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    config: PPOConfig = struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        updates, opt_state = self.tx.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=params,
            opt_state=opt_state,
            **kwargs,
        )

    def batch_size(self) -> int:
        return self.config.horizon * self.config.n_actors

    def horizon(self) -> int:
        return self.config.horizon

    def epochs(self) -> int:
        return self.config.epochs

    def mini_batch_size(self) -> int:
        return self.config.mini_batch_size

    def epsilon(self) -> chex.Numeric:
        if isinstance(self.config.epsilon, Callable):
            return self.config.epsilon(self.step)
        return self.config.epsilon

    def learning_rate(self) -> chex.Numeric:
        return self.opt_state.hyperparams["learning_rate"]  # type:ignore

    def gamma(self) -> chex.Numeric:
        return self.config.gamma

    def lam(self) -> chex.Numeric:
        return self.config.gamma

    def c1(self) -> chex.Numeric:
        return self.config.c1

    def c2(self) -> chex.Numeric:
        return self.config.c2

    def n_actors(self) -> int:
        return self.config.n_actors

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Callable,
        params: flax.core.FrozenDict,
        lr: Union[float, optax.Schedule],
        config: PPOConfig,
        max_grad_norm: Optional[float] = None,
    ):
        @optax.inject_hyperparams
        def make_optimizer(learning_rate):
            tx_comps = []
            if max_grad_norm is not None:
                tx_comps.append(optax.clip_by_global_norm(max_grad_norm))
            tx_comps.append(optax.adam(learning_rate))
            return optax.chain(*tx_comps)

        tx = make_optimizer(lr)
        opt_state = tx.init(params)

        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            config=config,
        )


@functools.partial(jax.jit, static_argnums=1)
def loss_fn(
    params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    batch: Tuple,
    epsilon: float,
    c1: float,
    c2: float,
):

    observations, old_log_p, actions, returns, advantages = batch

    logits, values = apply_fn(params, observations)
    values = jnp.squeeze(values)
    log_probs = jax.nn.log_softmax(logits)
    log_p = select_log_prob(actions, log_probs)

    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    # Compute actor loss using conservative policy iteration with an
    # additional clipped surrogate and take minimum between the two.
    # See Eqn.(7) of ArXiv:1707.06347
    prob_ratio = jnp.exp(log_p - old_log_p)
    surrogate1 = advantages * prob_ratio
    surrogate2 = advantages * jnp.clip(prob_ratio, 1.0 - epsilon, 1.0 + epsilon)
    actor_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2), axis=0)

    # Use mean-squared error loss for value function
    critic_loss = c1 * jnp.mean(jnp.square(returns - values), axis=0)
    # Entropy bonus to ensure exploration
    entropy_loss = -c2 * jnp.mean(jnp.sum(-jnp.exp(log_probs) * log_probs, axis=1))

    loss = actor_loss + critic_loss + entropy_loss

    return loss, (actor_loss, critic_loss, entropy_loss)


@functools.partial(jax.jit, static_argnums=2)
def optimize(state: PPOTrainState, traj: Tuple):
    epsilon = state.epsilon()
    c1 = state.c1()
    c2 = state.c2()

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (aloss, closs, eloss)), grads = grad_fn(
        state.params, state.apply_fn, traj, epsilon, c1, c2
    )
    state = state.apply_gradients(grads=grads)  # type: ignore

    return state, loss, aloss, closs, eloss


def train_step(state: PPOTrainState, trajectory: Trajectory, key):
    losses = {
        "total": 0.0,
        "actor": 0.0,
        "critic": 0.0,
        "entropy": 0.0,
    }

    batch_size = state.batch_size()
    mini_batch_size = state.mini_batch_size()

    for _ in range(state.config.epochs):
        key, rng = jax.random.split(key, 2)
        traj_reshaped = trajectory_reshape(trajectory, rng, batch_size, mini_batch_size)
        for traj in zip(*traj_reshaped):
            state, *t_losses = optimize(state, traj)

            losses["total"] += t_losses[0]
            losses["actor"] += t_losses[1]
            losses["critic"] += t_losses[2]
            losses["entropy"] += t_losses[3]

    return state, losses


def evaluate_model(
    state: PPOTrainState,
    env: GymEnv,
    episodes: int,
    key,
    expand_dims=True,
    max_reward=None,
):
    episode_rewards = []
    for _ in range(episodes):
        episode_reward = 0
        observation = env.reset()
        done = False
        while not done:
            if expand_dims:
                observation = jnp.expand_dims(observation, 0)
            logits, _ = apply_model(state.apply_fn, state.params, observation)
            key, rng = jax.random.split(key, 2)
            action = jax.random.categorical(rng, logits)
            if expand_dims:
                observation, reward, done, _ = env.step(int(action[0]))
            else:
                observation, reward, done, _ = env.step(int(action))
            episode_reward += reward

            if max_reward is not None:
                if episode_reward > max_reward:
                    break

        episode_rewards.append(episode_reward)

    return np.average(episode_rewards)


def train(
    model: Union[ActorCriticMlp, ActorCriticCnn],
    learning_rate: Union[float, optax.Schedule],
    train_env: GymVecEnv,
    eval_env: GymEnv,
    key,
    config: PPOConfig,
    model_dir: str,
    log_frequency: int,
    eval_frequency: int,
    eval_episodes: int,
    max_grad_norm: Optional[float] = None,
    checkpoint_dir: Optional[str] = None,
):
    # Initialize model
    observation = env_reset(train_env)
    key, rng = jax.random.split(key, 2)
    params = model.init(rng, observation)
    state = PPOTrainState.create(
        apply_fn=model.apply,
        params=params,
        lr=learning_rate,
        config=config,
        max_grad_norm=max_grad_norm,
    )
    del params

    summary_writer = tensorboard.SummaryWriter(model_dir)
    summary_writer.hparams(config._asdict())

    batch_size = config.horizon * config.n_actors
    frames_per_train_step = batch_size
    num_train_steps = config.total_frames // frames_per_train_step

    reward = 0.0

    horizon = state.config.horizon
    gamma = state.config.gamma
    lam = state.config.lam

    with trange(num_train_steps) as t:
        for step in t:
            frame = step * frames_per_train_step
            t.set_description(f"frame: {step}")

            key, rng1, rng2 = jax.random.split(key, 3)
            trajectory, observation = create_trajectory(
                observation,
                state.apply_fn,
                state.params,
                train_env,
                rng1,
                horizon,
                gamma,
                lam,
                config.clip_reward,
            )
            state, losses = train_step(state, trajectory, rng2)

            if step % log_frequency == 0:
                summary_writer.scalar("train/loss", losses["total"], frame)
                summary_writer.scalar("train/loss-actor", losses["actor"], frame)
                summary_writer.scalar("train/loss-critic", losses["critic"], frame)
                summary_writer.scalar("train/loss-entropy", losses["entropy"], frame)
                summary_writer.scalar(
                    "train/learning-rate", state.learning_rate(), frame
                )
                summary_writer.scalar("train/clipping", state.epsilon(), frame)

            if step % eval_frequency == 0:
                key, rng = jax.random.split(key, 2)
                reward = evaluate_model(state, eval_env, eval_episodes, rng)
                summary_writer.scalar("train/reward", reward, frame)

            t.set_description_str(f"loss: {losses['total']}, reward: {reward}")

            if checkpoint_dir is not None:
                checkpoints.save_checkpoint(checkpoint_dir, state, frame)

    return state.params


def train_from_checkpoint(
    checkpoint: str,
    model: Union[ActorCriticMlp, ActorCriticCnn],
    learning_rate: Union[float, optax.Schedule],
    train_env: GymVecEnv,
    eval_env: GymEnv,
    key,
    config: PPOConfig,
    model_dir: str,
    log_frequency: int,
    eval_frequency: int,
    eval_episodes: int,
    max_grad_norm: Optional[float] = None,
    checkpoint_dir: Optional[str] = None,
):

    # Initialize model
    observation = env_reset(train_env)
    key, rng = jax.random.split(key, 2)
    params = model.init(rng, observation)
    state = PPOTrainState.create(
        apply_fn=model.apply,
        params=params,
        lr=learning_rate,
        config=config,
        max_grad_norm=max_grad_norm,
    )
    del params
    state = checkpoints.restore_checkpoint(checkpoint, state)
    frame = 0
    for f in pathlib.Path(checkpoint).iterdir():
        if f.name.startswith("checkpoint"):
            frame = max(frame, int(f.name.split("_")[1]))

    summary_writer = tensorboard.SummaryWriter(model_dir)
    summary_writer.hparams(config._asdict())

    batch_size = config.horizon * config.n_actors
    frames_per_train_step = batch_size
    num_train_steps = config.total_frames // frames_per_train_step

    reward = 0.0

    horizon = state.config.horizon
    gamma = state.config.gamma
    lam = state.config.lam

    try:
        with trange(num_train_steps) as t:
            for step in t:
                frame += frames_per_train_step
                t.set_description(f"frame: {step}")

                key, rng1, rng2 = jax.random.split(key, 3)
                trajectory, observation = create_trajectory(
                    observation,
                    state.apply_fn,
                    state.params,
                    train_env,
                    rng1,
                    horizon,
                    gamma,
                    lam,
                    config.clip_reward,
                )
                state, losses = train_step(state, trajectory, rng2)

                if step % log_frequency == 0:
                    summary_writer.scalar("train/loss", losses["total"], frame)
                    summary_writer.scalar("train/loss-actor", losses["actor"], frame)
                    summary_writer.scalar("train/loss-critic", losses["critic"], frame)
                    summary_writer.scalar(
                        "train/loss-entropy", losses["entropy"], frame
                    )
                    summary_writer.scalar(
                        "train/learning-rate", state.learning_rate(), frame
                    )
                    summary_writer.scalar("train/clipping", state.epsilon(), frame)

                if step % eval_frequency == 0:
                    key, rng = jax.random.split(key, 2)
                    reward = evaluate_model(state, eval_env, eval_episodes, rng)
                    summary_writer.scalar("train/reward", reward, frame)

                t.set_description_str(f"loss: {losses['total']}, reward: {reward}")

                if checkpoint_dir is not None:
                    checkpoints.save_checkpoint(checkpoint_dir, state, frame)
    except KeyboardInterrupt:
        pass

    return state
