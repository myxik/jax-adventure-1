import os
import jax
import time
import flax
import optax
import orbax
import wandb
import numpy as np
import jax.numpy as jnp
import gymnasium as gym

from numba import njit

from einops import rearrange
from jax import random
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from distrax import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from utils import parse_args, RolloutBuffer


# To enable parallelism for CPU
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true" 


class Numpyzi(gym.ObservationWrapper):
    def observation(self, observation):
        return np.asarray(observation)

class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / 255.
        x = nn.Conv(32, kernel_size=(8, 8), strides=4)(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=1)(x)
        x = nn.relu(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        return x
    
class Actor(nn.Module):
    action_space: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_space)(x)
        return x
    
class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

@flax.struct.dataclass    
class AgentParams:
    backbone_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


def make_env(env_id, seed, idx):
    def fn():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env = Numpyzi(env)

        env.action_space.seed(seed)
        return env
    return fn

# --- PARAMS ---
args = parse_args()
update_every = 5
algo_name = f"A2C-Atari-{args.env_id}"
run_name = f"{algo_name}_{time.time()}"

# --- LOGGER ---
if args.track:
    wandb.init(
        project=args.project_name,
        entity=args.entity,
        name=run_name,
        monitor_gym=True,
        sync_tensorboard=True,
        save_code=True,
    )
writer = SummaryWriter(f"runs/{run_name}")

# --- RNG HANDLE ---
rng = random.PRNGKey(args.seed)
key, _ = random.split(rng, 2)
np.random.seed(args.seed)

# --- ENV HANDLE ---
env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed+idx, idx) for idx in range(args.num_envs)])

s, _ = env.reset(seed=args.seed)
s = jnp.asarray(s)

# --- DEFINING AGENTS/NETWORKS ---
backbone = Network()
actor = Actor(env.single_action_space.n)
critic = Critic()

backbone_params = backbone.init(key, s)
agent_state = TrainState.create(
    apply_fn=None,
    params=AgentParams(
        backbone_params,
        actor.init(key, backbone.apply(backbone_params, s)),
        critic.init(key, backbone.apply(backbone_params, s)),
    ),
    tx=optax.chain(
        optax.clip(.1),
        optax.rmsprop(learning_rate=7e-4, initial_scale=1.),
    )
)

@jax.jit
def update(agent_state, observations, actions, returns):
    def reinforce(params):
        embed = backbone.apply(params.backbone_params, observations)

        logits = actor.apply(params.actor_params, embed)
        probs = nn.softmax(logits, axis=1)
        logprobs = jnp.log(probs)

        entropy = jnp.mean(jnp.sum(-probs * logprobs, axis=1))

        logprobs = jnp.take_along_axis(logprobs, actions, axis=-1)

        values = critic.apply(params.critic_params, embed)
        scale = jax.lax.stop_gradient(returns - values)

        scaled_logprobs = -logprobs * scale  # actor loss
        actor_loss = jnp.mean(scaled_logprobs, axis=0).squeeze(0)
        
        critic_loss = jnp.mean((returns - values.squeeze(1)) ** 2)  # critic loss

        loss = actor_loss + 0.5 * critic_loss
        return loss, (actor_loss, critic_loss, entropy)
    
    (general_loss, aux), grads = jax.value_and_grad(reinforce, has_aux=True)(agent_state.params)
    actor_loss, critic_loss, entropy = aux
    agent_state = agent_state.apply_gradients(grads=grads)
    return actor_loss, critic_loss, entropy, agent_state


backbone.apply = jax.jit(backbone.apply)
actor.apply = jax.jit(actor.apply)
critic.apply = jax.jit(critic.apply)

rollouts = RolloutBuffer(update_every)

for global_step in range(args.global_steps):
    for step in range(update_every):
        key, _ = random.split(key, 2)  # handle new random

        # act according to policy
        logits = actor.apply(agent_state.params.actor_params, backbone.apply(agent_state.params.backbone_params, s))
        dist = Categorical(logits=logits)
        a = dist.sample(seed=key)

        s_new, r, term, trun, infos = env.step(a)

        rollouts.push((s, a, r, term))

        s = s_new

        # Gracefully stolen from cleanRL  TODO: rewrite to make more customizable
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if not info:
                    continue
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

    # Calculate last value
    last_value = critic.apply(agent_state.params.critic_params, backbone.apply(agent_state.params.backbone_params, s_new))

    obs, act, returns = rollouts.sample(last_value, args.gamma)

    actor_loss, critic_loss, entropy, agent_state = update(agent_state, obs, act, returns)

    writer.add_scalar("healthcheck/actor_loss", actor_loss.item(), global_step)
    writer.add_scalar("healthcheck/critic_loss", critic_loss.item(), global_step)
    writer.add_scalar("healthcheck/policy_entropy", entropy.item(), global_step)
    
# --- CHECKPOINT ---
ckpt = {"model": agent_state, "data": [s]}

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save(f"checkpoints/{run_name}", ckpt, save_args=save_args)
