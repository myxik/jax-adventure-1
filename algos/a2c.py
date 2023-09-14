import jax
import sys
import time
import flax
import optax
import numpy as np
import jax.numpy as jnp
import gymnasium as gym

from collections import deque
from jax import random
from flax import linen as nn
from flax.training.train_state import TrainState
from distrax import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import parse_args

class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        return x
    
class Actor(nn.Module):
    action_space: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.action_space)(x)
        return x
    
class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x

@flax.struct.dataclass    
class AgentParams:
    backbone_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

# --- PARAMS ---
args = parse_args()
update_every = 5
algo_name = "A2C"

# --- LOGGER ---
writer = SummaryWriter(f"runs/{algo_name}_{time.time()}")

# --- RNG HANDLE ---
rng = random.PRNGKey(args.seed)
key, _ = random.split(rng, 2)
np.random.seed(args.seed)

# --- ENV HANDLE ---
env = gym.make(args.env_id, autoreset=True)
env = gym.wrappers.RecordEpisodeStatistics(env)

s, _ = env.reset(seed=args.seed)

# --- DEFINING AGENTS/NETWORKS ---
backbone = Network()
actor = Actor(env.action_space.n)
critic = Critic()

backbone_params = backbone.init(key, s[None, :])
agent_state = TrainState.create(
    apply_fn=None,
    params=AgentParams(
        backbone_params,
        actor.init(key, backbone.apply(backbone_params, s[None, :])),
        critic.init(key, backbone.apply(backbone_params, s[None, :])),
    ),
    tx=optax.sgd(learning_rate=3e-4),
)

# @jax.jit
def update(agent_state, observations, actions, returns):
    def reinforce(params):
        embed = backbone.apply(params.backbone_params, observations)

        logits = actor.apply(params.actor_params, embed)
        probs = nn.softmax(logits, axis=0)
        logprobs = jnp.log(probs)

        entropy = jnp.mean(jnp.sum(-probs * logprobs, axis=1))

        logprobs = jnp.take_along_axis(logprobs, actions[:, None], axis=-1)

        values = jax.lax.stop_gradient(critic.apply(params.critic_params, embed))
        scale = returns - values.squeeze(1)

        scaled_logprobs = -logprobs.squeeze(1) * scale
        return jnp.sum(scaled_logprobs, axis=0), entropy
    
    def mse_value(params):
        embed = backbone.apply(params.backbone_params, observations)
        values = critic.apply(params.critic_params, embed)
        return jnp.mean((returns - values.squeeze(1)) ** 2)
    
    (actor_loss, entropy), actor_grads = jax.value_and_grad(reinforce, has_aux=True)(agent_state.params)
    critic_loss, critic_grads = jax.value_and_grad(mse_value)(agent_state.params)
    agent_state = agent_state.apply_gradients(grads=actor_grads)
    agent_state = agent_state.apply_gradients(grads=critic_grads)
    return actor_loss, critic_loss, entropy, agent_state

def calculate_returns(rewards, gamma):
    returns = np.zeros((len(rewards))).astype(rewards.dtype)
    returns[-1] = rewards[-1]
    for i in reversed(range(returns.shape[0] - 1)):
        returns[i] = rewards[i] + gamma * returns[i+1]
    return returns

backbone.apply = jax.jit(backbone.apply)
actor.apply = jax.jit(actor.apply)
critic.apply = jax.jit(critic.apply)

s, _ = env.reset(seed=args.seed)
buffer = deque(maxlen=update_every)

for global_step in range(args.global_steps):
    key, _ = random.split(key, 2)

    # act according to policy
    logits = actor.apply(agent_state.params.actor_params, backbone.apply(agent_state.params.backbone_params, s))
    dist = Categorical(logits=logits)
    a = dist.sample(seed=key)

    s_new, r, term, trun, infos = env.step(a.item())

    buffer.append((s, a, r, term+trun))

    s = s_new
    if term + trun:
        writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
        writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

    if (global_step % update_every == update_every - 1) or (term+trun):
        obs, act, rew, dones = map(np.array, zip(*buffer))

        if dones[-1]:
            rew[-1] = 0
        else:
            rew[-1] = critic.apply(agent_state.params.critic_params, backbone.apply(agent_state.params.backbone_params, obs[-1]))

        returns = calculate_returns(rew, gamma=args.gamma)
        actor_loss, critic_loss, entropy, agent_state = update(agent_state, obs, act, returns)

        writer.add_scalar("healthcheck/actor_loss", actor_loss.item(), global_step)
        writer.add_scalar("healthcheck/critic_loss", critic_loss.item(), global_step)
        writer.add_scalar("healthcheck/policy_entropy", entropy.item(), global_step)
