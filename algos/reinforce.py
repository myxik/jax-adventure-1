import jax
import time
import optax
import orbax
import numpy as np
import jax.numpy as jnp
import gymnasium as gym

from flax.training import orbax_utils
from jax import random
from flax import linen as nn
from flax.training.train_state import TrainState
from distrax import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils import parse_args

class Policy(nn.Module):
    action_space: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_space)(x)
        return x

# --- PARAMS ---
args = parse_args()
num_episodes = 10
timesteps = 1000
algo_name = "REINFORCE"
run_name = f"{algo_name}_{time.time()}"

# --- LOGGER ---
writer = SummaryWriter(f"runs/{run_name}")

# --- RNG HANDLE ---
rng = random.PRNGKey(args.seed)
key, _ = random.split(rng, 2)
np.random.seed(args.seed)

# --- ENV HANDLE ---
env = gym.make(args.env_id)
env = gym.wrappers.RecordEpisodeStatistics(env)

s, _ = env.reset(seed=args.seed)

agent = Policy(env.action_space.n)
agent_state = TrainState.create(
    apply_fn=agent.apply,
    params=agent.init(key, s),
    tx=optax.sgd(learning_rate=3e-4),
)

@jax.jit
def update(agent_state, observations, actions, returns):
    def reinforce(params):
        logits = agent.apply(params, observations)
        probs = nn.softmax(logits, axis=0)
        logprobs = jnp.log(probs)

        entropy = jnp.mean(jnp.sum(-probs * logprobs, axis=1))

        logprobs = jnp.take_along_axis(logprobs, actions[:, None], axis=-1)
        scaled_logprobs = -logprobs.squeeze(1) * returns
        return jnp.sum(scaled_logprobs, axis=0), entropy
    
    (loss_value, entropy), grads = jax.value_and_grad(reinforce, has_aux=True)(agent_state.params)
    agent_state = agent_state.apply_gradients(grads=grads)
    return loss_value, entropy, agent_state

def calculate_returns(rewards, gamma):
    returns = np.zeros((len(rewards))).astype(rewards.dtype)
    returns[-1] = rewards[-1]
    for i in reversed(range(returns.shape[0] - 1)):
        returns[i] = rewards[i] + gamma * returns[i+1]
    return returns

global_step = 0
agent.apply = jax.jit(agent.apply)

for episode in range(num_episodes):
    s, _ = env.reset()

    observations = []
    actions = []
    rewards = []

    for t in range(timesteps):
        key, _ = random.split(key, 2)

        global_step += 1
        logits = agent.apply(agent_state.params, s)
        dist = Categorical(logits=logits)
        a = dist.sample(seed=key)

        s_new, r, term, trun, infos = env.step(a.item())
        
        observations.append(s)
        actions.append(a.item())
        rewards.append(r)

        s = s_new
        if term + trun:
            writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
            break

    observations = jnp.asarray(observations)
    actions = jnp.asarray(actions)
    rewards = jnp.asarray(rewards)

    returns = calculate_returns(rewards, args.gamma)
    loss_value, entropy, agent_state = update(agent_state, observations, actions, returns)
    writer.add_scalar("healthcheck/loss_value", loss_value.item(), global_step)
    writer.add_scalar("healthcheck/policy_entropy", entropy.item(), global_step)

# --- SAVE CHECKPOINT ---
ckpt = {"model": agent_state, "data": [s]}

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save(f"checkpoints/{run_name}", ckpt, save_args=save_args)
