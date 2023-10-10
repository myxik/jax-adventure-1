import argparse
import numpy as np
import jax.numpy as jnp

from einops import rearrange
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-e", "--env_id", help="Environment name for your experiments",
                        default="CartPole-v1", type=str, required=False)
    parser.add_argument("-s", "--seed", help="Seed to your experiments",
                        default=777, type=int, required=False)
    parser.add_argument("-t", "--global_steps", help="Global timesteps to pass",
                        default=1_000_000, type=int, required=False)
    parser.add_argument("-g", "--gamma", help="Discount for return",
                        default=0.99, type=float, required=False)
    parser.add_argument("-n", "--num_envs", help="Number of parallel environments",
                        default=2, type=int, required=False)
    parser.add_argument("--track", action="store_true", help="Whether to track wandb")
    parser.add_argument("--project_name", help="Name of the project on wandb",
                        default="JimmyRL", type=str, required=False)
    parser.add_argument("--entity", help="Wandb entity",
                        default="myxik", type=str, required=False)
    return parser.parse_args()


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = rearrange(next_value, "n_envs 1 -> 1 n_envs")
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1 - masks[step])
        returns.insert(0, R)
    return jnp.array(returns)


class RolloutBuffer:  # TODO: rewrite it in jax
    def __init__(self, length):
        self. deque = deque(maxlen=length)

    def push(self, rollout):  # rollout should be in a form of s_i, a_i, r_i, done_i
        """
        Rollout tuple should be in form of tuple with state, action, reward and done on this step
        """
        self.deque.append(rollout)

    def sample(self, last_value, gamma):
        obs, act, rew, dones = map(jnp.array, zip(*self.deque))
        returns = compute_returns(last_value, rew, dones, gamma)

        obs = rearrange(obs, "b n_envs c h w -> (b n_envs) c h w")
        act = rearrange(act, "b n_envs -> (b n_envs) 1")
        returns = rearrange(returns, "b 1 n_envs -> (b n_envs) 1")

        return obs, act, jnp.asarray(returns)
    

class InfosBuffer:  # TODO: rewrite it in numpy 
    def __init__(self):
        self.rs = []
        self.ls = []

    def push(self, infos):
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if not info:
                    continue
                self.rs.append(info["episode"]["r"])
                self.ls.append(info["episode"]["l"])

    def get(self):
        if len(self.rs) == 0:
            return 0., 0.
        r = sum(self.rs) / len(self.rs)
        l = sum(self.ls) / len(self.ls)
        self.rs = []
        self.ls = []
        return r, l
