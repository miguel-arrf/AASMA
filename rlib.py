import gym
import rware
from rware import ObserationType
from rware import RewardType

import os
import pickle

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

# 2 cases: APEX_DQN, RAINBOW_DQN or everything else
MLPv2methods = ["APEX_DQN", "RAINBOW_DQN"]
METHOD = "RAINBOW_DQN"  # APEX_DQN # ""

env_name = "rware:rware-tiny-1ag-v1"
# path should end with checkpoint-<> data file
# checkpoint_path = "./ray_results/pursuit/APEX_DQN/checkpoint_1000/checkpoint-1000"
# Trainer = ApexTrainer
checkpoint_path = "./ray_results/pursuit/RAINBOW_DQN/checkpoint_6630/checkpoint-6630"
Trainer = PPOTrainer

# TODO: see ray/rllib/rollout.py -- `run` method for checkpoint restoring

# register env -- For some reason, ray is unable to use already registered env in config

layout = """
          ..x..
          ..x..
          .....
          .g.g.
          """


def env_creator(args):
    return gym.make("rware-tiny-1ag-v1", layout=layout, observation_type=ObserationType.FLATTENED,
                    reward_type=RewardType.GLOBAL, n_agents=1, sensor_range=4, give_reward_help=False)


env = env_creator(1)
register_env(env_name, env_creator)

ray.init(num_cpus=3, ignore_reinit_error=True, log_to_driver=False)
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 3
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config['model']['fcnet_hiddens'] = [100, 100]
config[
    'num_cpus_per_worker'] = 0  # This avoids running out of resources in the notebook environment when this cell is re-executed

agent = Trainer(config, env=env_name)

for i in range(1000):
    result = agent.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = agent.save()
        print("checkpoint saved at", checkpoint)

env.close()
