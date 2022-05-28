import argparse
import os
import random
from time import sleep

import gym
from ray.rllib.agents.ppo import PPOTrainer, ppo

import rware
import ray
from ray import tune
from ray.rllib.examples.models.shared_weights_model import (
    SharedWeightsModel1,
    SharedWeightsModel2,
    TF2SharedWeightsModel,
    TorchSharedWeightsModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import register_env


def main():
    layout = """
            .......
            ...x...
            ...x...
            ...x...
            ...x...
            ...x...
            .g...g.
            """

    def env_creator(args):
        return gym.make("rware:rware-tiny-2ag-v1", reward_type=0, layout=layout)

    register_env("rware:rware-tiny-2ag-v1", env_creator)
    env = env_creator({})

    config = {
        "env": "rware:rware-tiny-2ag-v1",
        "env_config": {
            "n_agents": 2,
            "max_steps": 10000,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_sgd_iter": 10,

        "framework": "tf",
    }

    checkpoint_path = "/Users/miguelferreira/ray_results/PPO/PPO_rware-tiny-2ag-v1_989be_00000_0_2022-05-27_15-37-55/checkpoint_000025/checkpoint-25"
    agent = ppo.PPOTrainer(config=config, env="rware:rware-tiny-2ag-v1")
    agent.restore(checkpoint_path)

    # instantiate env class
    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        print(action)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        env.render()
    print("IT'S DONE!!... ", episode_reward)

if __name__ == '__main__':
    main()