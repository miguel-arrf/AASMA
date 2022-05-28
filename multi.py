from time import sleep

import gym
from ray.rllib.contrib.maddpg import maddpg

import rware
from ray.rllib.agents.ppo import (
    PPOTrainer,
)
from ray.rllib.policy.policy import PolicySpec
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_trainable, register_env
import argparse

layout = """
.......
...x...
...x...
...x...
...x...
...x...
.g...g.
"""


class CustomStdOut(object):
    def _log_result(self, result):
        if result["training_iteration"] % 50 == 0:
            try:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    result["timesteps_total"],
                    result["episodes_total"],
                    result["episode_reward_mean"],
                    result["policy_reward_mean"],
                    round(result["time_total_s"] - self.cur_time, 3)
                ))
            except:
                pass

            self.cur_time = result["time_total_s"]


def main(args):
    env = gym.make("rware-tiny-2ag-v1", reward_type=0, layout=layout)
    print("--> ", env.agents)
    obs = env.reset()  # a tuple of observations

    MADDPGAgent = maddpg.MADDPGTrainer.with_updates(
        mixins=[CustomStdOut]
    )

    register_trainable("MADDPG", MADDPGAgent)

    def gen_policy(i):
        use_local_critic = [
            args.adv_policy == "ddpg" if i < args.num_adversaries else
            args.good_policy == "ddpg" for i in range(env.num_agents)
        ]
        return (
            None,
            env.observation_space_dict[i],
            env.action_space_dict[i],
            {
                "agent_id": i,
                "use_local_critic": use_local_critic[i],
                "obs_space_dict": env.observation_space_dict,
                "act_space_dict": env.action_space_dict,
            }
        )

    policies = {"policy_%d" % i: gen_policy(i) for i in range(len(env.observation_space_dict))}
    policy_ids = list(policies.keys())

    run_experiments({
        "MADDPG_RLLib": {
            "run": "MADDPG",
            "env": "mpe",
            "stop": {
                "episodes_total": args.num_episodes,
            },
            "checkpoint_freq": args.checkpoint_freq,
            "local_dir": args.local_dir,
            "restore": args.restore,
            "config": {
                # === Log ===
                "log_level": "ERROR",

                # === Environment ===
                "env_config": {
                    "scenario_name": args.scenario,
                },
                "num_envs_per_worker": args.num_envs_per_worker,
                "horizon": args.max_episode_len,

                # === Policy Config ===
                # --- Model ---
                "good_policy": args.good_policy,
                "adv_policy": args.adv_policy,
                "actor_hiddens": [args.num_units] * 2,
                "actor_hidden_activation": "relu",
                "critic_hiddens": [args.num_units] * 2,
                "critic_hidden_activation": "relu",
                "n_step": args.n_step,
                "gamma": args.gamma,

                # --- Exploration ---
                "tau": 0.01,

                # --- Replay buffer ---
                "buffer_size": int(1e6),

                # --- Optimization ---
                "actor_lr": args.lr,
                "critic_lr": args.lr,
                "learning_starts": args.train_batch_size * args.max_episode_len,
                "sample_batch_size": args.sample_batch_size,
                "train_batch_size": args.train_batch_size,
                "batch_mode": "truncate_episodes",

                # --- Parallelism ---
                "num_workers": args.num_workers,
                "num_gpus": args.num_gpus,
                "num_gpus_per_worker": 0,

                # === Multi-agent setting ===
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": ray.tune.function(
                        lambda i: policy_ids[i]
                    )
                },
            },
        },
    }, verbose=0)

    while True:
        actions = env.action_space.sample()  # the action space can be sampled
        print("actions: ", actions)  # (1, 0)
        n_obs, reward, done, info = env.step(actions)
        print("n_obs: ", n_obs)
        print("n_obs_shape: ", n_obs[0])

        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        env.render()

        sleep(0.1)
