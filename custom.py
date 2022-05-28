import gym
import rware

import os
import pickle

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env

# 2 cases: APEX_DQN, RAINBOW_DQN or everything else
MLPv2methods = ["APEX_DQN", "RAINBOW_DQN"]
METHOD = "RAINBOW_DQN"  # APEX_DQN # ""

env_name = "rware:rware-tiny-1ag-v1"
# path should end with checkpoint-<> data file
# checkpoint_path = "./ray_results/pursuit/APEX_DQN/checkpoint_1000/checkpoint-1000"
# Trainer = ApexTrainer
checkpoint_path = "./ray_results/pursuit/RAINBOW_DQN/checkpoint_6630/checkpoint-6630"
Trainer = DQNTrainer


# TODO: see ray/rllib/rollout.py -- `run` method for checkpoint restoring

# register env -- For some reason, ray is unable to use already registered env in config
def env_creator(args):
    return gym.make("rware:rware-tiny-1ag-v1")


env = env_creator(1)
register_env(env_name, env_creator)


ray.init()

RLAgent = Trainer(env=env_name)

# init obs, action, reward
observations = env.reset()
rewards, action_dict = {}, {}
for agent_id in env.agent_ids:
    assert isinstance(agent_id, int), "Error: agent_ids are not ints."
    # action_dict = dict(zip(env.agent_ids, [np.array([0,1,0]) for _ in range(len(env.agent_ids))])) # no action = [0,1,0]
    rewards[agent_id] = 0

totalReward = 0
done = False
# action_space_len = 3 # for all agents

# TODO: extra parameters : /home/miniconda3/envs/maddpg/lib/python3.7/site-packages/ray/rllib/policy/policy.py

iteration = 0
while not done:
    action_dict = {}
    # compute_action does not cut it. Go to the policy directly
    for agent_id in env.agent_ids:
        # print("id {}, obs {}, rew {}".format(agent_id, observations[agent_id], rewards[agent_id]))
        action, _, _ = RLAgent.get_policy("policy_0").compute_single_action(observations[agent_id], prev_reward=rewards[
            agent_id])  # prev_action=action_dict[agent_id]
        # print(action)
        action_dict[agent_id] = action

    observations, rewards, dones, info = env.step(action_dict)
    env.render()
    totalReward += sum(rewards.values())
    done = any(list(dones.values()))
    print("iter:", iteration, sum(rewards.values()))
    iteration += 1

env.close()
print("done", done, totalReward)
