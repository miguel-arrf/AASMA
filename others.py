import gym
import rware
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

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
        return gym.make("rware:rware-tiny-1ag-v1", layout=layout)

    register_env("rware:rware-tiny-1ag-v1", env_creator)
    env = env_creator({})

    tune.run(
        "PPO",
        stop={"episodes_total": 60000},
        checkpoint_freq=10,
        config={
            # Enviroment specific
            "env": "rware:rware-tiny-1ag-v1",
            # General
            "num_workers": 5,
            # Method specific
        },
    )
