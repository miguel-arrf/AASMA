from time import sleep

import gym
import rware
from ray.rllib.agents.ppo import (
    PPOTrainer,
)
from ray.rllib.policy.policy import PolicySpec

layout = """
.......
...x...
...x...
...x...
...x...
...x...
.g...g.
"""

env = gym.make("rware-tiny-2ag-v1", reward_type=0, layout=layout)
print("--> ", env.agents)
obs = env.reset()  # a tuple of observations
print("reset:")
print(obs)
'''
trainer = PPOTrainer(env=env, config={
    "multiagent": {
        "policies": {
            # Use the PolicySpec namedtuple to specify an individual policy:
            "car1": PolicySpec(
                policy_class=None,  # infer automatically from Trainer
                observation_space=None,  # infer automatically from env
                action_space=None,  # infer automatically from env
                config={"gamma": 0.85},  # use main config plus <- this override here
                ),  # alternatively, simply do: `PolicySpec(config={"gamma": 0.85})`

            # Deprecated way: Tuple specifying class, obs-/action-spaces,
            # config-overrides for each policy as a tuple.
            # If class is None -> Uses Trainer's default policy class.
            "car2": (None, car_obs_space, car_act_space, {"gamma": 0.99}),

            # New way: Use PolicySpec() with keywords: `policy_class`,
            # `observation_space`, `action_space`, `config`.
            "traffic_light": PolicySpec(
                observation_space=tl_obs_space,  # special obs space for lights?
                action_space=tl_act_space,  # special action space for lights?
                ),
        },
        "policy_mapping_fn":
            lambda agent_id, episode, worker, **kwargs:
                "traffic_light"  # Traffic lights are always controlled by this policy
                if agent_id.startswith("traffic_light_")
                else random.choice(["car1", "car2"])  # Randomly choose from car policies
    },
})

while True:
    print(trainer.train())
'''

while True:
    actions = env.action_space.sample()  # the action space can be sampled
    print("actions: ", actions)  # (1, 0)
    n_obs, reward, done, info = env.step(actions)
    print("n_obs: ", n_obs)

    print("reward: ", reward)
    print("done: ", done)
    print("info: ", info)
    env.render()

    sleep(0.1)
