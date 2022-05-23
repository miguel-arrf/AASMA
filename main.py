import gym
import rware
env = gym.make("rware-tiny-4ag-v1")

obs = env.reset()  # a tuple of observations

while True:
    actions = env.action_space.sample()  # the action space can be sampled
    print(actions)  # (1, 0)
    n_obs, reward, done, info = env.step(actions)
    env.render()
