import os.path

import gym
import rware
from stable_baselines3 import DQN, PPO, SAC, A2C
from rware import ObserationType
from rware import RewardType
from stable_baselines3.common.env_checker import check_env


def main():
    print("Hi!")

    layout = """
          ..x..
          ..x..
          .....
          .g.g.
          """

    logdir = "logs"
    model_name = "A2C"
    models_dir = f"models/{model_name}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = gym.make("rware-tiny-1ag-v1", layout=layout, observation_type=ObserationType.IMAGE,
                   reward_type=RewardType.GLOBAL,  n_agents=1, sensor_range=4)
    env.reset()
    check_env(env)

    print("sample action: ", env.action_space)
    print("observation space shape: ", env.observation_space.shape)
    print("sample observation: ", env.observation_space.sample())

    model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 5000
    for i in range(1, 30):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name)
        model.save(f"{models_dir}/{TIMESTEPS * i}")
    
    print("APRENDEU!")

    episodes = 10

    for ep in range(episodes):
        print("EPISODE -> ", ep)
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
'''

    done = False
    while not done:
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        env.render()
        print("reward: ", reward)

    env.close()
'''

if __name__ == '__main__':
    main()
