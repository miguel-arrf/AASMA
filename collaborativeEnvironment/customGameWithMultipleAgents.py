from collections import Counter
from time import sleep

import gym

from heuristic1 import HeuristicAgent1
from heuristic2 import HeuristicAgent2
from heuristic3 import HeuristicAgent3
from heuristic4 import HeuristicAgent4
from heuristic5 import HeuristicAgent5

env = gym.make("rware-tiny-2ag-v1")

number_episodes = 100
number_max_steps = 500


def simulateT(Agent1, Agent2, render=True):
    obs = env.reset()  # a tuple of observations

    ag1Obs = obs[0]
    ag2Obs = obs[1]

    maximumReward = 0

    agentShelves = ag1Obs['sensors']
    for shelf in agentShelves:
        if shelf != {}:
            maximumReward += shelf['shelf_level'][0]

    timeToWait = 0.05
    if render:
        env.render()
        sleep(timeToWait)

    totalReward = [0, 0]
    dones = [False, False]
    agent1 = Agent1(1)
    agent2 = Agent2(2)

    reward_ag1 = []
    reward_ag2 = []
    whoCatchedList = []
    whoCatchedListTogether = []
    i = 0
    while i < number_max_steps and not any(dones):
        ag1Action = agent1.get_action(ag1Obs)
        ag2Action = agent2.get_action(ag2Obs)

        obs, reward, dones, info = env.step([ag1Action, ag2Action])
        ag1Obs = obs[0]
        ag2Obs = obs[1]

        if render:
            env.render()
            sleep(timeToWait)

        totalReward[0] += reward[0]
        totalReward[1] += reward[1]

        for item in info['alone']:
            whoCatchedList.append(item)

        for item in info['together']:
            whoCatchedListTogether.append(item)

        i += 1
        reward_ag1.append(totalReward[0])
        reward_ag2.append(totalReward[1])

    # while i < number_max_steps:
    #    reward_ag1.append(totalReward[0])
    #    reward_ag2.append(totalReward[1])
    #    i += 1

    return totalReward, dones, reward_ag1, reward_ag2, maximumReward, i, Counter(whoCatchedList), Counter(
        whoCatchedListTogether)


def get_heuristic(heuristic):
    if heuristic == "h1":
        return HeuristicAgent1
    if heuristic == "h2":
        return HeuristicAgent2
    if heuristic == "h3":
        return HeuristicAgent3
    if heuristic == "h4":
        return HeuristicAgent4
    return HeuristicAgent5


def simulate(first_heuristic, second_heuristic):
    print()
    agent1Selected = get_heuristic(first_heuristic)
    agent2Selected = get_heuristic(second_heuristic)

    for _ in range(20):
        simulateT(agent1Selected, agent2Selected)
    exit()
