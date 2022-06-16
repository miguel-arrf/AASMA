import random
from time import sleep

import gym
import astar
from rware import ObserationType
from rware import RewardType
from rware import Action
from rware import Action
from scipy.spatial import distance
import numpy as np
import time
import pyastar2d

import os
import pickle


pickedUpAShelf = 0
stepsToPickup3 = 0
steps = 0


def generate_view(target, current_position, is_carrying):
    length = 15
    height = 15
    matrix = np.ones((length, height), dtype=np.float32)
    shelves = list(tracking_buffer.keys())
    if is_carrying:
        for i in range(length):
            for j in range(height):
                if (i, j) in shelves and (i, j) != target and (i, j) != current_position and ([i, j] not in goal):
                    matrix[i][j] = length * height

        return matrix
    else:
        return matrix


def min_distance_action(obs, agent_pos, target):
    if tuple(agent_pos) == tuple(target):
        return random.randint(1, 3)

    maze = generate_view(tuple(target), tuple(agent_pos), (obs["self"]["carrying_shelf"][0] == 1))
    print_maze = generate_view(tuple(target), tuple(agent_pos), (obs["self"]["carrying_shelf"][0] == 1))
    t = tuple(target)
    e = tuple(agent_pos)

    print_maze[agent_pos[0]][agent_pos[1]] = 2
    print_maze[target[0]][target[1]] = 3

    # path = astar.astar(maze,tuple(agent_pos),tuple(target))
    path = pyastar2d.astar_path(maze, tuple(agent_pos), tuple(target), allow_diagonal=False)
    next_pos = tuple(path[1])

    selected_action = Action.NOOP
    if next_pos == (agent_pos[0], agent_pos[1] - 1):  # up
        if obs["self"]["direction"] == 0:
            selected_action = Action.FORWARD
        if obs["self"]["direction"] == 1:
            selected_action = Action.LEFT
        if obs["self"]["direction"] == 2:
            selected_action = Action.RIGHT
        if obs["self"]["direction"] == 3:
            selected_action = Action.LEFT
    elif next_pos == (agent_pos[0], agent_pos[1] + 1):  # down
        if obs["self"]["direction"] == 0:  # UP = 0
            selected_action = Action.LEFT
        if obs["self"]["direction"] == 1:
            selected_action = Action.FORWARD
        if obs["self"]["direction"] == 2:  # LEFT = 2
            selected_action = Action.LEFT
        if obs["self"]["direction"] == 3:  # RIGHT = 3
            selected_action = Action.RIGHT
    elif next_pos == (agent_pos[0] - 1, agent_pos[1]):  # left
        if obs["self"]["direction"] == 0:
            selected_action = Action.LEFT
        if obs["self"]["direction"] == 1:
            selected_action = Action.RIGHT
        if obs["self"]["direction"] == 2:
            selected_action = Action.FORWARD
        if obs["self"]["direction"] == 3:
            selected_action = Action.LEFT
    elif next_pos == (agent_pos[0] + 1, agent_pos[1]):
        if obs["self"]["direction"] == 0:
            selected_action = Action.RIGHT
        if obs["self"]["direction"] == 1:
            selected_action = Action.LEFT
        if obs["self"]["direction"] == 2:
            selected_action = Action.LEFT
        if obs["self"]["direction"] == 3:
            selected_action = Action.FORWARD

    return selected_action


def min_distance_action2(obs, agent_pos, target):
    # para ja fica euclidean distance mas gostava de por a* ou dijkstra
    # https: // pypi.org / project / Dijkstar /

    # FORWARD = 1
    # LEFT = 2
    # RIGHT = 3

    # UP = 0
    # DOWN = 1
    # LEFT = 2
    # RIGHT = 3

    min_distance = 999999
    selected_action = Action.NOOP

    for mov in range(0, 4):
        if mov == 0:  # up
            projected_pos = [agent_pos[0], agent_pos[1] - 1]
            dist = distance.euclidean(projected_pos, target)
            block_in_front = (obs["self"]["carrying_shelf"][0] == 1) and (
                    tuple(projected_pos) not in list(tracking_buffer.keys()))
            if dist < min_distance:
                min_distance = dist
                if obs["self"]["direction"] == 0 and not block_in_front:
                    selected_action = Action.FORWARD
                if obs["self"]["direction"] == 1:
                    selected_action = Action.LEFT
                if obs["self"]["direction"] == 2:
                    selected_action = Action.RIGHT
                if obs["self"]["direction"] == 3:
                    selected_action = Action.LEFT
        if mov == 1:  # down
            projected_pos = [agent_pos[0], agent_pos[1] + 1]
            dist = distance.euclidean(projected_pos, target)
            block_in_front = (obs["self"]["carrying_shelf"][0] == 1) and (
                    tuple(projected_pos) in list(tracking_buffer.keys()))
            if dist < min_distance:
                min_distance = dist
                if obs["self"]["direction"] == 0:  # UP = 0
                    selected_action = Action.LEFT
                if obs["self"]["direction"] == 1:
                    selected_action = Action.FORWARD
                if obs["self"]["direction"] == 2:  # LEFT = 2
                    selected_action = Action.LEFT
                if obs["self"]["direction"] == 3:  # RIGHT = 3
                    selected_action = Action.RIGHT
        if mov == 2:  # left
            projected_pos = [agent_pos[0] - 1, agent_pos[1]]
            dist = distance.euclidean(projected_pos, target)
            block_in_front = (obs["self"]["carrying_shelf"][0] == 1) and (
                    tuple(projected_pos) not in list(tracking_buffer.keys()))
            if dist < min_distance:
                min_distance = dist
                if obs["self"]["direction"] == 0:
                    selected_action = Action.LEFT
                if obs["self"]["direction"] == 1:
                    selected_action = Action.RIGHT
                if obs["self"]["direction"] == 2 and not block_in_front:
                    selected_action = Action.FORWARD
                if obs["self"]["direction"] == 3:
                    selected_action = Action.LEFT
        if mov == 3:  # right
            projected_pos = [agent_pos[0] + 1, agent_pos[1]]
            dist = distance.euclidean(projected_pos, target)
            block_in_front = (obs["self"]["carrying_shelf"][0] == 1) and (
                    tuple(projected_pos) not in list(tracking_buffer.keys()))
            if dist < min_distance:
                min_distance = dist
                if obs["self"]["direction"] == 0:
                    selected_action = Action.RIGHT
                if obs["self"]["direction"] == 1:
                    selected_action = Action.LEFT
                if obs["self"]["direction"] == 2:
                    selected_action = Action.DOWN
                if obs["self"]["direction"] == 3 and not block_in_front:
                    selected_action = Action.FORWARD
    return selected_action


# para ja ainda so guarda a localizacao de shelfes e se estao ou nao requested
def add_observations(observations, current_pos):
    global tracking_buffer
    sensor_count = 0
    x = current_pos[0]
    y = current_pos[1]

    if "goal" in observations.keys():
        goal_x = observations["goal"][0]
        goal_y = observations["goal"][1]

        if [goal_x, goal_y] not in goal:
            goal.append([goal_x, goal_y])

    for sensor in list(observations["sensors"]):
        if sensor["has_shelf"][0] == 1:
            # adiciona a shelf ao tracking buffer e o seu status e da lhe um nome
            if sensor_count == 0:
                s_x = x - 1
                s_y = y - 1
                tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 1:
                s_x = x
                s_y = y - 1
                tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 2:
                s_x = x + 1
                s_y = y - 1
                tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 3:
                s_x = x - 1
                s_y = y
                tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 5:
                s_x = x + 1
                s_y = y
                tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 6:
                s_x = x - 1
                s_y = y + 1
                tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 7:
                s_x = x
                s_y = y + 1
                tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 8:
                s_x = x + 1
                s_y = y + 1
                tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
        sensor_count = sensor_count + 1


walk_around_buffer = []
current_target = None
seen_it_all = False
last_action = 1


def walk_around_action(obs, agent_pos):
    global walk_around_buffer
    global current_target
    global last_action

    # looks for requested shelfes in the buffer
    requested = None
    for known_shelves in tracking_buffer:
        if tracking_buffer[known_shelves]["shelf_requested"][0] == 1:
            requested = tuple(known_shelves)
            return min_distance_action(obs, agent_pos, requested)

    seen_it_all = len(walk_around_buffer) == len(tracking_buffer)
    # if he knows nothign about requetsed shelves
    if not seen_it_all:
        if current_target is None:
            for shelf in list(tracking_buffer.keys()):
                if tuple(shelf) not in walk_around_buffer:
                    current_target = tuple(shelf)
                    return min_distance_action(obs, agent_pos, current_target)
        else:
            # se ainda nao chegou
            if current_target != tuple(agent_pos):
                return min_distance_action(obs, agent_pos, current_target)

            # se ja chegou
            else:
                # ja chegou ao target, aicionar ao buffer pois ja passou por la
                walk_around_buffer.append(current_target)

                # procura novo target
                for shelf in list(tracking_buffer.keys()):
                    if tuple(shelf) not in walk_around_buffer:
                        current_target = tuple(shelf)
                        return min_distance_action(obs, agent_pos, current_target)

    # ja viu tudo, faz aleatorio
    if last_action == 2 or last_action == 3:
        last_action = 1
        return Action.FORWARD
    else:
        action = random.randint(1, 3)
        last_action = action
        return action


def walk_around_action2(obs, agent_pos):
    global last_action
    requested = None
    for known_shelves in tracking_buffer:
        if tracking_buffer[known_shelves]["shelf_requested"][0] == 1:
            requested = list(known_shelves)
    if requested is not None:
        return min_distance_action(obs, agent_pos, requested)
    else:
        if last_action == 2 or last_action == 3:
            last_action = 1
            return Action.FORWARD
        else:
            action = random.randint(1, 3)
            last_action = action
            return action


def choose_action(obs, rewards, done, info) -> int:
    global loaded
    global current_shelf_original_pos
    global knows_goal
    global goal
    global tracking_buffer
    global walk_around_buffer
    global pickedUpAShelf
    global stepsToPickup3

    current_pos = obs["self"]["location"]
    add_observations(obs, current_pos)

    # reached goal at least once
    if rewards == 100:
        # adds this goal to the goal list
        knows_goal = True
        if tuple(current_pos) not in goal:
            goal.append(tuple(current_pos))
        loaded = False
        pickedUpAShelf += 1

        if pickedUpAShelf == 3:
            stepsToPickup3 = steps

        return Action.TOGGLE_LOAD

    # if the agent is on a requested shelf, picks it up, updates attributes
    if obs["sensors"][4]["shelf_requested"][0] == 1 and not loaded:
        walk_around_buffer = []  # restarts the walk around buffer
        current_shelf_original_pos = current_pos
        loaded = True
        return Action.TOGGLE_LOAD

    # carrying a shelf
    if obs["self"]["carrying_shelf"][0] == 1:
        if loaded:  # shelf is loaded so goes to goal
            if knows_goal:
                # which is the closest goal?
                # closest_goal = min(distance.euclidean(current_pos, g) for g in goal)
                return min_distance_action(obs, current_pos, goal[0])
            else:
                return walk_around_action(obs, current_pos)

        else:  # shelf is unloaded
            if tuple(current_pos) == tuple(current_shelf_original_pos):
                tracking_buffer[tuple(current_shelf_original_pos)]["shelf_requested"][0] = 0
                current_shelf_original_pos = []
                return Action.TOGGLE_LOAD
            else:
                return min_distance_action(obs, current_pos, current_shelf_original_pos)

    # not carrying a shelf goes search for the one being asked
    else:
        return walk_around_action(obs, current_pos)


env = gym.make("rware-tiny-1ag-v1")
nr_episodes = 100
max_steps = 500

tracking_buffer = dict()
loaded = False
current_shelf_original_pos = []
knows_goal = False
goal = list()

# INITIAL PHASE

rotating = False
stepsToRotate = 0
actionsToExecute = []

atTheRight = False
leftCoords = None
atTheBottom = False
bottomCords = None
verticalPercolation = True
wasAtCorner = False
searchDone = False
width = 0
height = 0


def reset_global():
    global rotating
    global stepsToRotate
    global actionsToExecute
    global atTheRight
    global leftCoords
    global atTheBottom
    global bottomCords
    global verticalPercolation
    global wasAtCorner
    global searchDone
    global width
    global height
    # --

    rotating = False
    stepsToRotate = 0
    actionsToExecute = []

    atTheRight = False
    leftCoords = None
    atTheBottom = False
    bottomCords = None
    verticalPercolation = True
    wasAtCorner = False
    searchDone = False
    width = 0
    height = 0


def turn_left(obs) -> [Action]:
    actions = []
    if obs["self"]["direction"] == 0:
        actions.append(Action.LEFT)
    elif obs["self"]["direction"] == 1:
        actions.append(Action.RIGHT)
    elif obs["self"]["direction"] == 3:
        actions.append(Action.RIGHT)
        actions.append(Action.RIGHT)
    return actions


def turn_right(obs) -> [Action]:
    actions = []
    if obs["self"]["direction"] == 0:
        actions.append(Action.RIGHT)
    elif obs["self"]["direction"] == 1:
        actions.append(Action.LEFT)
    elif obs["self"]["direction"] == 2:
        actions.append(Action.LEFT)
        actions.append(Action.LEFT)
    return actions


def turn_up(obs) -> [Action]:
    actions = []
    if obs["self"]["direction"] == 1:
        actions.append(Action.LEFT)
        actions.append(Action.LEFT)
    elif obs["self"]["direction"] == 2:
        actions.append(Action.RIGHT)
    elif obs["self"]["direction"] == 3:
        actions.append(Action.LEFT)
    return actions


def all_the_way() -> [Action]:
    global width

    actions = []
    for i in range(width - 3):
        actions.append(Action.FORWARD)

    return actions


def turnRightMoveUpTurnRight() -> [Action]:
    actions = [Action.RIGHT, Action.FORWARD, Action.FORWARD, Action.RIGHT]

    return actions


def turnLeftMoveUpTurnLeft() -> [Action]:
    actions = [Action.LEFT, Action.FORWARD, Action.FORWARD, Action.LEFT]

    return actions


def mapMapAndFindGoals(obs) -> Action:
    global tracking_buffer
    global rotating
    global stepsToRotate
    global actionsToExecute
    global atTheRight
    global atTheBottom
    global bottomCords
    global leftCoords
    global verticalPercolation
    global width
    global height
    global wasAtCorner
    global searchDone

    current_pos = obs["self"]["location"]
    add_observations(obs, current_pos)

    if len(actionsToExecute) != 0:
        actionToExecute = actionsToExecute.pop(0)
        return actionToExecute

    if atTheRight and atTheBottom and not wasAtCorner:
        width = current_pos[0] + 1
        height = current_pos[1] + 1
        wasAtCorner = True
        # Vou virar para a esquerda
        if obs["self"]["direction"] == 0:
            actionsToExecute.append(Action.LEFT)
        elif obs["self"]["direction"] == 1:
            actionsToExecute.append(Action.RIGHT)
        elif obs["self"]["direction"] == 3:
            actionsToExecute.append(Action.RIGHT)
            actionsToExecute.append(Action.RIGHT)

        actionsToExecute.append(Action.FORWARD)
        actionsToExecute.append(Action.FORWARD)
        actionsToExecute.append(Action.FORWARD)

        if height % 2 == 0:
            # É par
            for i in range(int(height / 4)):
                for curAcc in all_the_way():
                    actionsToExecute.append(curAcc)

                for curAcc in turnRightMoveUpTurnRight():
                    actionsToExecute.append(curAcc)

                for curAcc in all_the_way():
                    actionsToExecute.append(curAcc)

                for curAcc in turnLeftMoveUpTurnLeft():
                    actionsToExecute.append(curAcc)
        else:
            # É impar
            for i in range(int(height / 3)):
                for curAcc in all_the_way():
                    actionsToExecute.append(curAcc)

                for curAcc in turnRightMoveUpTurnRight():
                    actionsToExecute.append(curAcc)

                for curAcc in all_the_way():
                    actionsToExecute.append(curAcc)

                for curAcc in turnLeftMoveUpTurnLeft():
                    actionsToExecute.append(curAcc)
        searchDone = True
        return actionsToExecute.pop(0)
    else:

        if len(actionsToExecute) != 0:
            actionToExecute = actionsToExecute.pop(0)
            return actionToExecute

        for i in [6, 7, 8]:
            if "BOTTOM" in obs["sensors"][i]["isWall"]:
                atTheBottom = True

        for i in [2, 5, 8]:
            if "RIGHT" in obs["sensors"][i]["isWall"]:
                atTheRight = True

        if not atTheRight:
            # TenhoQueVirarParaAdireita
            if obs["self"]["direction"] == 0:
                actionsToExecute.append(Action.RIGHT)
            elif obs["self"]["direction"] == 1:
                actionsToExecute.append(Action.LEFT)
            elif obs["self"]["direction"] == 2:
                actionsToExecute.append(Action.RIGHT)
                actionsToExecute.append(Action.RIGHT)

            if len(actionsToExecute) != 0:
                return actionsToExecute.pop(0)
            else:
                return Action.FORWARD

        if atTheRight and not atTheBottom:
            # Tenho que virar para baixo
            if obs["self"]["direction"] == 0:
                actionsToExecute.append(Action.LEFT)
                actionsToExecute.append(Action.LEFT)
            elif obs["self"]["direction"] == 2:
                actionsToExecute.append(Action.LEFT)
            elif obs["self"]["direction"] == 3:
                actionsToExecute.append(Action.RIGHT)

            if len(actionsToExecute) != 0:
                return actionsToExecute.pop(0)
            else:
                return Action.FORWARD

        return Action.FORWARD


# END INITIAL PHASE

shelvesPickedUp = []
stepsToPickup3List = []

for ep in range(nr_episodes):
    try:
        reset_global()
        observations = env.reset()
        observations, rewards, done, info = env.step(Action.NOOP)

        tracking_buffer = dict()
        loaded = False
        current_shelf_original_pos = []
        knows_goal = True
        goal = []
        steps = 0
        stepsToPickup3 = 0
        pickedUpAShelf = 0

        for step in range(max_steps):
            steps += 1
            action = None
            if not searchDone or len(actionsToExecute) != 0:
                action = mapMapAndFindGoals(observations)
            else:
                action = choose_action(observations, rewards, done, info)

            observations, rewards, done, info = env.step(action)
            env.render()
            sleep(0.01)
        print("{}<->{}".format(pickedUpAShelf, stepsToPickup3))
    except IndexError:
        nr_episodes += 1

