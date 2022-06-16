import json
import random
from time import sleep

import gym
import numpy as np
import pyastar2d
from scipy.spatial import distance

from rware import Action


def generate_view(target, current_position, is_carrying, agent):
    length = 15
    height = 15
    matrix = np.ones((length, height), dtype=np.float32)
    shelves = list(tracking_buffer.keys())

    if is_carrying:
        for i in range(length):
            for j in range(height):

                if (i, j) in shelves and (i, j) != target and (i, j) != current_position and ([i, j] not in goal):
                    matrix[i][j] = length * height
                if agent == 1:
                    if (i, j) == tuple(agent2_pos): #and obs2["self"]["carrying_shelf"][0] == 1:
                        matrix[i][j] = length * height
                else:
                    if (i, j) == tuple(agent1_pos): #and obs1["self"]["carrying_shelf"][0] == 1:
                        matrix[i][j] = length * height
        return matrix
    else:
        for i in range(length):
            for j in range(height):
                if agent == 1:
                    if (i, j) == tuple(agent2_pos): #and obs2["self"]["carrying_shelf"][0] == 1:
                        matrix[i][j] = length * height
                else:
                    if (i, j) == tuple(agent1_pos): #and obs1["self"]["carrying_shelf"][0] == 1:
                        matrix[i][j] = length * height
        return matrix

def is_there_collision(pos1, pos2, dir1, dir2):
    if distance.euclidean(pos1, pos2) == 1:
        if (dir1 == 1 and dir2 == 0) or (dir1 == 0 and dir2 == 1) or (dir1 == 2 and dir2 == 3) or (
            dir1 == 3 and dir2 == 2):
            return True
        else:
            return False
    else:
        return False


def min_distance_action(obs, agent_pos, target, agent):
    if tuple(agent_pos) == tuple(target):
        return random.randint(1, 3)
    maze = generate_view(tuple(target), tuple(agent_pos), (obs["self"]["carrying_shelf"][0] == 1), agent)
    print_maze = generate_view(tuple(target), tuple(agent_pos), (obs["self"]["carrying_shelf"][0] == 1), agent)
    t = tuple(target)
    e = tuple(agent_pos)

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



def add_observations(observations, current_pos, agent):
    global tracking_buffer
    global goal
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
                if agent == 1:
                    if (s_x, s_y) != tuple(agent2_pos) and obs2["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
                else:
                    if (s_x, s_y) != tuple(agent1_pos) and obs1["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 1:
                s_x = x
                s_y = y - 1
                if agent == 1:
                    if (s_x, s_y) != tuple(agent2_pos) and obs2["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
                else:
                    if (s_x, s_y) != tuple(agent1_pos) and obs1["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 2:
                s_x = x + 1
                s_y = y - 1
                if agent == 1:
                    if (s_x, s_y) != tuple(agent2_pos) and obs2["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
                else:
                    if (s_x, s_y) != tuple(agent1_pos) and obs1["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 3:
                s_x = x - 1
                s_y = y
                if agent == 1:
                    if (s_x, s_y) != tuple(agent2_pos) and obs2["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
                else:
                    if (s_x, s_y) != tuple(agent1_pos) and obs1["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 5:
                s_x = x + 1
                s_y = y
                if agent == 1:
                    if (s_x, s_y) != tuple(agent2_pos) and obs2["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
                else:
                    if (s_x, s_y) != tuple(agent1_pos) and obs1["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 6:
                s_x = x - 1
                s_y = y + 1
                if agent == 1:
                    if (s_x, s_y) != tuple(agent2_pos) and obs2["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
                else:
                    if (s_x, s_y) != tuple(agent1_pos) and obs1["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 7:
                s_x = x
                s_y = y + 1
                if agent == 1:
                    if (s_x, s_y) != tuple(agent2_pos) and obs2["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
                else:
                    if (s_x, s_y) != tuple(agent1_pos) and obs1["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
            elif sensor_count == 8:
                s_x = x + 1
                s_y = y + 1
                if agent == 1:
                    if (s_x, s_y) != tuple(agent2_pos) and obs2["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
                else:
                    if (s_x, s_y) != tuple(agent1_pos) and obs1["self"]["carrying_shelf"][0] != 1:
                        tracking_buffer[(s_x, s_y)] = {"shelf_requested": sensor["shelf_requested"]}
        sensor_count = sensor_count + 1


walk_around_buffer1 = []
current_target1 = None
seen_it_all1 = False
last_action1 = 1

walk_around_buffer2 = []
current_target2 = None
seen_it_all2 = False
last_action2 = 1


def walk_around_action1(obs, agent_pos, agent):
    global walk_around_buffer1
    global current_target1
    global last_action1
    global seen_it_all1
    # looks for requested shelfes in the buffer
    requested = None
    for known_shelves in tracking_buffer:
        if tracking_buffer[known_shelves]["shelf_requested"][0] == 1:
            requested = tuple(known_shelves)
            return min_distance_action(obs, agent_pos, requested, agent)
    seen_it_all1 = len(walk_around_buffer1) == len(tracking_buffer)
    # if he knows nothign about requetsed shelves
    if not seen_it_all1:
        if current_target1 is None:
            for shelf in list(tracking_buffer.keys()):
                if tuple(shelf) not in walk_around_buffer1:
                    current_target1 = tuple(shelf)
                    return min_distance_action(obs, agent_pos, current_target1, agent)
        else:
            # se ainda nao chegou
            if current_target1 != tuple(agent_pos):
                return min_distance_action(obs, agent_pos, current_target1, agent)
            # se ja chegou
            else:
                # ja chegou ao target, aicionar ao buffer pois ja passou por la
                walk_around_buffer1.append(current_target1)
                # procura novo target
                for shelf in list(tracking_buffer.keys()):
                    if tuple(shelf) not in walk_around_buffer1:
                        current_target1 = tuple(shelf)
                        return min_distance_action(obs, agent_pos, current_target1, agent)
    # ja viu tudo, faz aleatorio
    if last_action1 == 2 or last_action1 == 3:
        last_action1 = 1
        return Action.FORWARD
    else:
        action = random.randint(1, 3)
        last_action1 = action
        return action


def walk_around_action2(obs, agent_pos, agent):
    global walk_around_buffer2
    global current_target2
    global last_action2
    global seen_it_all2

    # looks for requested shelfes in the buffer
    requested = None
    for known_shelves in tracking_buffer:
        if tracking_buffer[known_shelves]["shelf_requested"][0] == 1:
            requested = tuple(known_shelves)
            return min_distance_action(obs, agent_pos, requested, agent)

    seen_it_all2 = len(walk_around_buffer2) == len(tracking_buffer)
    # if he knows nothign about requetsed shelves
    if not seen_it_all2:
        if current_target2 is None:
            for shelf in list(tracking_buffer.keys()):
                if tuple(shelf) not in walk_around_buffer2:
                    current_target2 = tuple(shelf)
                    return min_distance_action(obs, agent_pos, current_target2, agent)
        else:
            # se ainda nao chegou
            if current_target2 != tuple(agent_pos):

                return min_distance_action(obs, agent_pos, current_target2, agent)

            # se ja chegou
            else:
                # ja chegou ao target, aicionar ao buffer pois ja passou por la
                walk_around_buffer2.append(current_target2)

                # procura novo target
                for shelf in list(tracking_buffer.keys()):
                    if tuple(shelf) not in walk_around_buffer2:
                        current_target2 = tuple(shelf)

                        return min_distance_action(obs, agent_pos, current_target2, agent)

    # ja viu tudo, faz aleatorio
    if last_action2 == 2 or last_action2 == 3:
        last_action2 = 1
        return Action.FORWARD
    else:
        action = random.randint(1, 3)
        last_action2 = action
        return action


def choose_action(agent, obs, rewards, done, info) -> Action:
    global loaded1
    global loaded2

    global current_shelf_original_pos1
    global current_shelf_original_pos2

    global knows_goal1
    global knows_goal2

    global tracking_buffer

    global goal1
    global walk_around_buffer1

    global goal2
    global walk_around_buffer2

    global nboxes_agent1
    global nboxes_agent2

    current_pos = obs["self"]["location"]
    add_observations(obs, current_pos, agent)
    # reached goal at least once


    if rewards == 1:
        # adds this goal to the goal list
        if current_episode in delivered_boxes.keys():
            delivered_boxes[current_episode] += 1
        else:
            delivered_boxes[current_episode] = 1

        if agent == 1:
            nboxes_agent1 = nboxes_agent1 + 1
            loaded1 = False
        else:
            nboxes_agent2 = nboxes_agent2 + 1
            loaded2 = False

        return Action.TOGGLE_LOAD

    # if the agent is on a requested shelf, picks it up, updates attributes

    if agent == 1:
        if obs["sensors"][4]["shelf_requested"][0] == 1 and not loaded1:
            walk_around_buffer1 = []  # restarts the walk around buffer
            current_shelf_original_pos1 = current_pos
            loaded1 = True
            tracking_buffer[tuple(current_pos)]["shelf_requested"][0] = 0
            return Action.TOGGLE_LOAD
    else:
        if obs["sensors"][4]["shelf_requested"][0] == 1 and not loaded2:
            walk_around_buffer2 = []  # restarts the walk around buffer
            current_shelf_original_pos2 = current_pos
            loaded2 = True
            tracking_buffer[tuple(current_pos)]["shelf_requested"][0] = 0
            return Action.TOGGLE_LOAD

    # carrying a shelf
    if agent == 1:
        if obs["self"]["carrying_shelf"][0] == 1:
            if loaded1:  # shelf is loaded so goes to goal
                tracking_buffer[tuple(current_shelf_original_pos1)]["shelf_requested"][0] = 0
                if knows_goal:
                    # which is the closest goal?
                    # closest_goal = min(distance.euclidean(current_pos, g) for g in goal)
                    return min_distance_action(obs, current_pos, goal[0], agent)
                else:
                    return walk_around_action1(obs, current_pos, agent)

            else:  # shelf is unloaded
                if tuple(current_pos) == tuple(current_shelf_original_pos1):
                    current_shelf_original_pos1 = []
                    return Action.TOGGLE_LOAD
                else:
                    return min_distance_action(obs, current_pos, current_shelf_original_pos1, agent)
        # not carrying a shelf goes search for the one being asked
        else:
            return walk_around_action1(obs, current_pos, agent)

    else:
        if obs["self"]["carrying_shelf"][0] == 1:
            if loaded2:  # shelf is loaded so goes to goal
                tracking_buffer[tuple(current_shelf_original_pos2)]["shelf_requested"][0] = 0

                if knows_goal:
                    # which is the closest goal?
                    # closest_goal = min(distance.euclidean(current_pos, g) for g in goal)
                    return min_distance_action(obs, current_pos, goal[0], agent)
                else:
                    return walk_around_action2(obs, current_pos, agent)
            else:  # shelf is unloaded
                if tuple(current_pos) == tuple(current_shelf_original_pos2):
                    tracking_buffer[tuple(current_shelf_original_pos2)]["shelf_requested"][0] = 0
                    current_shelf_original_pos2 = []
                    return Action.TOGGLE_LOAD
                else:
                    return min_distance_action(obs, current_pos, current_shelf_original_pos2, agent)
        # not carrying a shelf goes search for the one being asked
        else:
            return walk_around_action2(obs, current_pos, agent)


env = gym.make("rware-tiny-2ag-v1")
nr_episodes = 1000
max_steps = 500

tracking_buffer = dict()

loaded1 = False
loaded2 = False

current_shelf_original_pos1 = []
current_shelf_original_pos2 = []

knows_goal = False
goal = list()

agent1_pos = [0, 0]
agent2_pos = [0, 0]

# MIGUEL'S PART:



inital_pos = None
actionsToExecute = []

atTheRight = False
leftCoords = None
atTheBottom = False
bottomCords = None
wasAtCorner = False
searchDone = False
width = 0
height = 0

# Second agent:
inital_pos_2 = None
actionsToExecute_2 = []

atTheLeft_2 = False
leftCoords_2 = None
atTheBottom_2 = False
bottomCords_2 = None
wasAtCorner_2 = False
searchDone_2 = False


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
    global height

    actions = []
    for i in range(height):
        actions.append(Action.FORWARD)

    return actions


def turnRightMoveUpTurnRight() -> [Action]:
    actions = [Action.RIGHT, Action.FORWARD, Action.FORWARD, Action.RIGHT]

    return actions


def turnLeftMoveUpTurnLeft() -> [Action]:
    actions = [Action.LEFT, Action.FORWARD, Action.FORWARD, Action.LEFT]

    return actions


def addGoal(obs):
    global goal

    if "goal" in obs[0].keys():
        goal_x = obs[0]["goal"][0]
        goal_y = obs[0]["goal"][1]
        if [goal_x, goal_y] not in goal:
            goal.append(tuple([goal_x, goal_y]))

    if "goal" in obs[1].keys():
        goal_x = obs[1]["goal"][0]
        goal_y = obs[1]["goal"][1]
        if [goal_x, goal_y] not in goal:
            goal.append(tuple([goal_x, goal_y]))


def mapMapAndFindGoals(obs) -> Action:
    global tracking_buffer
    global actionsToExecute
    global atTheRight
    global atTheBottom
    global bottomCords
    global leftCoords
    global width
    global height
    global wasAtCorner
    global searchDone

    current_pos = obs["self"]["location"]
    add_observations(obs, current_pos, 1)

    if len(actionsToExecute) != 0:
        actionToExecute = actionsToExecute.pop(0)
        return actionToExecute

    if atTheRight and atTheBottom and not wasAtCorner:
        width = current_pos[0] + 1
        height = current_pos[1] + 1
        wasAtCorner = True
        # Vou virar para cima
        if obs["self"]["direction"] == 1:
            actionsToExecute.append(Action.RIGHT)
            actionsToExecute.append(Action.RIGHT)
        elif obs["self"]["direction"] == 2:
            actionsToExecute.append(Action.RIGHT)
        elif obs["self"]["direction"] == 3:
            actionsToExecute.append(Action.LEFT)

        if height % 2 == 0:
            # É par
            for curAcc in all_the_way():
                actionsToExecute.append(curAcc)

            actionsToExecute.append(Action.LEFT)
            actionsToExecute.append(Action.FORWARD)
            actionsToExecute.append(Action.FORWARD)
            actionsToExecute.append(Action.FORWARD)

            actionsToExecute.append(Action.LEFT)

            for curAcc in all_the_way():
                actionsToExecute.append(curAcc)

        else:
            # É impar
            for curAcc in all_the_way():
                actionsToExecute.append(curAcc)

            actionsToExecute.append(Action.LEFT)
            actionsToExecute.append(Action.FORWARD)
            actionsToExecute.append(Action.FORWARD)
            actionsToExecute.append(Action.FORWARD)

            actionsToExecute.append(Action.LEFT)

            for curAcc in all_the_way():
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


def mapMapAndFindGoalsAgent2(obs) -> Action:
    global tracking_buffer
    global actionsToExecute_2
    global atTheLeft_2
    global atTheBottom_2
    global bottomCords_2
    global leftCoords_2
    global wasAtCorner_2
    global searchDone_2
    global atTheRight
    global atTheBottom

    current_pos = obs["self"]["location"]
    add_observations(obs, current_pos, 2)

    if atTheBottom:
        if len(actionsToExecute_2) != 0:
            actionToExecute_2 = actionsToExecute_2.pop(0)
            return actionToExecute_2

        if atTheLeft_2 and atTheBottom_2 and not wasAtCorner_2:
            wasAtCorner_2 = True
            # Vou virar para cima
            if obs["self"]["direction"] == 1:
                actionsToExecute_2.append(Action.RIGHT)
                actionsToExecute_2.append(Action.RIGHT)
            elif obs["self"]["direction"] == 2:
                actionsToExecute_2.append(Action.RIGHT)
            elif obs["self"]["direction"] == 3:
                actionsToExecute_2.append(Action.LEFT)

            if height % 2 == 0:
                # É par
                for curAcc in all_the_way():
                    actionsToExecute_2.append(curAcc)

                actionsToExecute_2.append(Action.RIGHT)
                actionsToExecute_2.append(Action.FORWARD)
                actionsToExecute_2.append(Action.FORWARD)
                actionsToExecute_2.append(Action.FORWARD)

                actionsToExecute_2.append(Action.RIGHT)

                for curAcc in all_the_way():
                    actionsToExecute_2.append(curAcc)

            else:
                # É impar
                for curAcc in all_the_way():
                    actionsToExecute_2.append(curAcc)

                actionsToExecute_2.append(Action.RIGHT)
                actionsToExecute_2.append(Action.FORWARD)
                actionsToExecute_2.append(Action.FORWARD)
                actionsToExecute_2.append(Action.FORWARD)

                actionsToExecute_2.append(Action.RIGHT)

                for curAcc in all_the_way():
                    actionsToExecute_2.append(curAcc)

            searchDone_2 = True
            return actionsToExecute_2.pop(0)
        else:

            if len(actionsToExecute_2) != 0:
                actionToExecute_2 = actionsToExecute_2.pop(0)
                return actionToExecute_2

            for i in [6, 7, 8]:
                if "BOTTOM" in obs["sensors"][i]["isWall"]:
                    atTheBottom_2 = True

            for i in [0, 3, 6]:
                if "LEFT" in obs["sensors"][i]["isWall"]:
                    atTheLeft_2 = True

            if not atTheLeft_2:
                # Virar Para A ESQUERDA
                if obs["self"]["direction"] == 0:
                    actionsToExecute_2.append(Action.LEFT)
                elif obs["self"]["direction"] == 1:
                    actionsToExecute_2.append(Action.RIGHT)
                elif obs["self"]["direction"] == 3:
                    actionsToExecute_2.append(Action.RIGHT)
                    actionsToExecute_2.append(Action.RIGHT)

                if len(actionsToExecute_2) != 0:
                    return actionsToExecute_2.pop(0)
                else:
                    return Action.FORWARD

            if atTheLeft_2 and not atTheBottom_2:
                # Tenho que virar para BAIXO
                if obs["self"]["direction"] == 0:
                    actionsToExecute_2.append(Action.RIGHT)
                    actionsToExecute_2.append(Action.RIGHT)
                elif obs["self"]["direction"] == 2:
                    actionsToExecute_2.append(Action.RIGHT)
                elif obs["self"]["direction"] == 3:
                    actionsToExecute_2.append(Action.LEFT)

                if len(actionsToExecute_2) != 0:
                    return actionsToExecute_2.pop(0)
                else:
                    return Action.FORWARD

            return Action.FORWARD
    else:
        # tenho que virar para a esquerda!

        if len(actionsToExecute_2) != 0:
            actionToExecute_2 = actionsToExecute_2.pop(0)
            return actionToExecute_2

        # Virar Para A ESQUERDA
        if obs["self"]["direction"] == 0:
            actionsToExecute_2.append(Action.LEFT)
        elif obs["self"]["direction"] == 1:
            actionsToExecute_2.append(Action.RIGHT)
        elif obs["self"]["direction"] == 3:
            actionsToExecute_2.append(Action.RIGHT)
            actionsToExecute_2.append(Action.RIGHT)
        else:
            actionsToExecute_2.append(Action.FORWARD)

        if len(actionsToExecute_2) != 0:
            actionToExecute_2 = actionsToExecute_2.pop(0)
            return actionToExecute_2

        return Action.NOOP

def order_shelfs():
    global tracking_buffer
    aux = tracking_buffer.copy()
    ordered = dict()
    for i in range(len(tracking_buffer)):
        current = (0,0)
        smallest = 99999
        for t in aux:
            sum = t[0] + t[1]
            if sum < smallest:
                smallest = sum
                current = t
        aux.pop(current)
        ordered[current] = tracking_buffer[current]
    tracking_buffer = ordered.copy()

def order_shelfs_2():
    global tracking_buffer
    aux = tracking_buffer.copy()
    ordered = dict()
    for i in range(len(tracking_buffer)):
        current = (0,0)
        smallest = 99999
        for t in aux:
            sum = t[0]
            if sum < smallest:
                smallest = sum
                current = t
        aux.pop(current)
        ordered[current] = tracking_buffer[current]
    tracking_buffer = ordered.copy()

def order_shelfs_3():
    global tracking_buffer
    aux = tracking_buffer.copy()
    ordered = dict()
    for i in range(len(tracking_buffer)):
        current = (0,0)
        smallest = 99999
        for t in aux:
            sum = t[1]
            if sum < smallest:
                smallest = sum
                current = t
        aux.pop(current)
        ordered[current] = tracking_buffer[current]
    tracking_buffer = ordered.copy()



def set_global_variables():
    global inital_pos
    global actionsToExecute

    global atTheRight
    global leftCoords
    global atTheBottom
    global bottomCords
    global wasAtCorner
    global searchDone
    global width
    global height

    global inital_pos_2
    global actionsToExecute_2
    global atTheLeft_2
    global leftCoords_2
    global atTheBottom_2
    global bottomCords_2
    global wasAtCorner_2
    global searchDone_2
    global is_ordered
    ##

    inital_pos = None
    actionsToExecute = []

    atTheRight = False
    leftCoords = None
    atTheBottom = False
    bottomCords = None
    wasAtCorner = False
    searchDone = False
    width = 0
    height = 0

    # Second agent:
    inital_pos_2 = None
    actionsToExecute_2 = []

    atTheLeft_2 = False
    leftCoords_2 = None
    atTheBottom_2 = False
    bottomCords_2 = None
    wasAtCorner_2 = False
    searchDone_2 = False
# END MIGUEL'S PART

delivered_boxes = {}
current_episode = 0

output1 = dict()
output2 = dict()
stepcount = dict()
global nboxes_agent1
global nboxes_agent2
num_steps = 0
ordered = False

for ep in range(20):
    nboxes_agent1 = 0
    nboxes_agent2 = 0
    num_steps = 0
    print("EPISODE: ", ep)
    try:
        set_global_variables()

        current_episode = ep
        observations = env.reset()
        observations, rewards, done, info = env.step([Action.NOOP, Action.NOOP])

        tracking_buffer = dict()
        loaded1 = False
        loaded2 = False

        current_shelf_original_pos1 = []
        current_shelf_original_pos2 = []

        inital_pos_1 = observations[0]["self"]["location"]
        initial_dir_1 = observations[0]["self"]["direction"]
        last_pos_1 = inital_pos_1

        inital_pos_2 = observations[1]["self"]["location"]
        initial_dir_2 = observations[1]["self"]["direction"]
        last_pos_2 = inital_pos_2

        knows_goal = False
        goal = []
        actionsToExecute = []
        actionsToExecute_2 = []
        for step in range(max_steps):
            action1 = None
            action2 = None

            obs1 = observations[0]
            obs2 = observations[1]

            if searchDone_2 and searchDone:
                order_shelfs_3()
                order_shelfs_2()

                #ordered = True

            if not searchDone_2 or not searchDone or (len(actionsToExecute) != 0 or len(actionsToExecute_2) != 0):
                action1 = mapMapAndFindGoals(observations[0])
                action2 = mapMapAndFindGoalsAgent2(observations[1])
                # addGoal(observations)


            else:
                knows_goal = True
                obs1 = observations[0]
                obs2 = observations[1]

                agent1_pos = obs1["self"]["location"]
                agent2_pos = obs2["self"]["location"]
                agent1_dir = obs1["self"]["direction"]
                agent2_dir = obs2["self"]["direction"]

                rewards1 = rewards[0]
                rewards2 = rewards[1]

                if is_there_collision(tuple(agent1_pos), tuple(agent2_pos), agent1_dir, agent2_dir):
                    action1 = random.randint(1, 3)
                    action2 = random.randint(1, 3)
                else:
                    action1 = choose_action(1, obs1, rewards1, done, info)
                    action2 = choose_action(2, obs2, rewards2, done, info)

            if nboxes_agent1 + nboxes_agent2 < 4:
                num_steps = step
            env.render()
            sleep(0.1)
            observations, rewards, done, info = env.step([action1, action2])
    except:
        print("exception")
        #nr_episodes = nr_episodes + 1
    output1[ep] = nboxes_agent1
    output2[ep] = nboxes_agent2
    stepcount[ep] = num_steps
#
print("delivered boxes: ", delivered_boxes)
