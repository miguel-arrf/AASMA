import math
import random

import gym
import numpy as np
import pyastar2d

from rware import Action

######################################################
#      AGENT HEURISTIC 3                             #
#      GOES TO THE CLOSEST BOX FROM HIGHER LEVEL     #
######################################################
# And if there are none he goes to any!



class HeuristicAgent3():

    def __init__(self, agentNumber):
        self.last_pos = None
        self.timeStepsInSamePosition = 0
        self.is_stuck_treshold = 10

        self.agent_level = 0

        self.agentNumber = agentNumber
        self.width = 8
        self.height = 8
        self.shelves = []
        self.mapState = np.ones((self.width, self.height), dtype=np.float32)
        self.currentPos: (int, int) = None
        self.nextTarget = []

    def hamming_distance(self, state1, state2):
        x1, y1 = state1
        x2, y2 = state2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def check_if_shelf_already_in_shelves(self, shelfToVerify):
        for shelf in self.shelves:
            if shelf[0] == shelfToVerify[0] and shelf[1] == shelfToVerify[1]:
                return True
        return False

    def is_in_shelf(self):
        for shelf in self.shelves:
            if self.currentPos[0] == shelf[0] and self.currentPos[1] == shelf[1]:
                return True
        return False

    def is_stuck(self):
        if self.currentPos == self.last_pos:
            self.timeStepsInSamePosition += 1
        else:
            self.timeStepsInSamePosition = 0

        if self.timeStepsInSamePosition > self.is_stuck_treshold:
            return True
        return False

    def exists_shelf_with_agent_level(self):
        shelfs = []
        for shelf in self.shelves:
            if shelf[2] == self.agent_level:
                shelfs.append(shelf)
        return shelfs


    def exists_shelf_with_lower_agent_level(self):
        shelfs = []
        for shelf in self.shelves:
            if shelf[2] < self.agent_level:
                shelfs.append(shelf)
        return shelfs

    def exists_shelf_with_higher_agent_level(self):
        shelfs = []
        for shelf in self.shelves:
            if shelf[2] > self.agent_level:
                shelfs.append(shelf)
        return shelfs

    def get_current_shelf(self):
        for shelf in self.shelves:
            if self.currentPos[0] == shelf[0] and self.currentPos[1] == shelf[1]:
                return shelf
        return None

    def get_action(self, observations):
        self.shelves = []
        self.make_map_from_agents(observations)

        if self.is_stuck():
            random_action = np.random.randint(0, 5)
            return Action(random_action)

        self.last_pos = self.currentPos

        if self.is_in_shelf() and len(self.shelves) == 0:
            return Action.TOGGLE_LOAD

        if self.is_in_shelf():
            current_shelf = self.get_current_shelf()

            if current_shelf is not None:
                if current_shelf[2] > self.agent_level:
                    return Action.TOGGLE_LOAD

                else:
                    with_higher_agent_level = self.exists_shelf_with_higher_agent_level()

                    if len(with_higher_agent_level) == 0:
                        if current_shelf[2] <= self.agent_level:
                            return Action.TOGGLE_LOAD

        next_pos = self.action_to_closest_box_from_higher_level()

        selected_action = Action.NOOP

        if next_pos == (self.currentPos[0], self.currentPos[1] - 1):  # up
            return Action.UP
        elif next_pos == (self.currentPos[0], self.currentPos[1] + 1):  # down
            return Action.DOWN
        elif next_pos == (self.currentPos[0] - 1, self.currentPos[1]):  # left
            return Action.LEFT
        elif next_pos == (self.currentPos[0] + 1, self.currentPos[1]):
            return Action.RIGHT

        return selected_action

    def obtain_shelves(self, obsFromOneAgent):
        agentShelves = obsFromOneAgent['sensors']
        for shelf in agentShelves:
            if shelf != {}:
                if not self.check_if_shelf_already_in_shelves((shelf['location'][0], shelf['location'][1])):
                    self.shelves.append((shelf['location'][0], shelf['location'][1], shelf['shelf_level'][0]))

    def other_agent_is_neighboor(self, obsFromOneAgent):
        otherAgentPosition = (
            obsFromOneAgent['otherAgent']['location'][0], obsFromOneAgent['otherAgent']['location'][1])

    def action_to_closest_box_from_higher_level(self):
        from_higher_level = 0
        for shelf in self.shelves:
            if shelf[2] > self.agent_level:
                from_higher_level += 1

        distances = []
        for shelf in self.shelves:
            if from_higher_level > 0:
                if shelf[2] > self.agent_level:
                    distance = self.hamming_distance((self.currentPos[0], self.currentPos[1]),
                                                     (shelf[0], shelf[1]))
                    distances.append(distance)
                else:
                    distances.append(9999999)
            else:
                if shelf[2] <= self.agent_level:
                    distance = self.hamming_distance((self.currentPos[0], self.currentPos[1]),
                                                     (shelf[0], shelf[1]))
                    distances.append(distance)
                else:
                    distances.append(9999999)

        if len(distances) == 0:
            if len(self.shelves) > 0:

                minCoiso = random.randint(0, len(self.shelves) - 1)
                minimum_distance_box = (self.shelves[minCoiso][0], self.shelves[minCoiso][1])

                path = pyastar2d.astar_path(self.mapState, tuple(self.currentPos), minimum_distance_box,
                                            allow_diagonal=False)
                if len(path) >= 2:
                    next_pos = tuple(path[1])
                else:
                    return self.currentPos

                return next_pos
            else:
                return Action.NOOP
        else:
            minCoiso = distances.index(min(distances))
            minimum_distance_box = (self.shelves[minCoiso][0], self.shelves[minCoiso][1])

            path = pyastar2d.astar_path(self.mapState, tuple(self.currentPos), minimum_distance_box,
                                        allow_diagonal=False)
            if len(path) >= 2:
                next_pos = tuple(path[1])
            else:
                return self.currentPos

            return next_pos

    def make_map_from_agents(self, obsFromOneAgent):
        self.agent_level = obsFromOneAgent["self"]["agent_level"]

        otherAgentPosition = (
            obsFromOneAgent['otherAgent']['location'][0], obsFromOneAgent['otherAgent']['location'][1])

        myPosition = (obsFromOneAgent["self"]["location"][0], obsFromOneAgent["self"]["location"][1])
        self.currentPos = myPosition



        self.obtain_shelves(obsFromOneAgent)
        self.mapState = np.ones((self.width, self.height), dtype=np.float32) * 10

        self.mapState[otherAgentPosition[1], otherAgentPosition[0]] = 20
        self.mapState[myPosition[1], myPosition[0]] = 2

        for shelf in self.shelves:
            self.mapState[shelf[1]][shelf[0]] = 1
