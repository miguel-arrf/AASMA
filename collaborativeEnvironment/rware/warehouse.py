import logging
import random

from collections import defaultdict, OrderedDict, Counter
import gym
from gym import spaces

from enum import Enum
import numpy as np

from typing import List, Tuple, Optional, Dict

import networkx as nx
import astar

_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2

_COLLISION_LAYERS = 2

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1


class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx: self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits


class Action(Enum):
    NOOP = 0
    UP = 1
    LEFT = 2
    RIGHT = 3
    DOWN = 5
    TOGGLE_LOAD = 4


class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObserationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2


class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """
    SHELVES = 0  # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1  # binary layer indicating requested shelves
    AGENTS = 2  # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3  # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4  # binary layer indicating agents with load
    GOALS = 5  # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6  # binary layer indicating accessible cells (all but occupied cells/ out of map)


class Entity:
    def __init__(self, id_: int, x: int, y: int, level: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y
        self.level = level


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, msg_bits: int, level: int):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y, level)
        self.message = np.zeros(msg_bits)
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS, _LAYER_SHELFS)
        else:
            return (_LAYER_AGENTS,)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action == Action.TOGGLE_LOAD:
            return self.x, self.y
        if self.req_action == Action.UP:
            return self.x, max(0, self.y - 1)
        if self.req_action == Action.NOOP:
            return self.x, self.y
        elif self.req_action == Action.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.req_action == Action.LEFT:
            return max(0, self.x - 1), self.y
        elif self.req_action == Action.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y


class Shelf(Entity):
    counter = 0

    def __init__(self, x, y, level):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y, level)

    @property
    def collision_layers(self):
        return (_LAYER_SHELFS,)


class Warehouse(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
            self,
            shelf_columns: int,
            column_height: int,
            shelf_rows: int,
            n_agents: int,
            msg_bits: int,
            sensor_range: int,
            request_queue_size: int,
            max_inactivity_steps: Optional[int],
            max_steps: Optional[int],
            reward_type: RewardType,
            layout: str = None,
            observation_type: ObserationType = ObserationType.DICT,
            image_observation_layers: List[ImageLayer] = [
                ImageLayer.SHELVES,
                ImageLayer.REQUESTS,
                ImageLayer.AGENTS,
                ImageLayer.GOALS,
                ImageLayer.ACCESSIBLE
            ],
            image_observation_directional: bool = True,
            normalised_coordinates: bool = False,
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param n_agents: Number of spawned and controlled agents
        :type n_agents: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)

        self.n_agents = n_agents
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.normalised_coordinates = normalised_coordinates

        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = spaces.MultiDiscrete(sa_action_space)
        self.action_space = spaces.Tuple(tuple(n_agents * [sa_action_space]))
        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents: List[Agent] = []

        # default values:
        self.fast_obs = None
        self.image_obs = None
        self.observation_space = None
        if observation_type == ObserationType.IMAGE:
            self._use_image_obs(image_observation_layers, image_observation_directional)
        else:
            # used for DICT observation type and needed as preceeding stype to generate
            # FLATTENED observations as well
            self._use_slow_obs()

        # for performance reasons we
        # can flatten the obs vector
        if observation_type == ObserationType.FLATTENED:
            self._use_fast_obs()

        self.renderer = None

    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)

        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        highway_func = lambda x, y: (
                (x % 3 == 0)  # vertical highways
                or (y % (self.column_height + 1) == 0)  # horizontal highways
                or (y == self.grid_size[0] - 1)  # delivery row
                or (  # remove a box for queuing
                        (y > self.grid_size[0] - (self.column_height + 3))
                        and ((x == self.grid_size[1] // 2 - 1) or (x == self.grid_size[1] // 2))
                )
        )
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)

    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.int32)
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1

    def _use_slow_obs(self):
        self.fast_obs = False

        self._obs_sensor_locations = 4
        if self.normalised_coordinates:
            location_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            )
        else:
            location_space = spaces.Discrete(2)

        self.observation_space = spaces.Tuple(
            tuple(
                [
                    spaces.Dict(
                        OrderedDict(
                            {
                                "self": spaces.Dict(
                                    OrderedDict(
                                        {
                                            "location": location_space,
                                            "agent_level": spaces.Discrete(1),
                                        }
                                    )
                                ),
                                "otherAgent": spaces.Dict(
                                    OrderedDict(
                                        {
                                            "location": location_space,
                                            "agent_level": spaces.Discrete(1),
                                        }
                                    )
                                )
                                ,
                                "sensors": spaces.Tuple(
                                    self._obs_sensor_locations
                                    * (
                                        spaces.Dict(
                                            OrderedDict(
                                                {
                                                    "location": location_space,
                                                    "shelf_level": spaces.MultiBinary(
                                                        1
                                                    ),

                                                }
                                            )
                                        ),
                                    )
                                ),
                            }
                        )
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

        coiso = spaces.Tuple(
            self._obs_sensor_locations
            * (
                spaces.Dict(
                    OrderedDict(
                        {
                            "has_agent": spaces.MultiBinary(1),
                            "direction": spaces.Discrete(4),
                            "shelf_requested": spaces.MultiBinary(
                                1
                            ),
                            "location": location_space,
                        }
                    )
                ),
            )
        )

    def _use_fast_obs(self):
        if self.fast_obs:
            return

        self.fast_obs = True
        ma_spaces = []
        for sa_obs in self.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def _make_obs(self, agent):

        min_x = 0
        max_x = 8

        min_y = 0
        max_y = 8

        # sensors
        if (
                (min_x < 0)
                or (min_y < 0)
                or (max_x > self.grid_size[1])
                or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]

        agents = padded_agents[0:11, 0:10].reshape(-1)

        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)

        if self.fast_obs:
            # write flattened observations
            obs = _VectorWriter(self.observation_space[agent.id - 1].shape[0])

            if self.normalised_coordinates:
                agent_x = agent.x / (self.grid_size[1] - 1)
                agent_y = agent.y / (self.grid_size[0] - 1)
            else:
                agent_x = agent.x
                agent_y = agent.y

            obs.write([agent_x, agent_y, agent.level])

            # MEU
            for agent_new in self.agents:
                if agent_new != agent:
                    obs.write([agent_new.x, agent_new.y])
                    obs.write([agent_new.level])

            # END MEU

            for shelf in self.shelfs:
                obs.write([shelf.x, shelf.y])
                obs.write([shelf.level])

            return obs.vector

        # write dictionary observations
        obs = {}
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y
        # --- self data


        newAgent = None
        for agent_new in self.agents:
            if agent_new != agent:
                newAgent = agent_new

        obs["self"] = {
            "location": np.array([agent_x, agent_y]),
            "agent_level": agent.level,
        }
        obs["otherAgent"] = {
            "location": np.array([newAgent.x, newAgent.y]),
             "agent_level": newAgent.level,
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring shelfs:
        for i in range(len(self.shelfs)):
            obs["sensors"][i]["location"] = [self.shelfs[i].x, self.shelfs[i].y]
            obs["sensors"][i]["shelf_level"] = [self.shelfs[i].level]


        return obs

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

        for a in self.agents:
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    def spawn_players(self, max_player_level, shelves_positions, players_number):
        positions = []
        for player in range(players_number):
            attempts = 0

            while attempts < 1000:
                row = random.randint(0, 7)
                col = random.randint(0, 7)
                if (row, col) not in shelves_positions:
                    positions.append((row, col, random.randint(1, max_player_level)))
                    break
                attempts += 1
        return positions

    def reset(self):
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9

        # make the shelfs

        self.shelfs = []
        max_player_level = 3

        self.agents = []
        shelfsToPut = []

        novoVetor = self.spawn_players(3, shelfsToPut, 2)
        agentPositions = []
        for value in novoVetor:
            agentPositions.append((value[0], value[1]))
            self.agents.append(Agent(x=value[0], y=value[1], msg_bits=self.msg_bits, level=value[2]))
        player_levels = sorted([player.level for player in self.agents])
        max_level = sum(player_levels)

        someBiggerThanSum = False

        while len(shelfsToPut) != 4:
            x = random.randint(1, 6)
            y = random.randint(1, 6)

            if (x, y) not in agentPositions:
                if (x, y) not in shelfsToPut:
                    if len(shelfsToPut) == 3:
                        if not someBiggerThanSum:
                            self.shelfs.append(Shelf(x, y, max_level))
                        else:
                            level = random.randint(1, max_level)
                            self.shelfs.append(Shelf(x, y, level))
                        shelfsToPut.append((x, y))
                    else:
                        shelfsToPut.append((x, y))
                        # self.shelfs.append(Shelf(x, y, levels[len(shelfsToPut)]))
                        level = random.randint(1, max_level)
                        self.shelfs.append(Shelf(x, y, level))
                        if level == max_level:
                            someBiggerThanSum = True

        self._recalc_grid()

        # self.request_queue = list(
        #    np.random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        # )

        return tuple([self._make_obs(agent) for agent in self.agents])
        # for s in self.shelfs:
        #     self.grid[0, s.y, s.x] = 1
        # print(self.grid[0])

    def get_agents_withoutGivenAgent(self, agentIDontWant):
        toReturn = []
        for agent in self.agents:
            if agent != agentIDontWant:
                toReturn.append(agent)
        return toReturn

    def getAdjacentPlayers(self, agent):
        adjacentPlayers = []
        for otherAgent in self.agents:
            if otherAgent != agent:
                isLeft = otherAgent.x == agent.x - 1 and otherAgent.y == agent.y
                isRight = otherAgent.x == agent.x + 1 and otherAgent.y == agent.y
                isTop = otherAgent.x == agent.x and otherAgent.y == agent.y - 1
                isBottom = otherAgent.x == agent.x and otherAgent.y == agent.y + 1
                if isLeft or isRight or isTop or isBottom:
                    adjacentPlayers.append(otherAgent)

        return adjacentPlayers

    def step(
            self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        assert len(actions) == len(self.agents)

        whoCatched = []
        whoCatched_Together = []

        rewards = np.zeros(self.n_agents)

        for agent, action in zip(self.agents, actions):
            if self.msg_bits > 0:
                agent.req_action = Action(action[0])
                agent.message[:] = action[1:]
            else:
                agent.req_action = Action(action)

        positions = []

        for agent in self.agents:
            target = agent.req_location(self.grid_size)
            positions.append(target)

        res = [ele for ele, count in Counter(positions).items()
               if count > 1]

        for repeatedElement in res:
            for agent in self.agents:
                target = agent.req_location(self.grid_size)
                if target == repeatedElement:
                    # this guy can't move!
                    #TODO: THIS IS QUITE IMPORTANT!
                    if agent.req_action != Action.TOGGLE_LOAD:
                        agent.req_action = Action.NOOP

                        # TODO: COLIDIRAM!
                        rewards -= 0.1

        '''# Caso em que querem 'trocar' um com o outro!
        for firstAgent in self.agents:
            for secondAgent in self.agents:
                if firstAgent != secondAgent:
                    firstAgentTarget = firstAgent.req_location(self.grid_size)
                    firstAgentInitial = (firstAgent.x, firstAgent.y)

                    secondAgentTarget = secondAgent.req_location(self.grid_size)
                    secondAgentInitial = (secondAgent.x, secondAgent.y)

                    if firstAgentTarget == secondAgentInitial and secondAgentTarget == firstAgentInitial:
                        firstAgent.req_action = Action.NOOP
                        secondAgent.req_action = Action.NOOP
                        # TODO: COLIDIRAM
                        rewards[firstAgent.id - 1] -= 0.1
                        rewards[secondAgent.id - 1] -= 0.1
        '''

        for agent in self.agents:
            agent.prev_x, agent.prev_y = agent.x, agent.y

            if agent.req_action in [Action.UP, Action.RIGHT, Action.LEFT, Action.DOWN]:
                agent.x, agent.y = agent.req_location(self.grid_size)
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]

                if shelf_id:
                    toRemove = None
                    for shelf in self.shelfs:
                        if shelf.id == shelf_id:
                            toRemove = shelf
                    shelf_level = toRemove.level
                    agent_level = agent.level

                    if shelf_level <= agent_level:
                        # print("POSSO COMER : X: {}, Y: {}".format(toRemove.x, toRemove.y))
                        rewards[agent.id - 1] += shelf_level
                        self.shelfs.remove(toRemove)

                        whoCatched.append(agent.id)

                    if shelf_level > agent_level:
                        adjacentPlayers = self.getAdjacentPlayers(agent)

                        if len(adjacentPlayers) != 0:
                            adjacentPlayersSum = 0
                            for adjacentPlayer in adjacentPlayers:
                                adjacentPlayersSum += adjacentPlayer.level
                            adjacentPlayersSum += agent_level

                            if adjacentPlayersSum >= shelf_level:
                                whoCatched_Together.append(1)
                                self.shelfs.remove(toRemove)

                                for toRewardAgent in adjacentPlayers:
                                    rewards[
                                        toRewardAgent.id - 1] += toRewardAgent.level / adjacentPlayersSum * shelf_level

                                rewards[agent.id - 1] += agent_level / adjacentPlayersSum * shelf_level

        self._recalc_grid()

        dones = []
        withEarlyStop = True

        if withEarlyStop:
            for agentN in self.agents:
                agentNLevel = agentN.level

                feito = False
                if len(self.shelfs) == 0:
                    feito = True

                shelfsLevelN = []
                for shelf in self.shelfs:
                    if shelf.level <= agentNLevel:
                        shelfsLevelN.append(shelf)
                if len(shelfsLevelN) == 0:
                    feito = True
                dones.append(feito)
        else:
            if self._cur_steps > 500:
                dones = [True, True]
            else:
                dones = [False, False]

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        info = {}
        return new_obs, list(rewards - 0.01), dones, {'alone': whoCatched, 'together': whoCatched_Together}

    def render(self, mode="human"):
        if not self.renderer:
            from rware.rendering import Viewer

            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        ...

    def optimal_returns(self, steps=None, output=False):
        """
        Compute optimal returns for environment for all agents given steps
        NOTE: Needs to be called on reset environment with shelves in their initial locations

        :param steps (int): number of steps available to agents
        :param output (bool): whether steps should be printed
        :return (List[int]): returns for all agents

        This function initially positions agents randomly in the warehouse and assumes
        full observability with agents directly moving to closest possible shelf to deliver
        or closest "open space" to return. Required steps for movement (including rotations)
        are computed using A* only moving on highways if shelves are loaded. This serves as a
        crude approximation. Observability with agents directly moving towards requested shelves/
        goals without search significantly simplifies the problem.
        """
        # if already computed --> return computed value
        if hasattr(self, 'calculated_optimal_returns'):
            return self.calculated_optimal_returns

        if steps is None:
            steps = self.max_steps

        def neighbore_locations(state):
            # given location get neighbours
            x, y, direction, loaded, empty_shelf_loc = state
            # neighbours for rotating
            neighbours = [
                (x, y, (direction - 1) % 4, loaded, empty_shelf_loc),
                (x, y, (direction + 1) % 4, loaded, empty_shelf_loc)
            ]
            # neighbour for forward movement
            if direction == 0:
                # going down
                target_x = x
                target_y = y + 1
            elif direction == 1:
                # going left
                target_x = x - 1
                target_y = y
            elif direction == 2:
                # going up
                target_x = x
                target_y = y - 1
            elif direction == 3:
                # going right
                target_x = x + 1
                target_y = y
            else:
                raise ValueError(f"Invalid direction {direction} for optimal return computation!")

            if target_x >= 0 and target_x < self.grid_size[1] and target_y >= 0 and target_y < self.grid_size[0]:
                # valid location
                if not loaded or (self._is_highway(target_x, target_y) or (target_x, target_y) == empty_shelf_loc):
                    neighbours.append((target_x, target_y, direction, loaded, empty_shelf_loc))
            # else:
            #     print(f"({target_x}, {target_y}) out of bounds")
            # print(state, neighbours)
            return neighbours

        def hamming_distance(state1, state2):
            x1, y1, _, _, _ = state1
            x2, y2, _, _, _ = state2
            return abs(x1 - x2) + abs(y1 - y2)

        def is_goal(state, goal):
            x, y, _, _, _ = state
            goal_x, goal_y, _, _, _ = goal
            return x == goal_x and y == goal_y

        def pathfinder(state1, state2):
            # pathfinder between two warehouse locations
            # print()
            # print("\tFind path:", state1, state2)
            return list(astar.find_path(
                state1,
                state2,
                neighbore_locations,
                reversePath=False,
                heuristic_cost_estimate_fnct=hamming_distance,
                distance_between_fnct=lambda a, b: 1.0,
                is_goal_reached_fnct=is_goal,
            ))

        # count delivered shelves
        agent_deliveries = [0] * self.n_agents
        agent_directions = list(np.random.randint(0, 4, self.n_agents))
        agent_locations = [(np.random.choice(self.grid_size[1]), np.random.choice(self.grid_size[0])) for _ in
                           range(self.n_agents)]
        # agent goal location with remaining distances to goal
        agent_goals = [loc for loc in agent_locations]
        agent_goal_distances = [0] * self.n_agents
        # original locations of collected shelves
        agent_shelf_original_locations = [None] * self.n_agents
        # agent status (0 - go to requested shelf, 1 - go to goal, 2 - bring back shelf)
        agent_status = [2] * self.n_agents

        # print(self.grid_size)
        # print(self.goals)

        for t in range(0, steps):
            if output:
                print()
                print(f"STEP {t}")
            for i in range(self.n_agents):
                agent_direction = agent_directions[i]
                goal = agent_goals[i]
                goal_distance = agent_goal_distances[i]
                agent_stat = agent_status[i]
                agent_shelf_orig_location = agent_shelf_original_locations[i]
                if output:
                    print(f"\tAgent {i}: {agent_locations[i]} --> {goal} ({goal_distance}) with stat={agent_stat}")
                if goal_distance == 0:
                    # reached goal
                    if agent_stat == 0:
                        # goal is to collect shelf --> now will be loaded
                        # new goal: go to goal location
                        agent_locations[i] = goal
                        agent_shelf_original_locations[i] = goal
                        # find closest goal
                        state = (goal[0], goal[1], agent_direction, True, goal)
                        closest_goal = None
                        closest_goal_distance = None
                        closest_goal_direction = None
                        for possible_goal in self.goals:
                            goal_state = (possible_goal[0], possible_goal[1], None, True, goal)
                            path = pathfinder(state, goal_state)
                            distance = len(path)
                            direction = path[-1][2]
                            if closest_goal_distance is None or distance < closest_goal_distance:
                                closest_goal = possible_goal
                                closest_goal_distance = distance
                                closest_goal_direction = direction
                        agent_goals[i] = closest_goal
                        agent_goal_distances[i] = closest_goal_distance
                        agent_directions[i] = closest_goal_direction
                        agent_status[i] = 1
                    elif agent_stat == 1:
                        # goal is to deliver shelf at goal --> now delivered
                        # new goal: bring back shelf
                        agent_deliveries[i] += 1
                        # for new goal: return to original location
                        assert agent_shelf_orig_location is not None
                        agent_locations[i] = goal
                        agent_goals[i] = agent_shelf_orig_location
                        state = (goal[0], goal[1], agent_direction, True, agent_shelf_orig_location)
                        goal_state = (agent_goals[i][0], agent_goals[i][1], None, True, agent_shelf_orig_location)
                        path = pathfinder(state, goal_state)
                        agent_goal_distances[i] = len(path)
                        agent_directions[i] = path[-1][2]
                        agent_shelf_original_locations[i] = None
                        agent_status[i] = 2
                    elif agent_stat == 2:
                        # goal is to bring back shelf --> now succeeded
                        # new goal: identify new random unrequested shelf to collect
                        # find unrequested shelf
                        shelf = np.random.choice(self.shelfs)
                        agent_locations[i] = goal
                        agent_goals[i] = (shelf.x, shelf.y)
                        agent_shelf_original_locations[i] = None
                        state = (goal[0], goal[1], agent_direction, False, (-1, -1))
                        goal_state = (agent_goals[i][0], agent_goals[i][1], None, False, (-1, -1))
                        path = pathfinder(state, goal_state)
                        agent_goal_distances[i] = len(path)
                        agent_status[i] = 0
                        agent_directions[i] = path[-1][2]
                else:
                    # not yet reached goal --> get one closer to goal
                    agent_goal_distances[i] -= 1

        if self.reward_type == RewardType.GLOBAL:
            total_returns = sum(agent_deliveries)
            self.calculated_optimal_returns = [total_returns] * self.n_agents
        else:
            self.calculated_optimal_returns = agent_deliveries
        return self.calculated_optimal_returns


if __name__ == "__main__":
    env = Warehouse(9, 8, 3, 10, 3, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    import time
    from tqdm import tqdm

    time.sleep(2)
    # env.render()
    # env.step(18 * [Action.LOAD] + 2 * [Action.NOOP])

    for _ in tqdm(range(1000000)):
        # time.sleep(2)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
