import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from collections import namedtuple
from enum import Enum

import logging
logger = logging.getLogger(__name__)


Action = Enum('Action', 'stay up down')
AgentState = Enum('AgentState', 'alive crashed finished out')

LaneSpec = namedtuple('LaneSpec', ['cars', 'speed_range'])
GridDrivingState = namedtuple('GridDrivingState', ['cars', 'agent_pos', 'finish_pos'])


class Constant:
    FINISH_REWARD = 10


class DefaultConfig:
    LANES = [
        LaneSpec(1, [-2, -1]),
        LaneSpec(2, [-3, -3]),
        LaneSpec(3, [-3, -1]),
    ]
    WIDTH = 10
    PLAYER_SPEED_RANGE = [-1, -1]
    STOCHASTICITY = 1.0


class ActionNotFoundException(Exception): 
    pass

class AgentCrashedException(Exception):
    pass

class AgentOutOfBoundaryException(Exception):
    pass

class AgentFinishedException(Exception):
    pass

class CarNotStartedException(Exception):
    pass
        


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y) 

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __mul__(self, other):
        return Point(self.x * other, self.y * other) 

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return "Point(x={},y={})".format(self.x, self.y)

    @property
    def tuple(self):
        return (self.x, self.y)


class Rectangle(object):
    def __init__(self, w, h, x=0, y=0):
        self.w, self.h = w, h
        self.x, self.y = x, y

    def bound(self, point, bound_x=True, bound_y=True):
        x = np.minimum(np.maximum(point.x, self.x), self.x + self.w - 1) if bound_x else point.x
        y = np.minimum(np.maximum(point.y, self.y), self.y + self.h - 1) if bound_y else point.y
        return Point(x, y)

    def circular_bound(self, point, bound_x=True, bound_y=True):
        x = self.x + ((point.x - self.x) % self.w) if bound_x else point.x
        y = self.y + ((point.y - self.y) % self.h) if bound_y else point.y
        return Point(x, y)

    def contains(self, point):
        return (point.x >= self.x and point.x < self.x + self.w) and (point.y >= self.y and point.y < self.y + self.h)

    def __str__(self):
        return "Rectangle(w={},h={},x={},y={})".format(self.w, self.h, self.x, self.y)


class Car(object):
    def __init__(self, position, speed_range, world, circular=True, auto_brake=True, p=1.0):
        self.position = position
        self.speed_range = speed_range
        self.world = world
        self.bound = self.world.boundary.circular_bound if circular else lambda x, **kwargs: x
        self.auto_brake = auto_brake
        self.p = p
        self.done()

    def sample_speed(self):
        if np.random.random() > self.p:
            return np.round(np.average(self.speed_range))
        return np.random.randint(*tuple(np.array(self.speed_range)+np.array([0, 1])))

    def start(self):
        self.destination = self.bound(self.position + Point(self.sample_speed(), 0), bound_y=False)

    def done(self):
        self.destination = None

    def _step(self):
        self.position = self.bound(self.position + Point(self.direction, 0), bound_y=False)

    def step(self):
        if not self.destination:
            raise CarNotStartedException
        if not self.need_step():
            return
        if self.auto_brake and not self.can_step():
            return
        self._step()

    def need_step(self):
        return self.position != self.destination

    def can_step(self):
        return self.world.lanes[self.position.y].gap(self)

    @property
    def direction(self):
        return np.sign(self.speed_range[0])

    def __repr__(self):
        return "Car({}, {})".format(self.position, self.speed_range)


class ActionableCar(Car):
    def act(self, action):
        if action == Action.up:
            dy = Point(0, -1)
        elif action == Action.down:
            dy = Point(0, 1)
        elif action == Action.stay:
            dy = Point(0, 0)
        else:
            raise ActionNotFoundException
        self.position = self.world.boundary.bound(self.position + dy, bound_x=False)


class OrderedLane(object):
    def __init__(self, world, cars=None):
        self.cars = cars or []
        self.world = world
        self.sort()

    def append(self, car):
        self.cars.append(car)
        self.sort()

    def remove(self, car):
        self.cars.remove(car)
        self.sort()

    def sort(self):
        self.cars = sorted(self.cars, key=lambda car: car.position.x, reverse=self.reverse)

    def front(self, car):
        return self.cars[self.cars.index(car) - 1]

    def gap(self, car):
        pos = [car.position.x, self.front(car).position.x][::-1 if self.reverse else 1]
        return (pos[0] - pos[1] - 1) % self.world.boundary.w

    @property
    def reverse(self):
        return False if len(self.cars) == 0 else np.sign(self.cars[0].speed_range[0]) > 0

    @property
    def ordered_cars(self):
        def first_gap_index(cars):
            for i, car in enumerate(cars):
                if self.gap(car) > 1:
                    return i
        index = first_gap_index(self.cars)
        return self.cars[index:]+self.cars[:index]

    def __repr__(self):
        view = np.chararray(self.world.boundary.w, unicode=True, itemsize=2)
        view[[car.position.x for car in filter(lambda c: isinstance(c, Car), self.cars)]] = 'O'
        view[[car.position.x for car in filter(lambda c: isinstance(c, ActionableCar), self.cars)]] = 'A'
        view[np.where(view == '')] = '-'
        return ' '.join('%03s' % i for i in view)

        

class World(object):
    def __init__(self, boundary, finish_position=None):
        self.boundary = boundary
        self.finish_position = finish_position

    def init(self, cars, agent=None):
        self.cars = cars
        self.agent = agent
        self.max_dist_travel = np.max([np.max(np.absolute(car.speed_range)) for car in cars])
        self.lanes = [OrderedLane(self) for i in range(self.boundary.h)]
        for car in cars:
            self.lanes[car.position.y].append(car)

    def reassign_lanes(self):
        unassigned_cars = []
        for y, lane in enumerate(self.lanes):
            for car in lane.cars:
                if car.position.y != y:
                    lane.remove(car)
                    unassigned_cars.append(car)
        for car in unassigned_cars:
            self.lanes[car.position.y].append(car)

    def step(self, action=None):
        self.agent.act(action)

        self.reassign_lanes()

        for car in self.cars:
            car.start()

        for i in range(self.max_dist_travel):
            occupancies = np.zeros((self.boundary.w, self.boundary.h))
            for lane in self.lanes:
                for car in lane.ordered_cars:
                    car.step()
                    if car != self.agent:
                        occupancies[car.position.x, car.position.y] = 1

            if self.agent and occupancies[self.agent.position.x, self.agent.position.y] > 0:
                raise AgentCrashedException
            if self.agent and not self.boundary.contains(self.agent.position):
                raise AgentOutOfBoundaryException
            if self.agent and self.finish_position and self.agent.position == self.finish_position:
                raise AgentFinishedException

        for car in self.cars:
            car.done()

    @property
    def tensor(self):
        t = np.zeros((3, self.boundary.w, self.boundary.h))
        for car in self.cars:
            if self.agent and car != self.agent:
                t[0, car.position.x, car.position.y] = 1
        if self.agent:
            t[1, self.agent.position.x, self.agent.position.y] = 1
        if self.finish_position:
            t[2, self.finish_position.x, self.finish_position.y] = 1
        return t



class GridDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self.lanes = kwargs.get('lanes', DefaultConfig.LANES)
        self.width = kwargs.get('width', DefaultConfig.WIDTH)

        self.agent_speed_range = kwargs.get('agent_speed_range', DefaultConfig.PLAYER_SPEED_RANGE)
        self.finish_position = kwargs.get('finish_position', Point(0, 0))
        self.agent_pos_init = kwargs.get('agent_pos_init', Point(self.width-1, len(self.lanes)-1))

        self.p = kwargs.get('stochasticity', DefaultConfig.STOCHASTICITY)
        self.tensor_state = kwargs.get('tensor_state', False)

        self.boundary = Rectangle(self.width, len(self.lanes))
        self.world = World(self.boundary, finish_position=self.finish_position)

        self.reset()

    def step(self, action):
        reward = 0

        if self.done:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. \
                You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            return self.state, reward, self.done, {}

        try:
            self.world.step(action)
        except AgentCrashedException:
            self.agent_state = AgentState.crashed
        except AgentOutOfBoundaryException:
            self.agent_state = AgentState.out
        except AgentFinishedException:
            self.agent_state = AgentState.finished
            reward = Constant.FINISH_REWARD

        if self.tensor_state:
            self.state = self.world.tensor
        else:
            self.state = GridDrivingState(set(self.cars) - set([self.agent]), self.agent, self.finish_position)

        return self.state, reward, self.done, {}

    def reset(self):
        self.agent_state = AgentState.alive
        self.agent = ActionableCar(self.agent_pos_init, self.agent_speed_range, self.world, circular=False, auto_brake=False, p=self.p)
        self.cars = [self.agent]
        for y, lane in enumerate(self.lanes):
            xs = np.random.choice(range(self.width), lane.cars, replace=False)
            for x in xs:
                self.cars.append(Car(Point(x,y), lane.speed_range, self.world, p=self.p))
        self.world.init(self.cars, agent=self.agent)

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError
        cars = self.world.tensor[0, :, :]
        view = np.chararray(cars.shape, unicode=True, itemsize=2)
        view[cars.nonzero()] = 'O'
        view[self.finish_position.tuple] += 'F'
        if self.boundary.contains(self.agent.position):
            if self.agent_state == AgentState.crashed:
                view[self.agent.position.tuple] += '#'
            else:
                view[self.agent.position.tuple] += '<'
        view[np.where(view == '')] = '-'
        view = np.transpose(view)
        print(''.join('====' for i in view[0]))
        for row in view:
            print(' '.join('%03s' % i for i in row))
        print(''.join('====' for i in view[0]))

    def close(self):
        pass

    @property
    def done(self):
        return self.agent_state != AgentState.alive
    