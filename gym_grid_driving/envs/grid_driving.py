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
    def __init__(self, position, speed_range, world, circular=True, auto_brake=True, auto_lane=True, p=1.0, id=None):
        self.id = id
        self.position = position
        self.speed_range = speed_range
        self.world = world
        self.bound = self.world.boundary.circular_bound if circular else lambda x, **kwargs: x
        self.auto_brake = auto_brake
        self.auto_lane = auto_lane
        self.p = p
        self.done()
        self.ignored = False

    def sample_speed(self):
        if np.random.random() > self.p:
            return np.round(np.average(self.speed_range))
        return np.random.randint(*tuple(np.array(self.speed_range)+np.array([0, 1])))

    def _start(self, **kwargs):
        delta = kwargs.get('delta', Point(0, 0))
        self.destination = self.world.boundary.bound(self.bound(self.position + delta, bound_y=False), bound_x=False)
        self.changed_lane = self.destination.y != self.position.y

    def start(self, **kwargs):
        self._start(delta=Point(self.sample_speed(), 0))

    def done(self):
        self.destination = None

    def _step(self, delta):
        if not self.destination:
            raise CarNotStartedException
        if not self.need_step():
            return
        if self.auto_brake and not self.can_step():
            return

        target = self.world.boundary.bound(self.bound(self.position + delta, bound_y=False), bound_x=False)

        if self.auto_lane and target.y != self.lane and not self.can_change_lane(target):
            return

        self.position = target

    def step(self, **kwargs):
        self._step(Point(self.direction if self.destination.x != self.position.x else 0, np.sign(self.destination.y - self.position.y)))

    def need_step(self):
        return self.position != self.destination

    def can_step(self):
        return self.world.lanes[self.lane].gap(self)

    def can_change_lane(self, target):
        return len(filter(lambda c: c.position == target, self.world.lanes[target.y])) == 0

    @property
    def direction(self):
        return np.sign(self.speed_range[0])

    @property
    def lane(self):
        return self.position.y
    
    def __repr__(self):
        return "Car({}, {}, {})".format(self.id, self.position, self.speed_range)


class ActionableCar(Car):
    def start(self, **kwargs):
        action = kwargs.get('action', Action.stay)
        if action == Action.up:
            dy = -1
        elif action == Action.down:
            dy = 1
        elif action == Action.stay:
            dy = 0
        else:
            raise ActionNotFoundException
        self._start(delta=Point(self.sample_speed(), dy))

        self.ignored = self.changed_lane


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
        self.recognize()

    def recognize(self):
        self.cars_recognized = [car for car in self.cars if not car.ignored]

    def front(self, car):
        if car not in self.cars_recognized: # car is ignored
            return self.cars[self.cars.index(car) - 1]
        return self.cars_recognized[self.cars_recognized.index(car) - 1]

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
        self.total_occupancies = np.zeros((self.boundary.w, self.boundary.h))

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
        for car in self.cars:
            if car == self.agent:
                car.start(action=action)
                self.lanes[car.lane].recognize()
            else:
                car.start()

        self.total_occupancies = np.zeros((self.boundary.w, self.boundary.h))
        for i in range(self.max_dist_travel):
            occupancies = np.zeros((self.boundary.w, self.boundary.h))
            for lane in self.lanes:
                for car in lane.ordered_cars:
                    last_y = car.position.y

                    car.step()

                    if car != self.agent:
                        occupancies[car.position.x, car.position.y] = 1
                        
                    if last_y != car.position.y:
                        self.reassign_lanes()

            self.total_occupancies += occupancies

        if self.agent and self.total_occupancies[self.agent.position.x, self.agent.position.y] > 0:
                raise AgentCrashedException
        if self.agent and not self.boundary.contains(self.agent.position):
            raise AgentOutOfBoundaryException
        if self.agent and self.finish_position and self.agent.position == self.finish_position:
            raise AgentFinishedException

        for car in self.cars:
            car.done()

    @property
    def tensor(self):
        t = np.zeros((4, self.boundary.w, self.boundary.h))
        for car in self.cars:
            if self.agent and car != self.agent:
                t[0, car.position.x, car.position.y] = 1
        if self.agent:
            t[1, self.agent.position.x, self.agent.position.y] = 1
        if self.finish_position:
            t[2, self.finish_position.x, self.finish_position.y] = 1
        t[3, :, :] = self.total_occupancies
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
        self.agent = ActionableCar(self.agent_pos_init, self.agent_speed_range, self.world, circular=False, auto_brake=False, auto_lane=False, p=self.p, id='<')
        self.cars = [self.agent]
        i = 0
        for y, lane in enumerate(self.lanes):
            choices = list(range(0, self.agent.position.x)) + list(range(self.agent.position.x+1, self.width)) if self.agent.lane == y else range(self.width)
            xs = np.random.choice(choices, lane.cars, replace=False)
            for x in xs:
                self.cars.append(ActionableCar(Point(x,y), lane.speed_range, self.world, p=self.p, id=i))
                i += 1
        self.world.init(self.cars, agent=self.agent)

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError
        cars = self.world.tensor[0, :, :]
        view = np.chararray(cars.shape, unicode=True, itemsize=2)
        view[self.world.total_occupancies.nonzero()] = '~'
        for car in self.cars:
            if car != self.agent:
                view[car.position.tuple] = car.id or 'O'
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
    