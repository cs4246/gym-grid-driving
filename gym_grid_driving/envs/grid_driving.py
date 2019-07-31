import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces

import numpy as np
from collections import namedtuple
from enum import Enum

import logging
logger = logging.getLogger(__name__)


random = None


AgentState = Enum('AgentState', 'alive crashed finished out')

LaneSpec = namedtuple('LaneSpec', ['cars', 'speed_range'])
GridDrivingState = namedtuple('GridDrivingState', ['cars', 'agent', 'finish_position', 'occupancy_trails'])
MaskSpec = namedtuple('MaskSpec', ['type', 'radius'])


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

    def sample_point(self):
        return Point(random.randint(self.x, self.x + self.w), random.randint(self.y, self.y + self.h))

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
        self.bound = self.world.boundary.circular_bound if self.world and self.world.boundary and circular else lambda x, **kwargs: x
        self.auto_brake = auto_brake
        self.auto_lane = auto_lane
        self.p = p
        self.done()
        self.ignored = False

    def sample_speed(self):
        if random.random_sample() > self.p:
            return np.round(np.average(self.speed_range))
        return random.randint(*tuple(np.array(self.speed_range)+np.array([0, 1])))

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


class Action(object):
    def __init__(self, name, delta):
        self.name = name
        self.delta = delta

    def __str__(self):
        return "{}".format(self.name)

    def __repr__(self):
        return self.__str__()


class ActionableCar(Car):
    def start(self, **kwargs):
        action = kwargs.get('action')
        self._start(delta=action.delta)
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


class Mask(object):
    def __init__(self, type, radius, boundary=None):
        assert type in ['follow', 'random']
        self.type = type
        self.radius = radius
        if self.type == 'random' and not boundary:
            raise Exception('Boundary must be defined for type: random')
        self.boundary = boundary

    def step(self, target=None):
        self.target = target
        if self.type == 'follow' and not self.target:
            raise Exception('Target must be defined for type: follow')
        if self.type == 'random':
            self.target = self.boundary.sample_point()
        return self

    def get(self):
        return Rectangle(self.radius * 2, self.radius * 2, self.target.x - self.radius, self.target.y - self.radius)
        

class World(object):
    def __init__(self, boundary, finish_position=None, flicker_rate=0.0, mask=None):
        self.boundary = boundary
        self.finish_position = finish_position
        self.flicker_rate = flicker_rate
        self.mask = Mask(mask.type, mask.radius, self.boundary) if mask else None

    def init(self, cars, agent=None):
        self.cars = cars
        self.agent = agent
        self.max_dist_travel = np.max([np.max(np.absolute(car.speed_range)) for car in cars])
        self.lanes = [OrderedLane(self) for i in range(self.boundary.h)]
        for car in cars:
            self.lanes[car.position.y].append(car)
        self.occupancy_trails = np.zeros((self.boundary.w, self.boundary.h))
        self.blackout = False
        if self.mask:
            self.mask.step(self.agent.position)
        self.update_state()

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
        exception = None
        try:
            for car in self.cars:
                if car == self.agent:
                    car.start(action=action)
                    self.lanes[car.lane].recognize()
                else:
                    car.start()

            self.occupancy_trails = np.zeros((self.boundary.w, self.boundary.h))
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

                # Handle car jump pass through other car
                if self.agent and occupancies[self.agent.position.x, self.agent.position.y] > 0:
                    raise AgentCrashedException

                # Handle car jump pass through finish
                if self.agent and self.finish_position and self.agent.position == self.finish_position:
                    raise AgentFinishedException

                self.occupancy_trails = np.clip(self.occupancy_trails + occupancies, 0, 1)

            if self.agent and self.occupancy_trails[self.agent.position.x, self.agent.position.y] > 0:
                    raise AgentCrashedException
            if self.agent and not self.boundary.contains(self.agent.position):
                raise AgentOutOfBoundaryException

            for car in self.cars:
                car.done()

            self.blackout = random.random_sample() <= self.flicker_rate

            if self.mask:
                self.mask.step(self.agent.position)

        except Exception as e:
            self.blackout = False
            self.mask = None
            exception = e
        finally:
            self.update_state()
            if exception:
                raise exception

    def as_tensor(self, pytorch=True):
        t = np.zeros(self.tensor_shape)
        for car in self.state.cars:
            if self.state.agent and car != self.state.agent:
                t[0, car.position.x, car.position.y] = 1
        if self.state.agent:
            t[1, self.state.agent.position.x, self.state.agent.position.y] = 1
        if self.state.finish_position:
            t[2, self.state.finish_position.x, self.state.finish_position.y] = 1
        t[3, :, :] = self.state.occupancy_trails
        if pytorch:
            t = np.transpose(t, (0, 2, 1)) # [C, H, W]
        assert t.shape == self.space(pytorch).shape
        return t

    def space(self, pytorch=True, channel=True):
        c, w, h = self.tensor_shape
        tensor_shape = (c, h, w) if pytorch else self.tensor_shape
        return spaces.Box(low=0, high=1, shape=tensor_shape[int(not channel):], dtype=np.uint8)

    def update_state(self):
        agent = self.agent
        other_cars = set(self.cars) - set([self.agent])
        finish_position = self.finish_position
        occupancy_trails = self.occupancy_trails

        if self.mask:
            mask = self.mask.get()
            agent = agent if mask.contains(agent.position) else None
            other_cars = set([car for car in list(other_cars) if mask.contains(car.position)])
            finish_position = finish_position if mask.contains(finish_position) else None
            occupancy_trails_mask = np.full(self.occupancy_trails.shape, False)
            occupancy_trails_mask[mask.x:mask.x+mask.w, mask.y:mask.y+mask.h] = True
            occupancy_trails[~occupancy_trails_mask] = 0.0

        if self.blackout:
            agent = None
            other_cars = set([])
            finish_position = None
            occupancy_trails = np.zeros(self.occupancy_trails.shape)

        self._state = GridDrivingState(other_cars, agent, finish_position, occupancy_trails)

    @property
    def tensor_state(self):
        return self.as_tensor()

    @property
    def state(self):
        return self._state

    @property
    def tensor_shape(self):
        return (4, self.boundary.w, self.boundary.h)



class GridDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (0, Constant.FINISH_REWARD)

    def __init__(self, **kwargs):
        self.random_seed = kwargs.get('random_seed', None)

        self.lanes = kwargs.get('lanes', DefaultConfig.LANES)
        self.width = kwargs.get('width', DefaultConfig.WIDTH)

        self.agent_speed_range = kwargs.get('agent_speed_range', DefaultConfig.PLAYER_SPEED_RANGE)
        self.finish_position = kwargs.get('finish_position', Point(0, 0))
        self.agent_pos_init = kwargs.get('agent_pos_init', Point(self.width-1, len(self.lanes)-1))

        self.p = kwargs.get('stochasticity', DefaultConfig.STOCHASTICITY)
        self.tensor_state = kwargs.get('tensor_state', False)

        self.flicker_rate = kwargs.get('flicker_rate', 0.0)

        self.mask_spec = kwargs.get('mask', None)

        self.boundary = Rectangle(self.width, len(self.lanes))
        self.world = World(self.boundary, finish_position=self.finish_position, flicker_rate=self.flicker_rate, mask=self.mask_spec)

        agent_direction = np.sign(self.agent_speed_range[0])
        self.actions = [Action('up', Point(agent_direction,-1)), Action('down', Point(agent_direction,1))]
        self.actions += [Action('forward[{}]'.format(i), Point(i, 0)) 
                            for i in range(self.agent_speed_range[0], self.agent_speed_range[1]+1)]

        self.action_space = spaces.Discrete(len(self.actions))
        if self.tensor_state:
            self.observation_space = self.world.space()
        else:
            n_cars = sum([l.cars for l in self.lanes])
            self.observation_space = spaces.Dict({
                'cars': spaces.Tuple(tuple([self.world.space(channel=False) for i in range(n_cars)])),
                'agent_pos': self.world.space(channel=False), 
                'finish_pos': self.world.space(channel=False), 
                'occupancy_trails': spaces.MultiBinary(self.world.space(channel=False).shape)
            })

        self.reset()

    def seed(self, seed=None):
        global random
        random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if isinstance(action, int):
            assert action in range(len(self.actions))
            action = self.actions[action]

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
            self.state = self.world.tensor_state
        else:
            self.state = self.world.state

        return self.state, reward, self.done, {}

    def reset(self):
        self.seed(self.random_seed)
        self.agent_state = AgentState.alive
        self.agent = ActionableCar(self.agent_pos_init, self.agent_speed_range, self.world, circular=False, auto_brake=False, auto_lane=False, p=self.p, id='<')
        self.cars = [self.agent]
        i = 0
        for y, lane in enumerate(self.lanes):
            choices = list(range(0, self.agent.position.x)) + list(range(self.agent.position.x+1, self.width)) if self.agent.lane == y else range(self.width)
            xs = random.choice(choices, lane.cars, replace=False)
            for x in xs:
                self.cars.append(Car(Point(x,y), lane.speed_range, self.world, p=self.p, id=i))
                i += 1
        self.world.init(self.cars, agent=self.agent)

        if self.tensor_state:
            self.state = self.world.tensor_state
        else:
            self.state = self.world.state

        return self.state

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError
        cars = self.world.as_tensor(pytorch=False)[0, :, :]
        view = np.chararray(cars.shape, unicode=True, itemsize=3)
        view[self.world.state.occupancy_trails.nonzero()] = '~'
        for car in self.world.state.cars:
            if car != self.world.state.agent:
                view[car.position.tuple] = car.id or 'O'
        if self.world.state.finish_position and self.boundary.contains(self.world.state.finish_position):
            view[self.world.state.finish_position.tuple] += 'F'
        if self.world.state.agent and self.boundary.contains(self.world.state.agent.position):
            if self.agent_state == AgentState.crashed:
                view[self.world.state.agent.position.tuple] += '#'
            else:
                view[self.world.state.agent.position.tuple] += '<'
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
    