import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from collections import namedtuple
from enum import Enum

import logging
logger = logging.getLogger(__name__)


Action = Enum('Action', 'stay up down')
ACTIONS = [Action.stay, Action.up, Action.down]

PlayerState = Enum('PlayerState', 'alive crashed finished out')

Lane = namedtuple('Lane', ['cars', 'speed_range'])
GridDrivingState = namedtuple('GridDrivingState', ['cars', 'player_pos', 'finish_pos'])



DEFAULT_LANES = [
	Lane(1, [-2, -1]),
	Lane(2, [-3, -3]),
	Lane(3, [-3, -1]),
]
DEFAULT_WIDTH = 10
DEFAULT_PLAYER_SPEED_RANGE = [-1, -1]


class ActionNotFoundException(Exception): 
	pass


def sample_speed(speed_range):
	return np.random.randint(*tuple(np.array(speed_range)+np.array([0, 1])))


def within_grid(grid, pos):
	return pos[0] >= 0 and pos[0] < grid.shape[0] and pos[1] >= 0 and pos[1] < grid.shape[1]


class GridDrivingEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, **kwargs):
		self.lanes = kwargs.get('lanes', DEFAULT_LANES)
		self.width = kwargs.get('width', DEFAULT_WIDTH)

		self.player_speed_range = kwargs.get('player_speed_range', DEFAULT_PLAYER_SPEED_RANGE)
		self.finish_pos = kwargs.get('finish_pos', (0, 0))
		self.player_pos_init = kwargs.get('player_pos_init', (len(self.lanes)-1, self.width-1))

		self.reset()

	def step(self, action):
		reward = 0

		if self.done:
			logger.warn("You are calling 'step()' even though this environment has already returned done = True. \
				You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			return self.state, reward, self.done, {}

		# Prepare occupancy grid
		self.car_occupancies = np.zeros_like(self.cars)

		# Move cars on each lane
		for l, lane in enumerate(self.lanes):
			speed = sample_speed(lane.speed_range)
			self.car_occupancies[l, :] = np.sum([np.roll(self.cars[l, :], np.sign(speed)*s) for s in range(1, np.abs(speed)+1)], axis=0)
			self.cars[l, :] = np.roll(self.cars[l, :], speed)

		# Move player based on the action
		speed = sample_speed(self.player_speed_range)
		x = self.player_pos[1] + speed
		if action == Action.up:
			l = np.maximum(self.player_pos[0] - 1, 0)
		elif action == Action.down:
			l = np.minimum(self.player_pos[0] + 1, len(self.lanes)-1)
		elif action == Action.stay:
			l = self.player_pos[0]
		else:
			raise ActionNotFoundException
		self.player_pos = (l, x)

		# Compute reward and done
		if within_grid(self.cars, self.player_pos):
			if self.player_pos == self.finish_pos:
				reward = 10
				self.player_state = PlayerState.finished
			if self.car_occupancies[self.player_pos] > 0:
				self.player_state = PlayerState.crashed
		else:
			self.player_state = PlayerState.out

		# Update state
		self.state = GridDrivingState(self.cars, self.player_pos, self.finish_pos)

		return self.state, reward, self.done, {'car_occupancies': self.car_occupancies}

	def reset(self):
		self.player_state = PlayerState.alive
		self.player_pos = self.player_pos_init
		self.cars = np.zeros((len(self.lanes), self.width), dtype=int)
		for l, lane in enumerate(self.lanes):
			car_positions = np.random.choice(range(self.width), lane.cars, replace=False)
			self.cars[l, car_positions] = 1
		self.car_occupancies = np.zeros_like(self.cars)

	def render(self, mode='human'):
		if mode != 'human':
			raise NotImplementedError
		view = np.chararray(self.cars.shape, unicode=True, itemsize=2)
		view[self.car_occupancies.nonzero()] = '~'
		view[self.cars.nonzero()] = 'O'
		view[self.finish_pos] += 'F'
		if within_grid(self.cars, self.player_pos):
			if self.player_state == PlayerState.crashed:
				view[self.player_pos] += '#'
			else:
				view[self.player_pos] += '<'
		view[np.where(view == '')] = '-'
		print(''.join('====' for i in view[0]))
		for row in view:
			print(' '.join('%03s' % i for i in row))
		print(''.join('====' for i in view[0]))

	def close(self):
		pass

	@property
	def done(self):
		return self.player_state != PlayerState.alive
	