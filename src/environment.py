from random import randint
from enum import Enum
import numpy as np
from math import atan2, pi, cos, sin
from itertools import product
from abc import ABC, abstractmethod

PI = round(pi,2)
PI_2 = round(pi * 0.5, 2)
TWO_PI = round(2 * pi, 2)

class StateSimple:
	def __init__(self):
		self._azimuth = 0.0

	def __call__(self, obs):
		return self._process(obs)

	def _process(self, obs):
		gx_ag, gy_ag = obs[3] - obs[0], obs[4] - obs[1]
		cos_rot, sin_rot = round(cos(obs[2]),2), round(sin(obs[2]),2)
		self._azimuth = round(atan2(-sin_rot * gx_ag + cos_rot * gy_ag,
			cos_rot * gx_ag + sin_rot * gy_ag), 2)
		self._azimuth = PI if self._azimuth == -PI else self._azimuth
		return self._azimuth

class StateComplex(StateSimple):
	def __init__(self):
		super().__init__()
		self.__los = []
	def _process(self, obs):
		return (super()._process(obs), obs[5])

class Actions(Enum):
	ROT_LEFT = -1
	FWD = 0
	ROT_RIGHT = 1

class DiscreteSpace:
	def __init__(self):
		self._space = []
	def __call__(self):
		return self._space
	def __repr__(self):
		return str(self._space)
	def __len__(self):
		return len(self._space)
	@property
	def space(self):
		return self._space
	@space.setter
	def space(self, space):
		self._space = space
	def concat(self, other):
		space_aux = []
		for elem1 in self._space:
			for elem2 in other.space:
				space_aux.append((elem1, elem2))
		self._space = space_aux

class DiscreteAngularSpace(DiscreteSpace):
	def __init__(self, width, height):
		super().__init__()
		self.__width, self.__height = width, height
		self.__generate_space()
	def __generate_space(self):
		for y in range(self.__height):
			for x in range(self.__width):
				self._space.append(round(atan2(y,x), 2))
				self._space.append(round(atan2(-y,x), 2))
				self._space.append(round(atan2(-y,-x), 2))
				self._space.append(round(atan2(y,-x), 2))
		self._space = list(dict.fromkeys(self._space))
		self._space.sort()

class DiscreteLineOfSightSpace(DiscreteSpace):
	def __init__(self, range, shape = 'T'):
		super().__init__()
		self.__range, self.__shape = range, shape
		self.__generate_space()
	def __generate_space(self):
		los_entities = list(filter(lambda ent: (ent != Entities.AGENT.value and ent != Entities.VOID.value), 
			list(map(lambda x: (x.value), Entities))))
		if self.__shape == '+':
			self._space = [los for los in product([ent for ent in los_entities], repeat = 4 * self.__range)]
		elif self.__shape == 'T':
			self._space = [los for los in product([ent for ent in los_entities], repeat = 3 * self.__range)]
		else: # shape == '*'
			self._space = [los for los in product([ent for ent in los_entities], repeat = 6 * self.__range)]

class Entities(Enum):
	VOID = -2
	OBSTACLE = -1
	EMPTY = 0
	AGENT = 1
	GOAL = 2

class EnvEntity:
	def __init__(self):
		self._pos = [0,0]

	def __repr__(self):
		return 'Entity with position: ' + str(self._pos) + '\n'

	def set_x(self, x):
		self._pos[0] = x

	def set_y(self, y):
		self._pos[1] = y

	def go_left(self):
		self._pos[0] -= 1

	def go_up(self):
		self._pos[1] += 1

	def go_right(self):
		self._pos[0] += 1

	def go_down(self):
		self._pos[1] -= 1

	@property
	def pos(self):
		return self._pos

	@property
	def x(self):
		return self._pos[0]

	@property
	def y(self):
		return self._pos[1]
		
	def move(self, x, y):
		self._pos[0], self._pos[1] = x, y

class EnvAgent(EnvEntity):
	def __init__(self):
		super().__init__()
		self.__theta = 0

	@staticmethod
	def normalize_angle(ang):
		ang = ((ang % TWO_PI) + TWO_PI) % TWO_PI
		if ang > PI:
			ang -= TWO_PI
		return ang

	def kinematics(self, action):
		if action == Actions.FWD:
			x, y = self._pos[0] + cos(self.__theta), self._pos[1] + sin(self.__theta)
			if x < 0 or y < 0:
				return self._pos[0], self._pos[1], self.__theta
			return int(round(self._pos[0] + cos(self.__theta))), int(round(self._pos[1] + sin(self.__theta))), self.__theta
		elif action == Actions.ROT_LEFT:
			return int(self._pos[0]), int(self._pos[1]), round(self.normalize_angle(self.__theta - PI_2), 2)
		elif action == Actions.ROT_RIGHT:
			return int(self._pos[0]), int(self._pos[1]), round(self.normalize_angle(self.__theta + PI_2), 2)
		else:
			raise Exception('Not a valid action')

	def move(self, x, y, ang):
		self._pos[0], self._pos[1], self.__theta = x, y, ang
		
	@property
	def theta(self):
		return self.__theta
	
class RewardFunction:
	GOAL_PRIZE = 1
	COLLISION_PENALTY = -1
	UNDEFINED = 0

	def __init__(self, state_space):
		self.__state_space = state_space.space
		self.__max_angle_variation = round(abs(atan2(1, -1)),2)
		self.__den = 2 * self.__max_angle_variation
		self.__inv_den = 1.0 / self.__den

	def __call__(self, cur_state, action, next_state, event = None):
		#if cur_state is None or next_state is None:
		#	return self.UNDEFINED
		#if event is Entities.VOID.value:
		#	return self.UNDEFINED
		print('cur:',cur_state, 'next:',next_state)

		if next_state == 0.0 or event == Entities.GOAL.value:
			print('reward: GOAL')
			return self.GOAL_PRIZE

		elif event is None:
			print('reward: normal NAO ESQUECER DE MUDAR!')
			return 1
			return self.normalize(-self.__max_angle_variation, 
				self.__max_angle_variation, 
				abs(cur_state) - abs(next_state)
			)

		elif event == Entities.OBSTACLE.value:
			print('reward: COLLIDED')
			return self.COLLISION_PENALTY

		else:
			return self.UNDEFINED

	# normalize between [-1, 1]
	def normalize(self, min, max, val):
		return 2 * (val - min)  * self.__inv_den - 1

class Environment(ABC):
	def __init__(self, w, h):
		self._w, self._h = w, h
		self._build_arena()

		# Entities
		self._agent, self._goal = EnvAgent(), EnvEntity()
		self._entities = {
			Entities.AGENT: self._agent,
			Entities.GOAL: self._goal,
		}
		# State space
		self._state_space = DiscreteAngularSpace(self._w, self._h)

		# Reward function
		self.__reward_function = RewardFunction(self._state_space)

	def __repr__(self):
		return str(self._arena)

	@property
	def w(self):
		return self._w

	@property
	def h(self):
		return self._h

	@property
	def agent(self):
		return self._agent

	@property
	def goal(self):
		return self._goal

	@property
	def state_space(self):
		return self._state_space

	@abstractmethod
	def reset(self, event = None):
		pass
		
	def _build_arena(self):
		self._arena = np.zeros((self._w, self._h), dtype = np.int8)

	def observe(self):
		return self.get_agent_state(), False, RewardFunction.UNDEFINED, Entities.VOID.value

	def _place_entity_random(self, ent):
		while True:
			r, c = randint(0, self._w - 1), randint(0, self._h - 1)
			if self._arena[r,c] == Entities.EMPTY.value:
				self._arena[r,c] = ent.value
				if ent == Entities.AGENT:
					self._agent.move(c, r, 0.0)
					break
				elif ent == Entities.GOAL:
					self._goal.move(c, r)
					break

	# When invalid action, skip to next iteration
	def skip(self, last_state):
		return last_state, False, RewardFunction.UNDEFINED, Entities.VOID.value

	# A physics iteration in the environment
	@abstractmethod
	def step(self, action):
		pass

	@abstractmethod
	def get_agent_state(self):
		pass

class EmptyEnvironment(Environment):
	def __init__(self, w, h):
		self.__agent_state = StateSimple()
		super().__init__(w,h)
		self.__reward_function = RewardFunction(self._state_space)
		self.reset()

	def reset(self, event = None):
		self._arena[1:self._h, 1:self._w] = Entities.EMPTY.value
		self._place_entity_random(Entities.AGENT)
		self._place_entity_random(Entities.GOAL)
		return self.get_agent_state(), False, RewardFunction.UNDEFINED, Entities.VOID.value

	# When invalid action, skip to next iteration
	def skip(self, last_state):
		return last_state, False, RewardFunction.UNDEFINED, Entities.VOID.value

	def step(self, action):
		# Clean old agent info
		self._arena[self._agent.y,self._agent.x] = Entities.EMPTY.value
		
		# Save last state
		last_state = self.get_agent_state()
		
		# Predict next pose
		x, y, ang = self._agent.kinematics(action)

		# Choose action
		if action == Actions.FWD:
			# If next pose is out of bounds, return and skip to next decision
			if x < 0 or x == self._w or y < 0 or y == self._h:
				return self.skip(last_state)

			# Get neighbour of next pose
			neighbour = self._arena[y,x]

			# Move to predicted pose
			self._agent.move(x, y, ang)

			# get_agent_state next state
			next_state = self.get_agent_state()

			# If neighbour is the goal, finish episode
			if neighbour == Entities.GOAL.value:
				return next_state, True, self.__reward_function(last_state, action, next_state, neighbour), neighbour
		# action == ROT_LEFT or ROT_RIGHT
		else:
			# If next pose is out of bounds, neighbour is None
			if x < 0 or x == self._w or y < 0 or y == self._h:
				neighbour = None
			else:
				neighbour = self._arena[y,x]

			# Move to predicted pose
			self._agent.move(x, y, ang)

			# get_agent_state next state
			next_state = self.get_agent_state()

		self._arena[self._agent.y, self._agent.x] = Entities.AGENT.value

		# get_agent_state next state
		next_state = self.get_agent_state()
		return next_state, False, self.__reward_function(last_state, action, next_state, None), neighbour

	def get_agent_state(self):
		return self.__agent_state(
			(	
				self._agent.x, 
				self._agent.y, 
				self._agent.theta,
				self._goal.x, 
				self._goal.y
			)
		)

class ObstacleEnvironment(Environment):
	def __init__(self, w, h, num_obstacles):
		self.__agent_state = StateComplex()
		if num_obstacles >= w * h:
			raise ValueError('Number of obstacles needs to be lower than width * height')
		if num_obstacles > 0:
			self.__num_obstacles = num_obstacles
		else:
			raise ValueError('Invalid number of obstacles')
		self.__obstacles = []

		super().__init__(w,h)

		# Add obstacles
		for i in range(self.__num_obstacles):
			self.__obstacles.append(EnvEntity())
		self._entities[Entities.OBSTACLE] = self.__obstacles

		# State space
		self.__los_state_space = DiscreteLineOfSightSpace(1,'T')
		self._state_space.concat(self.__los_state_space)
		print('State space len:', len(self._state_space))

		self.__reward_function = RewardFunction(self._state_space)
		self.reset()

	def reset(self, event = None):
		self._arena[1:self._h, 1:self._w] = [Entities.EMPTY.value]
		self._place_entity_random(Entities.AGENT)
		self._place_entity_random(Entities.GOAL)
		self.__generate_obstacles(self.__num_obstacles)
		return self.get_agent_state(), False, RewardFunction.UNDEFINED, Entities.VOID.value

	def _build_arena(self):
		self._arena = np.zeros((self._w + 2, self._h + 2), dtype = np.int8)
		self._arena[:,0] = Entities.OBSTACLE.value
		self._arena[0,1:] = Entities.OBSTACLE.value
		self._arena[1:,-1] = Entities.OBSTACLE.value
		self._arena[-1,1:self._w+1] = Entities.OBSTACLE.value

	def __generate_obstacles(self, num):
		for i in range(num):
			while True: 
				r, c = randint(1, self._w), randint(1, self._h)
				if self._arena[r,c] == Entities.EMPTY.value:
					self._arena[r,c] = Entities.OBSTACLE.value
					self._entities[Entities.OBSTACLE][i].move(c, r)
					break

	def step(self, action):
		# Clean old agent info
		self._arena[self._agent.y,self._agent.x] = Entities.EMPTY.value
		
		# Save last state
		last_state = self.get_agent_state()
		
		# Predict next pose
		x, y, ang = self._agent.kinematics(action)

		if action == Actions.FWD:
			# If next pose is out of bounds, return and skip to next decision
			if x < 0 or x == self._w or y < 0 or y == self._h:
				return self.skip(last_state)

			# Get neighbour of next pose
			neighbour = self._arena[y,x]

			# Move to predicted pose
			self._agent.move(x, y, ang)

			# get_agent_state next state
			next_state = self.get_agent_state()

			# If neighbour is the goal, finish episode
			if neighbour == Entities.GOAL.value:
				return next_state, True, self.__reward_function(last_state, action, next_state, neighbour), neighbour
		# action == ROT_LEFT or ROT_RIGHT
		else:
			# If next pose is out of bounds, neighbour is None
			if x < 0 or x == self._w or y < 0 or y == self._h:
				neighbour = None
			else:
				neighbour = self._arena[y,x]

			# Move to predicted pose
			self._agent.move(x, y, ang)

			# get_agent_state next state
			next_state = self.get_agent_state()

		self._arena[self._agent.y, self._agent.x] = Entities.AGENT.value

		# get_agent_state next state
		next_state = self.get_agent_state()
		return next_state, False, self.__reward_function(last_state, action, next_state, None), neighbour

	def __generate_los(self):
		# [left, up, right, down]
		# TODO: adapt to los dependent on the orientation
		return (
			self._arena[self._agent.y][self._agent.x - 1],
			self._arena[self._agent.y - 1][self._agent.x],
			self._arena[self._agent.y][self._agent.x + 1],
		)
	
	def get_agent_state(self):
		return self.__agent_state((
			self._agent.x,
			self._agent.y,
			self._agent.theta,
			self._goal.x,
			self._goal.y,
			self.__generate_los()
			)
		)
