from random import randint
from enum import Enum
import numpy as np
from math import atan2, pi, cos, sin
from itertools import product
from abc import ABC, abstractmethod

PI = round(pi,2)
PI_2 = round(pi * 0.5, 2)
PI_4 = round(pi * 0.25, 2)
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
		self.__vectors = []
		self.__generate_space()
	def __generate_space(self):
		los_entities = list(filter(lambda ent: (ent != Entities.AGENT.value and ent != Entities.VOID.value), 
			list(map(lambda x: (x.value), Entities))))
		if self.__shape == '+':
			self._space = [los for los in product([ent for ent in los_entities], repeat = 4 * self.__range)]
		elif self.__shape == 'T':
			self._space = [los for los in product([ent for ent in los_entities], repeat = 3 * self.__range)]
			self.__vectors = np.array([[1,1,1],[1,0,-1]])
		elif self.__shape == '.':
			self._space = [los for los in product([ent for ent in los_entities], repeat = 1 * self.__range)]
			self.__vectors = np.array([[1],[0]])
		else: # shape == '*'
			self._space = [los for los in product([ent for ent in los_entities], repeat = 6 * self.__range)]
	@property
	def vectors(self):
		return self.__vectors
	
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

	def __init__(self, environment):
		self.__environment = environment
		self.__max_angle_variation = round(abs(atan2(1, -1)),2)
		self.__den = 2 * self.__max_angle_variation
		self.__inv_den = 1.0 / self.__den
		self.__k = 0.7
		if isinstance(self.__environment, ObstacleEnvironment):
			self.reward = self.collision_avoidance
		else:
			self.reward = self.collision_free
	def __call__(self, cur_state, action, next_state, event = None):
		print('[RewardFunction] processing...')
		if any(cur_state) and any(next_state):
			print('[RewardFunction] returning real value')
			return self.reward(cur_state, action, next_state, event)
		else:
			print('[RewardFunction] returning UNDEFINED')
			return self.UNDEFINED

	def collision_free(self, cur_state, action, next_state, event = None):
		if (next_state == 0.0 or event == Entities.GOAL.value):
			print('reward: GOAL')
			return self.GOAL_PRIZE

		elif event is None:
			print('reward: goal reaching')
			return self.goal_reaching(cur_state, next_state)

		elif event == Entities.OBSTACLE.value:
			print('reward: COLLIDED')
			return self.COLLISION_PENALTY

		else:
			print('reward: undefined')
			return self.UNDEFINED

	def collision_avoidance(self, cur_state, action, next_state, event = None):
		# 1. Goal reached
		if next_state[0] == 0 or event == Entities.GOAL.value and Entities.OBSTACLE.value != next_state[1]:
			print('reward: GOAL')
			return self.GOAL_PRIZE
		# 2. Obstacle collision
		elif event == Entities.OBSTACLE.value:
			print('reward: COLLIDED')
			return self.COLLISION_PENALTY
		# 3. If normal navigation
		elif event is None:
			if Entities.OBSTACLE.value != next_state[1] or Entities.OBSTACLE.value != cur_state[1]:
				print('reward: obstacle avoidance')
				return self.__k * self.obstacle_avoidance(cur_state[1], next_state[1]) \
				+ (1 - self.__k) * self.goal_reaching(cur_state[0], next_state[0])
			else:
				print('reward: obstacle avoidance')
				return self.goal_reaching(cur_state[0], next_state[0])
		else:
			print('reward: undefined')
			return self.UNDEFINED

	def goal_reaching(self, cur_state, next_state):
		return self.normalize(-self.__max_angle_variation, 
				self.__max_angle_variation, 
				abs(cur_state) - abs(next_state)
				)
	def obstacle_avoidance(self, cur_state, next_state):
		return next_state - cur_state

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
		self._reward_function = RewardFunction(self)

		# Auxiliar variables
		self._cur_state = []
		self._next_state = []

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

	@property
	def entities(self):
		return self._entities

	@abstractmethod
	def reset(self, event = None):
		pass
		
	def _build_arena(self):
		self._arena = np.zeros((self._w, self._h), dtype = np.int8)

	def observe(self):
		print('[observe]')
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
		self.reset()

	def reset(self, event = None):
		self._build_arena()
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
				return next_state, True, self._reward_function(last_state, action, next_state, neighbour), neighbour
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
			#next_state = self.get_agent_state()

		self._arena[self._agent.y, self._agent.x] = Entities.AGENT.value

		# get_agent_state next state
		next_state = self.get_agent_state()
		return next_state, False, self._reward_function(last_state, action, next_state, None), neighbour

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
		self.__los_state_space = DiscreteLineOfSightSpace(1,'.')
		self._state_space.concat(self.__los_state_space)
		print('State space len:', len(self._state_space))

		# Auxiliar variables
		self.__translation_walls = np.array([1,1]).reshape(2,1)

		# Reset
		self.reset()


	def reset(self, terminal = False, cur_state = None, next_state = None, action = None, event = None):
		print('>>>Reseting')
		self._arena[1:self._h + 1, 1:self._w + 1] = [Entities.EMPTY.value]
		self._place_entity_random(Entities.AGENT)
		self._place_entity_random(Entities.GOAL)
		self.__generate_obstacles(self.__num_obstacles)
		return self.get_agent_state(terminal), terminal, self._reward_function(self._cur_state, action, self._next_state, event), event # Change this to reward function

	def _build_arena(self):
		self._arena = np.zeros((self._w + 2, self._h + 2), dtype = np.int8)
		self._arena[:,0] = Entities.OBSTACLE.value
		self._arena[0,1:] = Entities.OBSTACLE.value
		self._arena[1:,-1] = Entities.OBSTACLE.value
		self._arena[-1,1:self._w+1] = Entities.OBSTACLE.value

	def _place_entity_random(self, ent):
		while True:
			r, c = randint(1, self._w), randint(1, self._h)
			if self._arena[r,c] == Entities.EMPTY.value:
				self._arena[r,c] = ent.value
				if ent == Entities.AGENT:
					self._agent.move(c - 1, r - 1, 0.0)
					break
				elif ent == Entities.GOAL:
					self._goal.move(c - 1, r - 1)
					break

	def __generate_obstacles(self, num):
		for i in range(num):
			while True: 
				r, c = randint(1, self._w), randint(1, self._h)
				if self._arena[r,c] == Entities.EMPTY.value:
					self._arena[r,c] = Entities.OBSTACLE.value
					self._entities[Entities.OBSTACLE][i].move(c - 1, r - 1)
					break

	def step(self, action): #TODO: adapt this step to obstacle
		# Clean old agent info
		self._arena[self._agent.y + 1,self._agent.x + 1] = Entities.EMPTY.value
		print('[step] -> get_agent_state()')
		# Save last state
		self._cur_state = self.get_agent_state()
		
		# Predict next pose
		x, y, ang = self._agent.kinematics(action)

		# Check entity for next position
		neighbour = self._arena[y + 1, x + 1]

		# Act
		if action == Actions.FWD:
			print('[Actions.FWD]')
			print('[Actions.FWD], neighbour:', neighbour)
			
			# If collision, quit
			if neighbour == Entities.OBSTACLE.value:
				print('[Actions.FWD] Collision! Reseting')
				return self.reset(True, self._cur_state, self._cur_state, action, neighbour)
			# If got goal, quit
			if neighbour == Entities.GOAL.value:
				print('[Action.FWD] Got goal! Reseting')
				return self.reset(True, self._cur_state, self._cur_state, action, neighbour)

			# Move
			self._agent.move(x, y, ang)

			# Observe next state
			self._next_state = self.get_agent_state()
			print('[Actions.FWD]: next_state: ',  self._next_state)

		# action == ROT_LEFT or ROT_RIGHT
		else:
			print('[Actions.ROT]:', action)
			# If got goal, quit
			if neighbour == Entities.GOAL.value:
				print('[Actions.ROT] Got goal! Reseting')
				return self.reset(True, self._cur_state, self._cur_state, action, neighbour)
			# Move
			self._agent.move(x, y, ang)

			# Observe next state
			self._next_state = self.get_agent_state()
			print('[Actions.ROT]: next_state: ',  self._next_state)

		# Place agent in arena in new position
		self._arena[self._agent.y + 1, self._agent.x + 1] = Entities.AGENT.value
		print('after step:\n', self._arena)

		return self._next_state, False, self._reward_function(self._cur_state, action, self._next_state, None), neighbour

	def __generate_los(self, terminal = False):
		print('[__generate_los]:', terminal)
		print('[__generate_los], cur_arena: \n', self._arena)
		if terminal:
			print('Generating terminal los')
			return self._cur_state[1]
		# [left, front, right]
		rot = np.array([
			[cos(self._agent.theta), sin(self._agent.theta)],
			[sin(self._agent.theta), -cos(self._agent.theta)]
		])
		translation_agent = np.array([self._agent.x, self._agent.y]).reshape(2,1)
		los_positions = np.rint(np.dot(rot, self.__los_state_space.vectors) + translation_agent + self.__translation_walls).astype(np.int8)
		print('x,y,ang: ', self._agent.x, self._agent.y, self._agent.theta)
		print('los_positions:\n', los_positions)
		#print((
		#	self._arena[los_positions[1,0]][los_positions[0,0]],
		#	self._arena[los_positions[1,1]][los_positions[0,1]],
		#	self._arena[los_positions[1,2]][los_positions[0,2]]
		#))
		return self._arena[los_positions[1,0]][los_positions[0,0]]
		#return (
		#	self._arena[los_positions[1,0]][los_positions[0,0]],
		#	self._arena[los_positions[1,1]][los_positions[0,1]],
		#	self._arena[los_positions[1,2]][los_positions[0,2]]
		#)
	
	def get_agent_state(self, terminal = False):
		return self.__agent_state((
			self._agent.x,
			self._agent.y,
			self._agent.theta,
			self._goal.x,
			self._goal.y,
			self.__generate_los(terminal)
			)
		)
