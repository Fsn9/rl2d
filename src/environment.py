from random import randint
from enum import Enum
import numpy as np
from math import atan2, pi, cos, sin, isclose
from itertools import product
from abc import ABC, abstractmethod

PI = round(pi,2)
PI_2 = round(pi * 0.5, 2)
PI_4 = round(pi * 0.25, 2)
TWO_PI = round(2 * pi, 2)
FLOAT_TOLERANCE = 0.1

class StateSimple:
	def __init__(self, azimuth = 0.0):
		self._azimuth = azimuth

	def __call__(self, obs):
		return self._process(obs)

	def __eq__(self, other):
		return self._azimuth == other.azimuth
		#return isclose(self._azimuth, other.azimuth, abs_tol = FLOAT_TOLERANCE)

	def __repr__(self):
		return 'Ss (' + str(self._azimuth) + ')'

	@property
	def azimuth(self):
		return self._azimuth

	def get_data_from(self, other):
		self._azimuth = other.azimuth

	def _process(self, obs):
		gx_ag, gy_ag = obs[3] - obs[0], obs[4] - obs[1]
		cos_rot, sin_rot = round(cos(obs[2]),2), round(sin(obs[2]), 2)
		#self._azimuth = round(atan2(-sin_rot * gx_ag + cos_rot * gy_ag,
		#	cos_rot * gx_ag + sin_rot * gy_ag), 2)
		self._azimuth = round(atan2(sin_rot * gx_ag - cos_rot * gy_ag,
			cos_rot * gx_ag + sin_rot * gy_ag), 2)
		# because the state space does not have -PI and PI is the same as -PI
		self._azimuth = PI if isclose(self._azimuth, -PI, abs_tol = FLOAT_TOLERANCE) else self._azimuth 
		return self

class StateComplex(StateSimple):
	def __init__(self, azimuth = 0.0, los = ()):
		super().__init__(azimuth)
		self.__los = los

	def __repr__(self):
		return "Sc (" + str(self._azimuth) + "," + str(self.__los) + ")"

	def __eq__(self, other):
		return isclose(self._azimuth, other.azimuth, abs_tol = FLOAT_TOLERANCE) and self.__los == other.los

	@property
	def los(self):
		return self.__los

	def get_data_from(self):
		self._azimuth, self.__los = other.azimuth, other.los

	def _process(self, obs):
		super()._process(obs)
		self.__los = tuple(obs[5])
		print('los: ', self.__los, obs[5])
		return self

class ActionsComplex(Enum):
	ROT_LEFT = -0.5 * PI
	FWD_LEFT = -0.25 * PI
	FWD = 0
	FWD_RIGHT = 0.25 * PI
	ROT_RIGHT = 0.5 * PI

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
				space_aux.append(StateComplex(elem1.azimuth, elem2))
		self._space = space_aux

class DiscreteAngularSpace(DiscreteSpace):
	def __init__(self, width, height):
		super().__init__()
		self.__width, self.__height = width, height
		self.__generate_space()
	def __generate_space(self):
		space_aux = []
		for y in range(self.__height):
			for x in range(self.__width):
				if x == 0 and y == 0: continue
				space_aux.append(round(atan2(y,x), 2))
				space_aux.append(round(atan2(-y,x), 2))
				space_aux.append(round(atan2(-y,-x), 2))
				space_aux.append(round(atan2(y,-x), 2))

		space_aux = list(dict.fromkeys(space_aux))
		space_aux.sort()
		
		for elem in space_aux: 
			self._space.append(StateSimple(elem))

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
		elif self.__shape == '-':
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
	@property
	def shape(self):
		return self.__shape

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
	def __init__(self, action_type):
		super().__init__()
		self.__theta = 0
		if action_type == 'complex':
			self.kinematics = self.kinematics_complex
			self.__action_type = action_type
		else:
			self.kinematics = self.kinematics_simple
			self.__action_type = action_type

	@staticmethod
	def normalize_angle(ang):
		ang = ((ang % TWO_PI) + TWO_PI) % TWO_PI
		if ang > PI:
			ang -= TWO_PI
		return ang

	def kinematics_simple(self, action):
		if action == Actions.FWD:
			#x, y = self._pos[0] + cos(self.__theta), self._pos[1] + sin(self.__theta)
			#if x < 0 or y < 0:
			#	print('\n\t\t\t SKIP x,y:', x,y)
			#	return self._pos[0], self._pos[1], self.__theta
			return int(round(self._pos[0] + cos(self.__theta))), int(round(self._pos[1] + sin(self.__theta))), self.__theta
		elif action == Actions.ROT_LEFT:
			return int(self._pos[0]), int(self._pos[1]), round(self.normalize_angle(self.__theta - PI_2), 2)
		elif action == Actions.ROT_RIGHT:
			return int(self._pos[0]), int(self._pos[1]), round(self.normalize_angle(self.__theta + PI_2), 2)
		else:
			raise Exception('Not a valid action')

	def kinematics_complex(self, action):
		if action == ActionsComplex.FWD:
			return int(round(self._pos[0] + cos(self.__theta))), \
			int(round(self._pos[1] + sin(self.__theta))), \
			self.__theta

		elif action == ActionsComplex.FWD_LEFT:
			return int(round(self._pos[0] + cos(self.__theta + ActionsComplex.FWD_LEFT.value))), \
			int(round(self._pos[1] + sin(self.__theta + ActionsComplex.FWD_LEFT.value))), \
			round(self.normalize_angle(self.__theta + ActionsComplex.FWD_LEFT.value), 2)

		elif action == ActionsComplex.FWD_RIGHT:
			return int(round(self._pos[0] + cos(self.__theta + ActionsComplex.FWD_RIGHT.value))), \
			int(round(self._pos[1] + sin(self.__theta + ActionsComplex.FWD_RIGHT.value))), \
			round(self.normalize_angle(self.__theta + ActionsComplex.FWD_RIGHT.value), 2)

		elif action == ActionsComplex.ROT_LEFT:
			return int(self._pos[0]), int(self._pos[1]), round(self.normalize_angle(self.__theta - PI_2), 2)

		elif action == ActionsComplex.ROT_RIGHT:
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
		self.__k = 0.5
		if isinstance(self.__environment, ObstacleEnvironment):
			self.reward = self.collision_avoidance
		else:
			self.reward = self.collision_free
	def __call__(self, cur_state, action, next_state, event = None):
		print('[RewardFunction] processing...')
		if cur_state is not None and next_state is not None:
			print('[RewardFunction] returning real value')
			return self.reward(cur_state, action, next_state, event)
		else:
			print('[RewardFunction] returning UNDEFINED')
			return self.UNDEFINED

	def collision_free(self, cur_state, action, next_state, event = None):
		if (next_state.azimuth == 0.0 or event == Entities.GOAL.value):
			print('reward: GOAL')
			return self.GOAL_PRIZE

		elif event is None:
			print('reward: goal reaching')
			return self.goal_reaching(cur_state.azimuth, next_state.azimuth)

		elif event == Entities.OBSTACLE.value:
			print('reward: COLLIDED')
			return self.COLLISION_PENALTY

		else:
			print('reward: undefined')
			return self.UNDEFINED

	def collision_avoidance(self, cur_state, action, next_state, event = None):
		# 1. Goal reached
		if next_state.azimuth == 0 or event == Entities.GOAL.value and Entities.OBSTACLE.value != next_state.los:
			print('reward: GOAL')
			return self.GOAL_PRIZE
		# 2. Obstacle collision
		elif event == Entities.OBSTACLE.value:
			print('reward: COLLIDED')
			return self.COLLISION_PENALTY
		# 3. If normal navigation
		elif event is None:
			if Entities.OBSTACLE.value == next_state.los or Entities.OBSTACLE.value == cur_state.los:
				print('reward: obstacle avoidance')
				return self.__k * self.obstacle_avoidance(cur_state.los, next_state.los) \
				+ (1 - self.__k) * self.goal_reaching(cur_state.azimuth, next_state.azimuth)
			else:
				print('reward: goal reaching')
				return self.goal_reaching(cur_state.azimuth, next_state.azimuth)
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
	def __init__(self, w, h, action_type):
		self._w, self._h = w, h
		self._action_type = action_type
		self._build_arena()

		# Entities
		self._agent, self._goal = EnvAgent(action_type), EnvEntity()
		self._entities = {
			Entities.AGENT: self._agent,
			Entities.GOAL: self._goal,
		}

		# State space
		self._state_space = DiscreteAngularSpace(self._w, self._h)

		# Reward function
		self._reward_function = RewardFunction(self)

		# Functions
		self.step = self.step_simple if self._action_type == 'simple' else self.step_complex

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

	@property
	def action_type(self):
		return self._action_type

	@abstractmethod
	def reset(self, event = None):
		pass
		
	def _build_arena(self):
		self._arena = np.zeros((self._w, self._h), dtype = np.int8)

	def observe(self):
		return self._cur_state

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

	# A physics iteration in the environment with simple actions
	@abstractmethod
	def step_simple(self, action):
		pass
	# A physics iteration in the environment with complex actions
	@abstractmethod
	def step_complex(self, action):
		pass
	# Processing agent state
	@abstractmethod
	def make_state(self):
		pass

class EmptyEnvironment(Environment):
	def __init__(self, w, h, action_type = 'simple'):
		self._cur_state, self._next_state = StateSimple(), StateSimple() # Initializing states
		super().__init__(w, h, action_type) # Initializing arena, reward, state space, action space and entities
		self.reset() # Reseting arena

	def reset(self):
		print('>>> Reseting')
		self._build_arena()
		self._place_entity_random(Entities.AGENT)
		self._place_entity_random(Entities.GOAL)
		self._cur_state(self.make_state()) # Define initial state

	# When invalid action, skip to next iteration
	def skip(self):
		print('>>> Skipping')
		return self._cur_state, False, RewardFunction.UNDEFINED, Entities.VOID.value

	def step_simple(self, action):
		self._arena[self._agent.y,self._agent.x] = Entities.EMPTY.value # Clean old agent info in the arena

		x, y, ang = self._agent.kinematics(action) # Predict next pose

		if x < 0 or x >= self._w or y < 0 or y >= self._h: # If next pose is out of bounds, return and skip to next decision
			return self.skip()

		# Choose action
		if action == Actions.FWD:
			neighbour = self._arena[y,x] # Get neighbour of next pose
			
			if neighbour == Entities.GOAL.value: # If neighbour is the goal, finish episode
				self.reset()
				return self._cur_state, True, self._reward_function(self._cur_state, action, self._cur_state, neighbour), neighbour

			self._agent.move(x, y, ang) # Move to predicted pose

			self._next_state(self.make_state())	# Observe next state
	
		else: # action == ROT_LEFT or ROT_RIGHT
			if x < 0 or x == self._w or y < 0 or y == self._h: # If next pose is out of bounds, neighbour is None
				neighbour = None
			else:
				neighbour = self._arena[y,x]
			
			if neighbour == Entities.GOAL.value: # If got goal, quit
				self.reset()
				return self._cur_state, True, self._reward_function(self._cur_state, action, self._cur_state, neighbour), neighbour

			self._agent.move(x, y, ang) # Move to predicted pose

			self._next_state(self.make_state()) 			# Observe after acting

		self._arena[self._agent.y, self._agent.x] = Entities.AGENT.value

		return self._next_state, False, self._reward_function(self._cur_state, action, self._next_state, None), neighbour

	def step_complex(self, action):
		# Clean old agent info
		self._arena[self._agent.y,self._agent.x] = Entities.EMPTY.value
		
		# Save last state
		self._cur_state(self.make_state())
		
		# Predict next pose
		x, y, ang = self._agent.kinematics(action)

		# Choose action
		if action == Actions.FWD:
			# If next pose is out of bounds, return and skip to next decision
			if x < 0 or x == self._w or y < 0 or y == self._h:
				return self.skip(self._cur_state)

			# Get neighbour of next pose
			neighbour = self._arena[y,x]

			# Move to predicted pose
			self._agent.move(x, y, ang)

			# make_state next state
			self._next_state(self.make_state())

			# If neighbour is the goal, finish episode
			if neighbour == Entities.GOAL.value:
				return self._next_state, True, self._reward_function(self._cur_state, action, self._next_state, neighbour), neighbour
		# action == ROT_LEFT or ROT_RIGHT
		else:
			# If next pose is out of bounds, neighbour is None
			if x < 0 or x == self._w or y < 0 or y == self._h:
				neighbour = None
			else:
				neighbour = self._arena[y,x]

			# Move to predicted pose
			self._agent.move(x, y, ang)

			# make_state next state
			#next_state(self.make_state())

		self._arena[self._agent.y, self._agent.x] = Entities.AGENT.value

		# make_state next state
		self._next_state(self.make_state())
		return self._next_state, False, self._reward_function(self._cur_state, action, self._next_state, None), neighbour

	def make_state(self):
		print('make_state (EmptyEnv)')
		return (	
			self._agent.x, 
			self._agent.y, 
			self._agent.theta,
			self._goal.x, 
			self._goal.y
		)

class ObstacleEnvironment(Environment):
	def __init__(self, w, h, num_obstacles, action_type = 'complex', los_type = '-'):
		#self.__agent_state = StateComplex()
		self._cur_state, self._next_state = StateComplex(), StateComplex()
		self.__los_type = los_type
		if num_obstacles >= w * h:
			raise ValueError('Number of obstacles needs to be lower than width * height')
		if num_obstacles > 0:
			self.__num_obstacles = num_obstacles
		else:
			raise ValueError('Invalid number of obstacles')
		self.__obstacles = []

		super().__init__(w,h,action_type)

		# Add obstacles
		for i in range(self.__num_obstacles):
			self.__obstacles.append(EnvEntity())
		self._entities[Entities.OBSTACLE] = self.__obstacles

		# State space
		self.__los_state_space = DiscreteLineOfSightSpace(1,self.__los_type)
		self._state_space.concat(self.__los_state_space)
		print(f'>> State space of len {len(self)} was created: {self}')

		# Auxiliar variables
		self.__translation_walls = np.array([1,1]).reshape(2,1)

		# Reset
		self.reset()

		# Functions
		self.step = self.step_simple if action_type == 'simple' else self.step_complex

	def reset(self, terminal = False, cur_state = None, next_state = None, action = None, event = None):
		print('>>>Reseting')
		self._arena[1:self._h + 1, 1:self._w + 1] = [Entities.EMPTY.value]
		self._place_entity_random(Entities.AGENT)
		self._place_entity_random(Entities.GOAL)
		self.__generate_obstacles(self.__num_obstacles)

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

	def step_simple(self, action):
		# Clean old agent info
		self._arena[self._agent.y + 1,self._agent.x + 1] = Entities.EMPTY.value

		# Save last state
		self._cur_state(self.make_state())
		
		# Predict next pose
		x, y, ang = self._agent.kinematics(action)

		# Check entity for next position
		neighbour = self._arena[y + 1, x + 1]

		# Act
		if action == Actions.FWD:			
			# If collision, quit
			if neighbour == Entities.OBSTACLE.value:
				return self.reset(True, self._cur_state, self._cur_state, action, neighbour)
			# If got goal, quit
			if neighbour == Entities.GOAL.value:
				return self.reset(True, self._cur_state, self._cur_state, action, neighbour)

			# Move
			self._agent.move(x, y, ang)

			# Observe next state
			self._next_state(self.make_state())

		# action == ROT_LEFT or ROT_RIGHT
		else:
			# If got goal, quit
			if neighbour == Entities.GOAL.value:
				return self.reset(True, self._cur_state, self._cur_state, action, neighbour)
			# Move
			self._agent.move(x, y, ang)

			# Observe next state
			self._next_state(self.make_state())

		# Place agent in arena in new position
		self._arena[self._agent.y + 1, self._agent.x + 1] = Entities.AGENT.value

		return self._next_state, False, self._reward_function(self._cur_state, action, self._next_state, None), neighbour

	def step_complex(self, action):
		# Clean old agent info
		self._arena[self._agent.y + 1,self._agent.x + 1] = Entities.EMPTY.value

		# Save last state
		self._cur_state(self.make_state())
		
		# Predict next pose
		x, y, ang = self._agent.kinematics(action)

		# Check entity for next position
		neighbour = self._arena[y + 1, x + 1]

		# Act
		if action == ActionsComplex.FWD or ActionsComplex.FWD_LEFT or ActionsComplex.FWD_RIGHT:			
			# If collision, quit
			if neighbour == Entities.OBSTACLE.value:
				return self.reset(True, self._cur_state, self._cur_state, action, neighbour)
			# If got goal, quit
			if neighbour == Entities.GOAL.value:
				return self.reset(True, self._cur_state, self._cur_state, action, neighbour)

			# Move
			self._agent.move(x, y, ang)

			# Observe next state
			self._next_state(self.make_state())

		# action == ROT_LEFT or ROT_RIGHT
		else:
			# If got goal, quit
			if neighbour == Entities.GOAL.value:
				return self.reset(True, self._cur_state, self._cur_state, action, neighbour)
			# Move
			self._agent.move(x, y, ang)

			# Observe next state
			self._next_state(self.make_state())

		# Place agent in arena in new position
		self._arena[self._agent.y + 1, self._agent.x + 1] = Entities.AGENT.value

		return self._next_state, False, self._reward_function(self._cur_state, action, self._next_state, None), neighbour

	def __generate_los(self, terminal = False):
		if terminal:
			return self._cur_state.los
		rot = np.array([
			[cos(self._agent.theta), sin(self._agent.theta)],
			[sin(self._agent.theta), -cos(self._agent.theta)]
		])
		translation_agent = np.array([self._agent.x, self._agent.y]).reshape(2,1)
		los_positions = np.rint(np.dot(rot, self.__los_state_space.vectors) + translation_agent + self.__translation_walls).astype(np.int8)
		los = []
		for i in range(los_positions.shape[1]):
			los.append(self._arena[los_positions[1,i]][los_positions[0,i]])
		return tuple(los)
	
	def make_state(self, terminal = False):
		return (
			self._agent.x,
			self._agent.y,
			self._agent.theta,
			self._goal.x,
			self._goal.y,
			self.__generate_los(terminal)
		)
