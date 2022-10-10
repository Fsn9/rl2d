from random import randint, choice
from enum import Enum
import numpy as np
from math import atan2, pi, cos, sin, isclose, sqrt
from itertools import product
from abc import ABC, abstractmethod
import yaml
import os

DECIMAL_PLACES = 3
PI = round(pi, DECIMAL_PLACES)
PI_2 = round(pi * 0.5, DECIMAL_PLACES)
PI_4 = round(pi * 0.25, DECIMAL_PLACES)
TWO_PI = round(2 * pi, DECIMAL_PLACES)
SQRT_2 = round(sqrt(2), DECIMAL_PLACES)
SQRT_5 = round(sqrt(5), DECIMAL_PLACES)
axis_angles = [-PI, -PI_2, 0.0, PI_2, PI]
TOLERANCE_ANGLE_EQUALITY = 0.1

class StateSimple:
	def __init__(self, azimuth = 0.0):
		self._azimuth = azimuth

	def __call__(self, obs):
		return self._process(obs)

	def __eq__(self, other):
		return self._azimuth == other.azimuth

	def __repr__(self):
		return 'Ss (' + str(self._azimuth) + ')'

	@property
	def azimuth(self):
		return self._azimuth

	def get_data_from(self, other):
		self._azimuth = other.azimuth

	def _process(self, obs):
		## prev
		gx_ag, gy_ag = obs[3] - obs[0], obs[4] - obs[1]
		## other approach
		#gx_ag, gy_ag = obs[3] - obs[0], obs[1] - obs[4]
		cos_rot, sin_rot = cos(obs[2]), sin(obs[2])
		# self._azimuth = round(atan2(-sin_rot * gx_ag + cos_rot * gy_ag, cos_rot * gx_ag + sin_rot * gy_ag), DECIMAL_PLACES)
		## prev
		self._azimuth = round(atan2(round(sin_rot * gx_ag - cos_rot * gy_ag), round(cos_rot * gx_ag + sin_rot * gy_ag)), DECIMAL_PLACES)
		## other approach
		#self._azimuth = round(atan2(round(sin_rot * gx_ag + cos_rot * gy_ag), round(cos_rot * gx_ag - sin_rot * gy_ag)), DECIMAL_PLACES)

		# because the state space does not have -PI and PI is the same as -PI
		if self._azimuth == -PI: self._azimuth = PI
		return self

class StateComplex(StateSimple):
	def __init__(self, azimuth = 0.0, los = ()):
		super().__init__(azimuth)
		self.__los = los

	def __repr__(self):
		return "Sc (" + str(self._azimuth) + "," + str(self.__los) + ")"

	def __eq__(self, other):
		return self._azimuth == other.azimuth and self.__los == other.los

	@property
	def los(self):
		return self.__los

	def get_data_from(self, other):
		self._azimuth, self.__los = other.azimuth, other.los

	def _process(self, obs):
		super()._process(obs)
		self.__los = tuple(obs[5])
		return self

class ActionsComplex(Enum):
	ROT_LEFT = -PI_2
	FWD_LEFT = -PI_4
	FWD = 0
	FWD_RIGHT = PI_4
	ROT_RIGHT = PI_2

class Actions(Enum):
	LEFT_LEFT = -PI_2
	LEFT = -PI_4
	STAY = 0
	RIGHT = PI_4
	RIGHT_RIGHT = PI_2

class DiscreteSpace:
	def __init__(self):
		self._space = []
	def __call__(self):
		return self._space
	def __repr__(self):
		return str(self._space)
	def __str__(self):
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
		dims = self.expand_dims((self.__height, self.__width))
		space_aux = []
		for y in range(dims[0]):
			for x in range(dims[1]):
				space_aux.append(round(atan2(y,x), DECIMAL_PLACES))
				space_aux.append(round(atan2(-y,x), DECIMAL_PLACES))
				space_aux.append(round(atan2(-y,-x), DECIMAL_PLACES))
				space_aux.append(round(atan2(y,-x), DECIMAL_PLACES))

		space_aux = list(dict.fromkeys(space_aux))
		space_aux.sort()
		
		for elem in space_aux: 
			self._space.append(StateSimple(elem))
	@property
	def raw_repr(self):
		sr = []
		for state in self._space:
			sr.append(state.azimuth)
		return sr

	# Depending on the grid resolution one needs to consider augmented state spaces
	# 9 is the maximum state space dimension
	@staticmethod
	def expand_dims(dims):
		dims_map = {
			(3,3) : (3,3),
			(4,4) : (5,5),
			(5,5) : (6,6),
			(6,6) : (7,7),
			(7,7) : (9,9),
			(8,8) : (10,10),
			(9,9) : (12,12)
		}
		return dims_map[dims]

class DiscreteLineOfSightSpace(DiscreteSpace):
	def __init__(self, range, shape = 'T'):
		super().__init__()
		self.__range, self.__shape = range, shape
		self.__vectors = []
		self.__generate_space()
	def __generate_space(self):
		# Generating all the possible line of sight states
		## Without AGENT and VOID entities
		los_entities = list(filter(lambda ent: (ent != Entities.AGENT.value and ent != Entities.VOID.value), 
			list(map(lambda x: (x.value), Entities))))

		if self.__shape == '|':
			self._space = [los for los in product([ent for ent in los_entities], repeat = 2 * self.__range)]
			self._space = list(filter(lambda los: (not los.count(2) > 1), self._space))
			self.__vectors = np.array([[1,2],[0,0]])
			self.__vectors_diag = np.array([[SQRT_2, SQRT_2],[0,0]])
		elif self.__shape == '-':
			self._space = [los for los in product([ent for ent in los_entities], repeat = 3 * self.__range)]
			## Some filtering
			### Filter cases with more than 1 goal. It is only possible to detect one goal in the neighborhood
			self._space = list(filter(lambda los: (not los.count(2) > 1), self._space))
			## Vectors depending on the los shape
			self.__vectors = np.array([[1,1,1],[1,0,-1]])
		elif self.__shape == 'T':
			self._space = [los for los in product([ent for ent in los_entities], repeat = 4 * self.__range)]
			self._space = list(filter(lambda los: (not los.count(2) > 1), self._space))
			self.__vectors = np.array([[1,2,2,2],[0,1,0,-1]])
			self.__vectors_diag = np.array([[SQRT_2, 3 * SQRT_2 * 0.5, 2 * SQRT_2, 3 * SQRT_2 * 0.5],[0, SQRT_2 * 0.5, 0, -SQRT_2 * 0.5]])
		else: # shape == '*'
			self._space = [los for los in product([ent for ent in los_entities], repeat = 6 * self.__range)]
	@property
	def vectors(self):
		return self.__vectors
	@property
	def vectors_diag(self):
		return self.__vectors_diag
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
	def __init__(self, x = 0, y = 0):
		self._pos = [x,y]

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
		if action in Actions:
			return int(round(self._pos[0] + cos(self.__theta + action.value))), \
			int(round(self._pos[1] + sin(self.__theta))), \
			self.__theta + action.value
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

		# Define type of reward function
		if isinstance(self.__environment, ObstacleEnvironment):
			self.reward = self.collision_avoidance2
		else:
			self.reward = self.collision_free

		# Parameters
		## Goal reaching
		### Next angle version
		self.__max_rew_gr, self.__min_rew_gr = 0.5, -0.5
		self.__amplitude_rew_gr = self.__max_rew_gr - self.__min_rew_gr
		self.__max_angle, self.__min_angle = PI, 0
		self.__slope_rew_gr = self.__amplitude_rew_gr / (self.__min_angle - self.__max_angle)
		self.__intercept_rew_gr = self.__max_rew_gr
		### Angle variation version
		#self.__max_angle_variation = round(abs(atan2(1, -1)), DECIMAL_PLACES)
		self.__max_angle_variation = PI_4
		self.__den = 2 * self.__max_angle_variation
		self.__inv_den = 1.0 / self.__den
		## Obstacle avoidance
		self.__k = 0.8

	def __call__(self, cur_state, action, next_state, event = None):
		if cur_state is not None and next_state is not None:
			return self.reward(cur_state, action, next_state, event)
		else:
			print('[RewardFunction] returning UNDEFINED')
			return self.UNDEFINED

	def collision_free(self, cur_state, action, next_state, event = None):
		if next_state.azimuth == 0 or event == Entities.GOAL.value:
			print('reward: GOAL')
			return self.GOAL_PRIZE

		elif event is None:
			print('reward: goal reaching')
			return self.goal_reaching(cur_state.azimuth, next_state.azimuth, action)

		else:
			print('reward: undefined')
			return self.UNDEFINED

	def collision_avoidance2(self, cur_state, action, next_state, event = None):
		if event == Entities.OBSTACLE.value:
			print('[RewardFunction] COLLIDED:',self.COLLISION_PENALTY)
			return self.COLLISION_PENALTY
		elif event == Entities.VOID.value:
			print('[RewardFunction] UNDEFINED:', self.UNDEFINED)
			return self.UNDEFINED
		else:
			return self.goal_reaching(cur_state.azimuth, next_state.azimuth, action)

	def forward_incentive(self, action):
		if action == Actions.FWD:
			print('\tIncentivizing')
			return self.GOAL_PRIZE
		else:
			return 0

	def goal_reaching(self, cur_state, next_state, action):
		return self.normalize(-self.__max_angle_variation, 
				self.__max_angle_variation, 
				abs(cur_state) - abs(next_state)
				)
	def obstacle_avoidance(self, cur_los, next_los):
		return (np.array(next_los) - np.array(cur_los)).sum()

	# normalize between [-0.5, 0.5]
	def normalize(self, min, max, val):
		return (val - min)  * self.__inv_den - 0.5

class Environment(ABC):
	def __init__(self, w, h, evaluation):
		self._w, self._h = w, h
		self._evaluation = evaluation
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

	def _define_resolution_parameters(self):
		global MIN_FLOAT_TOLERANCE, MAX_FLOAT_TOLERANCE, DECIMAL_PLACES
		MIN_FLOAT_TOLERANCE = round(min([j - i for i, j in zip(self._state_space.raw_repr[:-1], self._state_space.raw_repr[1:])]), DECIMAL_PLACES)
		MAX_FLOAT_TOLERANCE = round(max([j - i for i, j in zip(self._state_space.raw_repr[:-1], self._state_space.raw_repr[1:])]), DECIMAL_PLACES)

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

	# A physics iteration in the environment
	@abstractmethod
	def step(self, action):
		pass

	# Processing agent state
	@abstractmethod
	def make_state(self):
		pass

class EmptyEnvironment(Environment):
	def __init__(self, w, h, evaluation):
		super().__init__(w, h, evaluation) # Initializing arena, reward, state space, action space and entities
		self._cur_state, self._next_state = StateSimple(), StateSimple() # Initializing states
		self._define_resolution_parameters() # Define MIN_FLOAT_TOLERANCE and DECIMAL_PLACES depending on the state space
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

	def step(self, action):
		self._arena[self._agent.y,self._agent.x] = Entities.EMPTY.value # Clean old agent info in the arena

		x, y, ang = self._agent.kinematics(action) # Predict next pose

		# Act
		if action == Actions.FWD:
			if x < 0 or x >= self._w or y < 0 or y >= self._h: # If next pose is out of bounds, return and skip to next decision
				return self.skip()
			neighbour = self._arena[y,x] # Get neighbour of next pose
	
		else: # action == ROT_LEFT or ROT_RIGHT
			if x < 0 or x == self._w or y < 0 or y == self._h: # If next pose is out of bounds, neighbour is None
				neighbour = None
			else:
				neighbour = self._arena[y,x]
			
		if neighbour == Entities.GOAL.value: # If got goal, quit
			return self._cur_state, True, self._reward_function(self._cur_state, action, self._cur_state, neighbour), neighbour

		self._agent.move(x, y, ang) # Move to predicted pose

		self._next_state(self.make_state()) # Observe after acting

		self._arena[self._agent.y, self._agent.x] = Entities.AGENT.value

		return self._next_state, False, self._reward_function(self._cur_state, action, self._next_state, None), neighbour

	def make_state(self):
		return (	
			self._agent.x, 
			self._agent.y, 
			self._agent.theta,
			self._goal.x, 
			self._goal.y
		)

class ObstacleEnvironment(Environment):
	def __init__(self, w, h, num_obstacles, evaluation, los_type = 'T'):
		super().__init__(w,h,evaluation)
		self._cur_state, self._next_state = StateComplex(), StateComplex()
		self.__los_type = los_type
		if num_obstacles >= w * h:
			raise ValueError('Number of obstacles needs to be lower than width * height')
		if num_obstacles > 0:
			self.__num_obstacles = num_obstacles
		else:
			raise ValueError('Invalid number of obstacles')
		self.__obstacles = []

		# State space
		self.__los_state_space = DiscreteLineOfSightSpace(1,self.__los_type)
		self._state_space.concat(self.__los_state_space)
		self._define_resolution_parameters() # Define MIN_FLOAT_TOLERANCE and DECIMAL_PLACES depending on the state space

		# Auxiliar variables
		self.__translation_walls = np.array([1,1]).reshape(2,1)

		# Start drawing environment
		if self._evaluation: 
			self.__scenarios = []
			scenario_files = sorted(os.listdir('../evaluation_scenarios'))
			for file in scenario_files:
				with open(os.path.join('../evaluation_scenarios',file), "r") as scenario_file:
					try:
						scenario = yaml.safe_load(scenario_file)
					except yaml.YAMLError as exc:
						print(exc)
					xs, ys = scenario['obstacles']['x'], scenario['obstacles']['y']
					if not any(scenario['agent']):
						pass
					else:
						xa, ya, theta = scenario['agent'][0], scenario['agent'][1], scenario['agent'][2]
						# @TODO: handle theta
						if xa >= self._w or xa < 0 or ya >= self._h or ya < 0:
							raise ValueError(f'Agent position ({xa},{ya}) not valid for scenario {file}. \
								It needs to be inside the environment dimensions that are ({self._w}x{self._h})')
					if not any(scenario['goal']):
						pass
					else:
						xg, yg = scenario['goal'][0], scenario['goal'][1]
						if xg >= self._w or xg < 0 or yg >= self._h or yg < 0:
							raise ValueError(f'Goal position ({xg},{yg}) not valid for scenario {file}. \
								It needs to be inside the environment dimensions that are ({self._w}x{self._h})')
						if any(scenario['agent']):
							if xa == xg and ya == yg:
								raise ValueError(f'Goal position ({xg},{yg}) overlaps agent position ({xa},{ya}) \
									for scenario {file}.')
					for x, y in zip(xs,ys):
						if x >= self._w or x < 0 or y >= self._h or y < 0:
							raise ValueError(f'Obstacle position ({x},{y}) not valid for scenario {file}. \
							It needs to be inside the environment dimensions that are ({self._w}x{self._h})')
						if any(scenario['agent']):
							if x == xa and y == ya:
								raise ValueError(f'Obstacle position ({x},{y}) overlaps agent position ({xa},{ya}) \
									for scenario {file}.')
						if any(scenario['goal']):
							if x == xg and y == yg:
								raise ValueError(f'Obstacle position ({x},{y}) overlaps goal position ({xg},{yg}) \
									for scenario {file}.')
					scenario['name'] = file
					self.__scenarios.append(scenario)
			self.__cur_scenario_idx = 0
			self.play_next_scenario()
			self.__evaluation_finished = False
		else:
			# Add obstacles
			for i in range(self.__num_obstacles):
				self.__obstacles.append(EnvEntity())
				self._entities[Entities.OBSTACLE] = self.__obstacles
			self.reset()

	@property
	def scenarios(self):
		return self.__scenarios
	@property
	def evaluation_finished(self):
		return self.__evaluation_finished
	@property
	def cur_scene_name(self):
		return self.__scenarios[self.__cur_scenario_idx - 1]['name']
	@property
	def los_type(self):
		return self.__los_type

	def play_next_scenario(self):
		if self.__cur_scenario_idx == len(self.__scenarios):
			return None
		print(f'>>Playing scenario {self.__scenarios[self.__cur_scenario_idx]["name"]}')
		self._arena[1:self._h + 1, 1:self._w + 1] = [Entities.EMPTY.value]

		self.__place_obstacles(self.__scenarios[self.__cur_scenario_idx]['obstacles'])

		if any(self.__scenarios[self.__cur_scenario_idx]['agent']):
			self._arena[self.__scenarios[self.__cur_scenario_idx]['agent'][1] + 1, self.__scenarios[self.__cur_scenario_idx]['agent'][0] + 1] = Entities.AGENT.value
			self._agent.move(self.__scenarios[self.__cur_scenario_idx]['agent'][0], self.__scenarios[self.__cur_scenario_idx]['agent'][1], self.__scenarios[self.__cur_scenario_idx]['agent'][2])
		else:
			self._place_entity_random(Entities.AGENT)

		if any(self.__scenarios[self.__cur_scenario_idx]['goal']):
			self._arena[self.__scenarios[self.__cur_scenario_idx]['goal'][1] + 1, self.__scenarios[self.__cur_scenario_idx]['goal'][0] + 1] = Entities.GOAL.value
			self._goal.move(self.__scenarios[self.__cur_scenario_idx]['goal'][0], self.__scenarios[self.__cur_scenario_idx]['goal'][1])
		else:
			self._place_entity_random(Entities.GOAL)

		self._cur_state(self.make_state())

		self.__cur_scenario_idx += 1

		return self.__scenarios[self.__cur_scenario_idx - 1]['name']

	def reset(self, terminal = False, cur_state = None, next_state = None, action = None, event = None):
		print('>>>Reseting')
		self._arena[1:self._h + 1, 1:self._w + 1] = [Entities.EMPTY.value]
		self._place_entity_random(Entities.AGENT)
		self._place_entity_random(Entities.GOAL)
		self.__generate_obstacles(self.__num_obstacles)
		self._cur_state(self.make_state()) # Define initial state

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

	def __place_obstacles(self, positions):
		del self.__obstacles[:]
		for idx,(x,y) in enumerate(zip(positions['x'], positions['y'])):
			self.__obstacles.append(EnvEntity(x,y))
			self._arena[y + 1,x + 1] = Entities.OBSTACLE.value
		self._entities[Entities.OBSTACLE] = self.__obstacles

	def step(self, action):
		# Clean old agent info
		self._arena[self._agent.y + 1,self._agent.x + 1] = Entities.EMPTY.value

		# Predict next pose
		x, y, ang = self._agent.kinematics(action)

		# Check entity for next position
		neighbour = self._arena[y + 1, x + 1]

		# Act		
		if neighbour == Entities.OBSTACLE.value:
			return neighbour, True, RewardFunction.COLLISION_PENALTY, neighbour
		if neighbour == Entities.GOAL.value:
			return neighbour, True, RewardFunction.GOAL_PRIZE, neighbour

		self._agent.move(x, y, ang) # Move to predicted pose

		self._next_state(self.make_state()) # Observe after acting

		self._arena[self._agent.y + 1, self._agent.x + 1] = Entities.AGENT.value # Place agent in arena in new position

		return self._next_state, False, self._reward_function(self._cur_state, action, self._next_state, None), neighbour

	def __generate_los(self, terminal = False):
		if terminal:
			return self._cur_state.los
		rot = np.array([
			[cos(self._agent.theta), sin(self._agent.theta)],
			[sin(self._agent.theta), -cos(self._agent.theta)]
		])
		translation_agent = np.array([self._agent.x, self._agent.y]).reshape(2,1)
		if sum(np.isclose(self._agent.theta, axis_angles, atol = TOLERANCE_ANGLE_EQUALITY)) == 1:
			los_positions = np.rint(np.dot(rot, self.__los_state_space.vectors) + translation_agent + self.__translation_walls).astype(np.int8)
		else:
			los_positions = np.rint(np.dot(rot, self.__los_state_space.vectors_diag) + translation_agent + self.__translation_walls).astype(np.int8)
		np.clip(los_positions, 0, self._w + 1, out = los_positions)
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
