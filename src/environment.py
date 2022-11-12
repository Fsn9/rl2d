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
SQRT_2_2 = round(sqrt(2) * 0.5, DECIMAL_PLACES)
SQRT_5 = round(sqrt(5), DECIMAL_PLACES)
AXIS_ANGLES = np.array([-PI, -PI_2, 0.0, PI_2, PI])
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
		gx_ag, gy_ag = obs[3] - obs[0], obs[4] - obs[1]
		cos_rot, sin_rot = cos(obs[2]), sin(obs[2])
		self._azimuth = round(atan2(round(sin_rot * gx_ag - cos_rot * gy_ag), round(cos_rot * gx_ag + sin_rot * gy_ag)), DECIMAL_PLACES)
		if self._azimuth == -PI: self._azimuth = PI # because the state space does not have -PI and PI is the same as -PI
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
				if type(self) == DiscreteAngularSpace:
					space_aux.append(StateComplex(elem1.azimuth, elem2))
				else:
					space_aux.append(StateComplex(elem2.azimuth, elem1))
		self._space = space_aux

class DiscreteAngularSpace(DiscreteSpace):
	POSSIBLE_ANGLES = [-PI, -3 * PI_4, -PI_2, -PI_4, 0, PI_4, PI_2, 3 * PI_4, PI]
	def __init__(self, width, height):
		super().__init__()
		self.__width, self.__height = width, height
		self.__generate_space()
		self.__raw_repr = []
		for state in self._space:
			self.__raw_repr.append(state.azimuth)

	def __generate_space(self):
		space_aux = []
		for y in range(self.__height):
			for x in range(self.__width):
				space_aux.append(round(atan2(y,x), DECIMAL_PLACES))
				space_aux.append(round(atan2(-y,x), DECIMAL_PLACES))
				space_aux.append(round(atan2(-y,-x), DECIMAL_PLACES))
				space_aux.append(round(atan2(y,-x), DECIMAL_PLACES))

		# Cases when one coordinate exceeds env dims
		max_dim = self.expand_dims((self.__height, self.__width))
		for x in range(self.__height, max_dim):			
			space_aux.append(round(atan2(x, 1), DECIMAL_PLACES))
			space_aux.append(round(atan2(x, -1), DECIMAL_PLACES))
			space_aux.append(round(atan2(-x, 1), DECIMAL_PLACES))
			space_aux.append(round(atan2(-x, -1), DECIMAL_PLACES))
			space_aux.append(round(atan2(1, x), DECIMAL_PLACES))
			space_aux.append(round(atan2(-1, x), DECIMAL_PLACES))
			space_aux.append(round(atan2(1, -x), DECIMAL_PLACES))
			space_aux.append(round(atan2(-1, -x), DECIMAL_PLACES))

		space_aux = list(dict.fromkeys(space_aux))
		space_aux.sort()
		
		for elem in space_aux: 
			self._space.append(StateSimple(elem))
		print(self._space)
		#exit()

	@property
	def raw_repr(self):
		return self.__raw_repr

	# Depending on the grid resolution one needs to consider augmented state spaces
	# 9 is the maximum state space dimension
	@staticmethod
	def expand_dims(dims):
		# assuming squared environment
		return int(SQRT_2 * (dims[0] - 1)) + 1

class DiscreteLineOfSightSpace(DiscreteSpace):
	TERMINAL_STATES = [(0,0,0), (0,1,0)]
	COLLISION_FREE_STATE = (2,2,2)
	def __init__(self, range_, shape = '-'):
		super().__init__()
		self.__range, self.__shape = range_, shape
		self.__possible_distances = [d for d in range(self.__range + 1)]
		self.__bins = []
		self.__vectors = []
		self.__vectors_diag = []
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
			self.__num_bins = 3
			self.__relaxing_tolerance = 0.05 # to make PI_4 values be digitized the same in -pi/4 < x < pi/4.
			#self.__bins = [-PI_4 + self.__relaxing_tolerance, PI_4 -  self.__relaxing_tolerance]
			self.__bins = [-PI_4 - self.__relaxing_tolerance, PI_4 +  self.__relaxing_tolerance]
			self._space = [los for los in product([ent for ent in self.__possible_distances], repeat = self.__num_bins)]
			## Some filtering
			### Filter cases with more than 1 goal. It is only possible to detect one goal in the neighborhood
			## Vectors depending on the los shape
			self.__basis_vectors = np.array([[1,1,1],[1,0,-1]])
			self.__basis_vectors_diag = np.array([[SQRT_2_2, SQRT_2, SQRT_2_2],[SQRT_2_2, 0, -SQRT_2_2]])
			self.__vectors = np.copy(self.__basis_vectors)
			self.__vectors_diag = np.copy(self.__basis_vectors_diag)
			for i in range(self.__range - 1):
				self.__vectors = np.concatenate((self.__vectors, self.__basis_vectors + np.array([[i+1,i+1,i+1],[0,0,0]])), axis = 1)
				self.__vectors_diag = np.concatenate((self.__vectors_diag, self.__basis_vectors_diag + np.array([[i+SQRT_2,i+SQRT_2,i+SQRT_2],[0,0,0]])), axis = 1)

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
	@property
	def num_bins(self):
		return self.__num_bins
	@property
	def range(self):
		return self.__range
	@property
	def possible_distances(self):
		return self.__possible_distances
	@property
	def bins(self):
		return self.__bins

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

	def __repr__(self):
		return 'Agent with pose: ' + f'({self._pos[0]},{self._pos[1]},{self.__theta})'

	@staticmethod
	def normalize_angle(ang):
		ang = ((ang % TWO_PI) + TWO_PI) % TWO_PI
		if ang > PI:
			ang -= TWO_PI
		return ang

	@staticmethod
	def discretize(x, arr):
		return arr[np.abs(np.array(x)-arr).argmin()]

	def kinematics(self, action):
		if action in Actions:
			if action == Actions.STAY:
				v = 1 if self.__theta in AXIS_ANGLES else SQRT_2
				return int(round(self._pos[0] + v * cos(self.__theta))), int(round(self._pos[1] + v * sin(self.__theta))), self.__theta
			else:
				v = SQRT_2 if self.__theta in AXIS_ANGLES else 1
				return int(round(self._pos[0] + v * cos(self.__theta + action.value * 0.5))), \
				int(round(self._pos[1] + v * sin(self.__theta + action.value * 0.5))), \
				self.discretize(round(self.normalize_angle(self.__theta + action.value), DECIMAL_PLACES), DiscreteAngularSpace.POSSIBLE_ANGLES)
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
	TERMINAL_STATE = -1
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
		self.__max_angle_variation = abs(round(PI, DECIMAL_PLACES))
		self.__den = 2 * self.__max_angle_variation
		self.__inv_den = 1.0 / self.__den
		## Obstacle avoidance
		self.__k = 0.5

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
			return self.goal_reaching(cur_state.azimuth, next_state.azimuth)

		else:
			print('reward: undefined')
			return self.UNDEFINED

	def collision_avoidance2(self, cur_state, action, next_state, event = None):
		# 1. If collision or terminal state
		if event == Entities.OBSTACLE.value or event == "terminal":
			print('[RewardFunction] COLLIDED')
			return self.COLLISION_PENALTY
		elif (event == Entities.GOAL.value or next_state.azimuth == 0.0) and cur_state.los != DiscreteLineOfSightSpace.COLLISION_FREE_STATE:
			print('[RewardFunction] GOAL ACHIEVED')
			return self.goal_reaching(cur_state.azimuth, next_state.azimuth)
		# 2. UNDEFINED cases
		elif event == Entities.VOID.value:
			print('[RewardFunction] UNDEFINED:', self.UNDEFINED)
			return self.UNDEFINED
		# 3. Collision avoidance
		else:
			print('[RewardFunction] GOAL REACHING')
			return self.goal_reaching(cur_state.azimuth, next_state.azimuth)
			los_idx = np.digitize(action.value, self.__environment.state_space.bins)
			dist_margin = min(cur_state.los[los_idx], 2)
			## If distance margin is safe
			if dist_margin >= 2:
				# If goal achieved
				if next_state.azimuth == 0.0:
					print('[RewardFunction] GOAL')
					return self.GOAL_PRIZE
				# otherwise find the goal
				else:
					print('[RewardFunction] GOAL REACHING')
					return self.goal_reaching(cur_state.azimuth, next_state.azimuth)
			else:
				# If goal achieved but distance margin is dangerous
				if cur_state.azimuth == 0.0:
					print('[RewardFunction] SPECIAL CASE dist_margin = 1')
					# Incentivize for strong avoidance
					if action == Actions.LEFT_LEFT or action == Actions.RIGHT_RIGHT:
						print('[RewardFunction] SPECIAL CASE critical turn')
						return 0.0
					# Otherwise penalize
					else:
						print('[RewardFunction] SPECIAL CASE bad action')
						return -0.5
				# If goal is not achieved, look for the goal
				else:
					print('[RewardFunction] OBSTACLE AVOIDANCE')
					if action == Actions.LEFT_LEFT or action == Actions.RIGHT_RIGHT:
						print('[RewardFunction] OBSTACLE AVOIDANCE - avoided')
						return 0.5 * self.goal_reaching(cur_state.azimuth, next_state.azimuth)
					else:
						print('[RewardFunction] OBSTACLE AVOIDANCE - critical')
						return -0.5 + 0.5 * self.goal_reaching(cur_state.azimuth, next_state.azimuth)

	def goal_reaching(self, cur_state, next_state):
		return abs(cur_state) - abs(next_state)
		return self.normalize(-self.__max_angle_variation, 
				self.__max_angle_variation, 
				abs(cur_state) - abs(next_state)
				)
	def obstacle_avoidance(self, cur_los, action):
		print(f'action: {action}, {action.value}')
		print(f'los:',{cur_los})
		print(f'roa: {(cur_los[np.digitize(action.value, self.__environment.state_space.bins)] - self.__environment.state_space.range) / self.__environment.state_space.range}')
		return (cur_los[np.digitize(action.value, self.__environment.state_space.bins)] - self.__environment.state_space.range) / self.__environment.state_space.range

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

	def get_agent_x(self):
		return self._agent.x
	
	def get_agent_y(self):
		return self._agent.y

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
					self._agent.move(c, r, choice(DiscreteAngularSpace.POSSIBLE_ANGLES))
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

	def _out_of_bounds(self, x, y):
		return x < 0 or y < 0 or x >= self._w or y >= self._h

class EmptyEnvironment(Environment):
	def __init__(self, w, h, evaluation):
		super().__init__(w, h, evaluation) # Initializing arena, reward, state space, action space and entities
		self._cur_state, self._next_state = StateSimple(), StateSimple() # Initializing states
		self.reset() # Reseting arena

	def reset(self):
		print('>>> Reseting')
		self._build_arena()
		self._place_entity_random(Entities.AGENT)
		self._place_entity_random(Entities.GOAL)
		self._cur_state(self.make_state()) # Define initial state

	def step(self, action):
		self._arena[self._agent.y,self._agent.x] = Entities.EMPTY.value # Clean old agent info in the arena

		x, y, ang = self._agent.kinematics(action) # Predict next pose

		# Act
		if self._out_of_bounds(x,y): # If next pose is out of bounds, return and skip to next decision
			return self._cur_state, False, RewardFunction.UNDEFINED, Entities.VOID.value, (x,y)

		neighbour = self._arena[y,x] # Get neighbour of next pose
		
		if neighbour == Entities.GOAL.value: # If got goal, quit
			return self._cur_state, True, self._reward_function(self._cur_state, action, self._cur_state, neighbour), neighbour, (x,y)

		self._agent.move(x, y, ang) # Move to predicted pose

		self._next_state(self.make_state()) # Observe after acting

		self._arena[self._agent.y, self._agent.x] = Entities.AGENT.value

		return self._next_state, False, self._reward_function(self._cur_state, action, self._next_state, None), neighbour, (x,y)

	def make_state(self):
		return (	
			self._agent.x, 
			self._agent.y, 
			self._agent.theta,
			self._goal.x, 
			self._goal.y
		)

class ObstacleEnvironment(Environment):
	def __init__(self, w, h, num_obstacles, evaluation, los_type = '-'):
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
		self.__los_state_space = DiscreteLineOfSightSpace(2, self.__los_type)
		self.__los_state_space.concat(self._state_space)
		self._state_space = self.__los_state_space

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
					if not scenario['agent']:
						pass
					else:
						xa, ya, theta = scenario['agent'][0], scenario['agent'][1], scenario['agent'][2]
						# @TODO: handle theta
						if xa >= self._w or xa < 0 or ya >= self._h or ya < 0:
							raise ValueError(f'Agent position ({xa},{ya}) not valid for scenario {file}. \
								It needs to be inside the environment dimensions that are ({self._w}x{self._h})')
					if not scenario['goal']:
						pass
					else:
						xg, yg = scenario['goal'][0], scenario['goal'][1]
						if xg >= self._w or xg < 0 or yg >= self._h or yg < 0:
							raise ValueError(f'Goal position ({xg},{yg}) not valid for scenario {file}. \
								It needs to be inside the environment dimensions that are ({self._w}x{self._h})')
						if scenario['agent']:
							if xa == xg and ya == yg:
								raise ValueError(f'Goal position ({xg},{yg}) overlaps agent position ({xa},{ya}) \
									for scenario {file}.')
					for x, y in zip(xs,ys):
						if x >= self._w or x < 0 or y >= self._h or y < 0:
							raise ValueError(f'Obstacle position ({x},{y}) not valid for scenario {file}. \
							It needs to be inside the environment dimensions that are ({self._w}x{self._h})')
						if scenario['agent']:
							if x == xa and y == ya:
								raise ValueError(f'Obstacle position ({x},{y}) overlaps agent position ({xa},{ya}) \
									for scenario {file}.')
						if scenario['goal']:
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
	@property
	def cur_scenario_idx(self):
		return self.__cur_scenario_idx

	def play_next_scenario(self):
		if self.__cur_scenario_idx == len(self.__scenarios):
			return None
		print(f'\n>>Playing scenario {self.__scenarios[self.__cur_scenario_idx]["name"]}')
		self._arena[1:self._h + 1, 1:self._w + 1] = [Entities.EMPTY.value]

		self.__place_obstacles(self.__scenarios[self.__cur_scenario_idx]['obstacles'])

		if self.__scenarios[self.__cur_scenario_idx]['agent']:
			self._arena[self.__scenarios[self.__cur_scenario_idx]['agent'][1], self.__scenarios[self.__cur_scenario_idx]['agent'][0]] = Entities.AGENT.value
			self._agent.move(self.__scenarios[self.__cur_scenario_idx]['agent'][0], self.__scenarios[self.__cur_scenario_idx]['agent'][1], self.__scenarios[self.__cur_scenario_idx]['agent'][2])
		else:
			self._place_entity_random(Entities.AGENT)

		if self.__scenarios[self.__cur_scenario_idx]['goal']:
			self._arena[self.__scenarios[self.__cur_scenario_idx]['goal'][1], self.__scenarios[self.__cur_scenario_idx]['goal'][0]] = Entities.GOAL.value
			self._goal.move(self.__scenarios[self.__cur_scenario_idx]['goal'][0], self.__scenarios[self.__cur_scenario_idx]['goal'][1])
		else:
			self._place_entity_random(Entities.GOAL)

		self._cur_state(self.make_state())

		self.__cur_scenario_idx += 1

		return self.__scenarios[self.__cur_scenario_idx - 1]['name']

	def reset(self, terminal = False, cur_state = None, next_state = None, action = None, event = None):
		print('>>>Reseting')
		self._arena[:,:] = [Entities.EMPTY.value]
		self.__generate_obstacles(randint(0, self.__num_obstacles))
		self._place_entity_random(Entities.GOAL)
		self._place_entity_random(Entities.AGENT)
		self._cur_state(self.make_state()) # Define initial state
		if 0 in self._cur_state.los:
			while True:
				a_loc = np.where(self._arena == Entities.AGENT.value)
				self._arena[a_loc[0].item()][a_loc[1].item()] = Entities.EMPTY.value
				self._place_entity_random(Entities.AGENT)
				self._cur_state(self.make_state())
				if not 0 in self._cur_state.los:
					break

	def _place_entity_random(self, ent):
		while True:
			r, c = randint(0, self._w - 1), randint(0, self._h - 1)
			if self._arena[r,c] == Entities.EMPTY.value:
				self._arena[r,c] = ent.value
				if ent == Entities.AGENT:
					self._agent.move(c, r, choice(DiscreteAngularSpace.POSSIBLE_ANGLES))
					break
				elif ent == Entities.GOAL:
					self._goal.move(c, r)
					break

	def __generate_obstacles(self, num):
		self._entities[Entities.OBSTACLE].clear()
		for i in range(num):
			while True: 
				r, c = randint(0, self._w - 1), randint(0, self._h - 1)
				if self._arena[r,c] == Entities.EMPTY.value:
					self._arena[r,c] = Entities.OBSTACLE.value
					self._entities[Entities.OBSTACLE].append(EnvEntity(c, r))
					break

	def __place_obstacles(self, positions):
		del self.__obstacles[:]
		for idx,(x,y) in enumerate(zip(positions['x'], positions['y'])):
			self.__obstacles.append(EnvEntity(x,y))
			self._arena[y,x] = Entities.OBSTACLE.value
		self._entities[Entities.OBSTACLE] = self.__obstacles

	def step(self, action):
		# Clean old agent info
		self._arena[self._agent.y,self._agent.x] = Entities.EMPTY.value

		# Predict next pose
		print(f'before pose: {self._agent.x,self._agent.y,self._agent.theta}')
		x, y, ang = self._agent.kinematics(action)
		print(f'after pose: {x,y,ang}')

		if self._out_of_bounds(x,y): # If next pose is out of bounds, return and skip to next decision
			return self._cur_state, False, RewardFunction.UNDEFINED, Entities.VOID.value, (x,y)

		# Check entity for next position
		neighbour = self._arena[y,x]

		# Act		
		if neighbour == Entities.OBSTACLE.value:
			print('event: Collision')
			return neighbour, True, self._reward_function(self._cur_state, action, self._next_state, neighbour), neighbour, (x,y)
		if neighbour == Entities.GOAL.value:
			print('event: GOAL')
			return neighbour, True, self._reward_function(self._cur_state, action, self._next_state, neighbour), neighbour, (x,y)

		self._agent.move(x, y, ang) # Move to predicted pose

		self._next_state(self.make_state()) # Observe after acting

		self._arena[self._agent.y, self._agent.x] = Entities.AGENT.value # Place agent in arena in new position

		if self._next_state.los in DiscreteLineOfSightSpace.TERMINAL_STATES:
			print('event: TERMINAL_STATE')
			return self._next_state, True, self._reward_function(self._cur_state, action, self._next_state, "terminal"), "terminal", (x,y)
		else:
			print('event: normal')
			return self._next_state, False, self._reward_function(self._cur_state, action, self._next_state, None), neighbour, (x,y)

	def __generate_los(self, terminal = False):
		if terminal:
			return self._cur_state.los
		rot = np.array([
			[cos(self._agent.theta), sin(self._agent.theta)],
			[sin(self._agent.theta), -cos(self._agent.theta)]
		])
		translation_agent = np.array([self._agent.x, self._agent.y]).reshape(2,1)

		# Obtain points that the agent is scanning in the agents frame
		if sum(np.isclose(self._agent.theta, AXIS_ANGLES, atol = TOLERANCE_ANGLE_EQUALITY)) == 1:
			los_positions = np.rint(np.dot(rot, self.__los_state_space.vectors) + translation_agent).astype(np.int8)
		else:
			los_positions = np.rint(np.dot(rot, self.__los_state_space.vectors_diag) + translation_agent).astype(np.int8)

		los = []
		for bin_idx in range(self.__los_state_space.num_bins):
			bin_to_scan = []
			for dist_idx in range(self.__los_state_space.range):
				cell_idx = dist_idx * self.__los_state_space.num_bins + bin_idx
				y, x = los_positions[1, cell_idx], los_positions[0, cell_idx]
				bin_to_scan.append(self.__los_state_space.range) if self._out_of_bounds(x,y) else bin_to_scan.append(self._arena[y][x])	
			los.append(self.__distance_to_obstacle(bin_to_scan, self.__los_state_space.range))
		return tuple(los)

	# Returns the index of the first occurence of an obstacle in a line of sight bin, otherwise returns len of the bin
	## e.g. [-1,0], it returns 0; [0,-1] it returns 1; [0,0], it returns 2
	@staticmethod
	def __distance_to_obstacle(los_bin, max_range):
		try: return los_bin.index(Entities.OBSTACLE.value)
		except: return max_range

	def make_state(self, terminal = False):
		return (self._agent.x, self._agent.y, self._agent.theta, self._goal.x, self._goal.y, self.__generate_los(terminal))
