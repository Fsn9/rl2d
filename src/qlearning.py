from enum import Enum
from math import atan2
from xml.dom import NotFoundErr
from numpy import linspace, pi
from random import random, choice

class StateSimple:
	def __init__(self):
		self.__azimuth = 0.0
	def __call__(self, obs):
		self.__azimuth = atan2(obs[3] - obs[1], obs[2] - obs[0])
		return self.__azimuth
	def get(self):
		return self.__azimuth

class StateComplex:
	pass

class Actions(Enum):
	LEFT = 0
	UP = 1
	RIGHT = 2
	DOWN = 3
class ActionsComplex(Enum):
	LEFT = 0
	LEFT_UP = 1
	UP = 2
	UP_RIGHT = 3
	RIGHT = 4
	RIGHT_DOWN = 5
	DOWN = 6
	DOWN_LEFT = 7

class DiscreteAngularSpace:
	def __init__(self, width, height):
		self.__width = width
		self.__height = height
		self.__space = []
		self.__generate_space()
	def __call__(self):
		return self.__space
	def __generate_space(self):
		for y in range(self.__height):
			for x in range(self.__width):
				self.__space.append(atan2(y,x))
				self.__space.append(atan2(-y,x))
				self.__space.append(atan2(-y,-x))
				self.__space.append(atan2(y,-x))
		self.__space = list(dict.fromkeys(self.__space))
		self.__space.sort()

class StateAction:
	def __init__(self, state, action):
		self.__state = state
		self.__action = action
		self.__q = 0
	def __repr__(self):
		return "("+str(self.__state)+","+str(self.__action)+")\n"
	def __str__(self):
		return "("+str(self.__state)+","+str(self.__action)+")\n"
	def __eq__(self, other):
		return self.__state == other.state and self.__action == other.action
	@property
	def state(self):
		return self.__state
	@property
	def action(self):
		return self.__action
	@property
	def q(self):
		return self.__q
	@q.setter
	def q(self, val):
		self.__q = val

class QTable:
	def __init__(self):
		self.__table = []
		angular_resolution_2d = 9
		self.__state_space = DiscreteAngularSpace(5,5)
		for state in self.__state_space():
			for action in Actions:
				self.__table.append(StateAction(state, action))

	def __repr__(self):
		repr_ = ""
		for sa in self.__table:
			repr_ += str(sa)
		return repr_

	def get_greedy_action(self, state):
		best_q = -10000
		action = []
		for sa in self.__table:
			if sa.state == state:
				if sa.q > best_q:
					best_q = sa.q
					action = sa.action
		return action
	def get_q(self, state, action):
		for sa in self.__table:
			if sa.state == state and sa.action == action:
				return sa.q
		raise NotFoundErr('(s,a) pair not found in table')
	def set_q(self, state, action, q):
		for sa in self.__table:
			if sa.state == state and sa.action == action:
				sa.q = q
				return
		raise NotFoundErr('(s,a) pair not found in table')

class QLearner:
	def __init__(self, 
    observation_type, 
	action_type,
    learning_rate, 
    discount_factor, 
    episodes,
	initial_epsilon,
	final_epsilon
    ):
		self.__observation_type = observation_type
		self.__action_type = action_type
		self.__alpha = learning_rate
		self.__gamma = discount_factor
		self.__episodes = episodes
		self.__initial_epsilon = initial_epsilon
		self.__epsilon = self.__initial_epsilon
		self.__final_epsilon = final_epsilon
		self.__action_space = list(Actions)
		self.__qtable = QTable()
		
		if self.__observation_type == "simple":
			self.__cur_state, self.__next_state = StateSimple(), StateSimple()
		else:
			self.__cur_state, self.__next_state = StateComplex(), StateComplex()

	def decide(self, observation):
		self.__cur_state(observation)
		x, y, xg, yg, los = observation
		if random() > self.__epsilon:
			return self.__qtable.get_greedy_action(self.__cur_state(x,y,xg,yg))
		else:
			return choice(self.__action_space)
	# q(s,a) = q(s,a) + alpha * (r + gamma * max_q(s',a') - q(s,a))
	def learn(self, cur_obs, next_obs, action, reward, terminal):
		cur_state, next_state = self.__cur_state.get(), self.__next_state(next_obs)
		print('s,a', cur_state, action)
		q = self.__qtable.get_q(cur_state, action)
		greedy_action = self.__qtable.get_greedy_action(next_state)
		max_q = self.__qtable.get_q(next_state, greedy_action)
		q = q + self.__alpha * (reward + self.__gamma * max_q - q)
		self.__qtable.set_q(cur_state, action, q)
