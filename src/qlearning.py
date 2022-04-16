from enum import Enum
from math import atan2
from numpy import linspace, pi
from random import random, choice

class StateSimple:
	def __init__(self):
		self.__azimuth = 0.0
	def __call__(self, x, y, xg, yg):
		self.__azimuth = atan2(yg - y, xg - x)
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

class QTable:
	def __init__(self):
		self.__table = []
		angular_resolution_2d = 9
		state_space = 180/pi * (linspace(-pi, pi, angular_resolution_2d - 1, endpoint = False) + pi)
		for state in state_space:
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
			self.__cur_state = StateSimple()
		else:
			self.__cur_state = StateComplex()

	def decide(self, observation):
		x, y, xg, yg, los = observation
		if random() > self.__epsilon:
			return self.__qtable.get_greedy_action(self.__cur_state(x,y,xg,yg))
		else:
			return choice(self.__action_space)
