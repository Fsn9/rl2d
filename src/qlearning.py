from enum import Enum
from math import atan2

class StateSimple:
	def __init__(self):
		self.__azimuth = 0.0
	def set(self, x, y, xg, yg):
		self.__azimuth = atan2(yg - y, xg - x)

class StateComplex:
	def set(self):
		pass

class Actions(Enum):
	LEFT = 0
	UP = 1
	RIGHT = 2
	DOWN = 3

class QTable:
	def __init__(self):
		pass

class QLearner:
	def __init__(self, observation_type, 
	learning_rate, 
	discount_factor,
	episodes,
	initial_epsilon,
	final_epsilon
	):
		self.alpha = learning_rate
		self.gamma = discount_factor
		self.episodes = episodes
		self.initial_epsilon = initial_epsilon
		self.final_epsilon = final_epsilon
