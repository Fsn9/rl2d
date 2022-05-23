from enum import Enum
from math import atan2
from numpy import linspace, pi
from random import random, choice
from environment import Actions, DiscreteAngularSpace, DiscreteLineOfSightSpace, StateSimple, StateComplex, Entities

class StateAction:
	def __init__(self, state, action, initial_alpha, gamma):
		self.__state = state
		self.__action = action
		self.__q = 0
		self.__visited = False
		self.__initial_alpha = initial_alpha
		self.__gamma = gamma
		self.__num_visits = 0
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
	@property
	def visited(self):
		return self.__visited
	@property
	def num_visits(self):
		return self.__num_visits
	def update_q(self, reward, max_q, terminal):
		self.__num_visits += 1
		if not self.__visited:
			self.__visited = True
		if terminal:
			self.__q = self.__q + self.__initial_alpha / self.__num_visits * (reward - self.__q)
		else:
			self.__q = self.__q + self.__initial_alpha / self.__num_visits * (reward + self.__gamma * max_q - self.__q)

class QTable:
	def __init__(self, env_width, env_height, alpha, gamma):
		self.__table = []
		self.__alpha = alpha
		self.__gamma = gamma
		self.__state_space = DiscreteAngularSpace(env_width, env_height)
		for state in self.__state_space():
			for action in Actions:
				self.__table.append(StateAction(state, action, self.__alpha, self.__gamma))

	def __repr__(self):
		repr_ = ""
		for sa in self.__table:
			repr_ += str(sa)
		return repr_
	
	@property
	def table(self):
		return self.__table

	def get_greedy_action(self, state):
		best_q = -10000
		action = None
		for sa in self.__table:
			if sa.state == state:
				if sa.q > best_q:
					best_q = sa.q
					action = sa.action
		if action is None:
			raise Exception('State not found: ', state)
		return action

	def get_sa(self, state, action):
		for sa in self.__table:
			if sa.state == state and sa.action == action:
				return sa
		raise Exception('(s,a) pair not found in table: ', state, action)

	def get_q(self, state, action):
		for sa in self.__table:
			if sa.state == state and sa.action == action:
				return sa.q
		raise Exception('(s,a) pair not found in table: ', state, action)

class QLearner:
	def __init__(self, 
    observation_type, 
	action_type,
    learning_rate, 
    discount_factor, 
    episodes,
	initial_epsilon,
	final_epsilon,
	environment
    ):
		self.__observation_type = observation_type
		self.__action_type = action_type
		self.__alpha = learning_rate
		self.__gamma = discount_factor
		self.__episodes = episodes
		self.__episodes_passed = 0
		self.__initial_epsilon = initial_epsilon
		self.__epsilon = self.__initial_epsilon
		self.__final_epsilon = final_epsilon
		self.__action_space = list(Actions)
		self.__environment = environment
		self.__qtable = QTable(self.__environment.w, self.__environment.h, self.__alpha, self.__gamma)
		self.__cur_state, self.__next_state = [], []

	def decide(self, state):
		if random() > self.__epsilon:
			print('GREEDY ', self.__epsilon)
			return self.__qtable.get_greedy_action(state)
		else:
			print('RANDOM', self.__epsilon)
			return choice(self.__action_space)

	# q(s,a) = q(s,a) + alpha * (r + gamma * max_q(s',a') - q(s,a))
	def learn(self, cur_state, next_state, action, reward, terminal, neighbour):
		self.__cur_state, self.__next_state = cur_state, next_state
		if terminal:
			self.__episodes_passed += 1
			self.__epsilon = max(self.__final_epsilon, 
				self.__episodes_passed * (self.__final_epsilon - self.__initial_epsilon) / self.__episodes + self.__initial_epsilon)
		
		max_q = self.__qtable.get_q(next_state, self.__qtable.get_greedy_action(next_state))
		sa = self.__qtable.get_sa(cur_state, action)
		sa.update_q(reward, max_q, terminal)

		#print('s,a,ns,r,q: ', cur_state, action, next_state,reward, sa.q)
		#print('num_visits: ', sa.num_visits)

	def print_result(self):
		visited_sa_pairs = 0
		for sa in self.__qtable.table:
			if sa.visited:
				visited_sa_pairs += 1
		print('Visited sa pairs: ', visited_sa_pairs, '/', len(self.__qtable.table))
		print('Visited percentage (%): ', 100 * visited_sa_pairs / len(self.__qtable.table))

