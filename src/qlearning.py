from enum import Enum
from math import atan2
from numpy import linspace, pi
from random import random, choice
from environment import Actions, DiscreteAngularSpace, DiscreteLineOfSightSpace, StateSimple, StateComplex, Entities
from collections import deque

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
		self.__action_type = action_type
		self.__alpha = learning_rate
		self.__gamma = discount_factor
		self.__current_gamma = self.__gamma
		self.__episodes = episodes
		self.__episodes_passed = 0
		self.__episodes_left = self.__episodes
		self.__initial_epsilon = initial_epsilon
		self.__current_epsilon = self.__initial_epsilon
		self.__final_epsilon = final_epsilon
		self.__action_space = list(Actions)
		self.__environment = environment
		self.__qtable = QTable(self.__environment.w, self.__environment.h, self.__alpha, self.__gamma)
		self.__cur_state, self.__next_state = [], []
		self.__finished = False

		# Stats
		## Reward
		self.__current_reward_sum = 0
		self.__mov_avgs_reward = []
		self.__window_size_reward_moving_avg = 50 # @TODO: put this in yaml or json
		self.__episodic_reward_sums = deque(maxlen = self.__window_size_reward_moving_avg)
		self.__mov_avg_reward = 0
		
		## Steps
		self.__current_steps_sum = 0
		self.__mov_avgs_steps = []
		self.__window_size_steps_moving_avg = 50 # @TODO: put this in yaml or json
		self.__episodic_steps_sums = deque(maxlen = self.__window_size_steps_moving_avg)
		self.__mov_avg_steps = 0

	@property
	def alpha(self):
		return self.__alpha
	@property
	def gamma(self):
		return self.__gamma
	@property
	def current_gamma(self):
		return self.__current_gamma
	@property
	def episodes(self):
		return self.__episodes
	@property
	def episodes_passed(self):
		return self.__episodes_passed
	@property
	def episodes_left(self):
		return self.__episodes_left
	@property
	def current_epsilon(self):
		return self.__current_epsilon
	@property
	def initial_epsilon(self):
		return self.__initial_epsilon
	@property
	def final_epsilon(self):
		return self.__final_epsilon
	@property
	def mov_avg_reward(self):
		return self.__mov_avg_reward
	@property
	def mov_avg_steps(self):
		return self.__mov_avg_steps
	@property
	def mov_avgs_reward(self):
		return self.__mov_avgs_reward
	@property
	def mov_avgs_steps(self):
		return self.__mov_avgs_steps
	@property
	def finished(self):
		return self.__finished
	@property
	def episodic_reward_sums(self):
		return self.__episodic_reward_sums
	@property
	def episodic_steps_sums(self):
		return self.__episodic_steps_sums	

	def act(self):
		# 1. Observe
		obs, _, _, _ = self.__environment.observe()

		# 2. Decide
		action = self.decide(obs)

		# 3. Act and observe again
		next_obs, terminal, reward, neighbour = self.__environment.step(action)

		# 4. If invalid action skip to next step
		if neighbour == Entities.VOID.value:
			return

		# 5. Learn
		self.learn(obs, next_obs, action, reward, terminal, neighbour)

		# 6. If terminal state then reset environment
		if terminal:
			self.__environment.reset()
		return terminal

	def decide(self, state):
		self.__current_steps_sum += 1
		if random() > self.__current_epsilon:
			print('GREEDY')
			return self.__qtable.get_greedy_action(state)
		else:
			print('RANDOM')
			return choice(self.__action_space)

	def learn(self, cur_state, next_state, action, reward, terminal, neighbour):
		self.__cur_state, self.__next_state = cur_state, next_state
		self.__current_reward_sum += reward
		if terminal:
			self.__episodes_passed += 1
			self.__episodes_left -= 1
			self.__current_epsilon = max(self.__final_epsilon, 
				self.__episodes_passed * (self.__final_epsilon - self.__initial_epsilon) / self.__episodes + self.__initial_epsilon)
			self.__episodic_reward_sums.appendleft(self.__current_reward_sum)
			self.__episodic_steps_sums.appendleft(self.__current_steps_sum)
			self.__mov_avg_reward = sum(self.__episodic_reward_sums) / len(self.__episodic_reward_sums)
			self.__mov_avg_steps = sum(self.__episodic_steps_sums) / len(self.__episodic_steps_sums)
			self.__mov_avgs_reward.append(self.__mov_avg_reward)
			self.__mov_avgs_steps.append(self.__mov_avg_steps)
			self.__current_reward_sum, self.__current_steps_sum = 0, 0
			if self.__episodes_passed == self.__episodes:
				self.__finished = True

		max_q = self.__qtable.get_q(next_state, self.__qtable.get_greedy_action(next_state))
		sa = self.__qtable.get_sa(cur_state, action)
		sa.update_q(reward, max_q, terminal)

	def print_result(self):
		visited_sa_pairs = 0
		for sa in self.__qtable.table:
			if sa.visited:
				visited_sa_pairs += 1
		print('Visited sa pairs: ', visited_sa_pairs, '/', len(self.__qtable.table))
		print('Visited percentage (%): ', 100 * visited_sa_pairs / len(self.__qtable.table))
		print('Moving average reward: ', self.__mov_avg_reward)
		print('Moving average steps: ', self.__mov_avg_steps)

	def get_stats(self):
		return self.__mov_avg_steps, \
			self.__current_gamma, \
			self.__current_epsilon, \
			self.__mov_avg_reward, \
			self.__episodes_left
