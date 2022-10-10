from enum import Enum
from math import atan2
from numpy import linspace, pi
from random import random, choice
from environment import *
from collections import deque
import datetime, time
import os
import pickle

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
		return "("+str(self.__state)+","+str(self.__action)+"), q: "+str(self.__q)+", nv: "+str(self.num_visits)
	def __str__(self):
		return "("+str(self.__state)+","+str(self.__action)+"), q: "+str(self.__q)+", nv: "+str(self.num_visits)
	def __eq__(self, other):
		return self.__state.__eq__(other.state) and self.__action == other.action
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
	def __init__(self, environment, alpha, gamma, state_space, qtable_path):
		self.__alpha = alpha
		self.__gamma = gamma
		self.__state_space = state_space
		self.__environment = environment

		if qtable_path:
			path_aux = qtable_path.replace('.','-')
			splits = path_aux.split('-')
			date_id = splits[1]
			with open(os.path.join('runs', 'run-' + date_id, qtable_path), 'rb') as q_table_object_file:
				self.__table = pickle.load(q_table_object_file)
		else:
			self.__table = []
			for idx, state in enumerate(self.__state_space()):
				if isinstance(self.__environment, ObstacleEnvironment):
					# Remove states where los has goal and does not correspond to 0, -45 or 45 in azimuth
					if self.__environment.los_type == '|':
						if state.los[0] == Entities.GOAL.value and not state.azimuth == 0.0 or \
						state.los[1] == Entities.GOAL.value and not state.azimuth == 0.0:
							continue
					elif self.__environment.los_type == '-':
						if state.los[0] == Entities.GOAL.value and not state.azimuth == 0.785 or \
						state.los[1] == Entities.GOAL.value and not state.azimuth == 0.0 or \
						state.los[2] == Entities.GOAL.value and not state.azimuth == -0.785:
							continue
					elif self.__environment.los_type == 'T':
						if state.los[0] == Entities.GOAL.value and not state.azimuth == 0.0 or \
						state.los[1] == Entities.GOAL.value and not state.azimuth == 0.464 or \
						state.los[2] == Entities.GOAL.value and not state.azimuth == 0.0 or \
						state.los[3] == Entities.GOAL.value and not state.azimuth == -0.464:
							continue
				for action in Actions:
					self.__table.append(StateAction(state, action, self.__alpha, self.__gamma))

		print(f'>>Q table of length {len(self)} was created.')
	def __repr__(self):
		repr_ = ""
		for sa in self.__table:
			repr_ += str(sa) + '\n'
		return repr_
	def __str__(self):
		return self.__repr__()
	def __len__(self):
		return len(self.__table)
	
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
    learning_rate, 
    discount_factor, 
    episodes,
	initial_epsilon,
	final_epsilon,
	environment,
	qtable_path,
	evaluation
    ):
		self.__alpha = learning_rate 
		self.__gamma = discount_factor
		self.__current_gamma = self.__gamma
		self.__episodes = episodes
		self.__episodes_passed = 0
		self.__episodes_left = self.__episodes
		self.__initial_epsilon = initial_epsilon
		self.__current_epsilon = self.__initial_epsilon
		self.__final_epsilon = final_epsilon
		self.__environment = environment
		self.__qtable_path = qtable_path
		self.__evaluation = evaluation
		self.__qtable = QTable(self.__environment, self.__alpha, self.__gamma, self.__environment.state_space, self.__qtable_path)
		self.__action_space = list(Actions)
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

		## Episode ending
		self.__ending_causes = [] # per epoch
		self.__window_size_ending_causes_moving_avg = 50 # @TODO: put this in yaml or json
		self.__window_ending_causes = deque(maxlen = self.__window_size_ending_causes_moving_avg)

		## Auxiliar variables
		self.__cur_state = []
		self.__next_state = []

		## Gui variables
		self.__user_interaction_mode = "Auto"
		self.__action_mode = "Random"

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
	@property
	def window_size_ending_causes_moving_avg(self):
		return self.__window_size_ending_causes_moving_avg
	@property
	def ending_causes(self):
		return self.__ending_causes
	@property
	def user_interaction_mode(self):
		return self.__user_interaction_mode
	@user_interaction_mode.setter
	def user_interaction_mode(self, mode):
		self.__user_interaction_mode = mode
	@property
	def action_mode(self):
		return self.__action_mode
	@action_mode.setter
	def action_mode(self, mode):
		self.__action_mode = mode
	@property
	def qtable_path(self):
		return self.__qtable_path
	@property
	def evaluation(self):
		return self.__evaluation

	def act(self):
		# 1. Observe
		obs = self.__environment.observe()

		# 2. Decide
		action = self.decide(obs)
		print('[act], cur_state, action:', obs, action)

		# 3. Act and observe again
		next_obs, terminal, reward, neighbour = self.__environment.step(action)

		# 4. If invalid action skip to next step
		if neighbour == Entities.VOID.value:
			print('[act] neighbour is VOID')
			return False
		print('[act] learning -> next, action, reward, terminal, neighbour:', next_obs, action, reward, terminal, neighbour)

		# 5. Learn
		self.learn(obs, next_obs, action, reward, terminal, neighbour)

		# If terminated, reset the environment
		if terminal:
			self.__environment.reset()
			return

		# 6. Save next observation
		obs.get_data_from(next_obs)
		print(f'[act] Quitting. terminal: {terminal}\n')
		return terminal

	def act_eval(self):
		# 1. Observe
		obs = self.__environment.observe()

		# 2. Decide
		action = self.decide_greedy(obs)
		print('[act], cur_state, action:', obs, action)

		# 3. Act and observe again
		next_obs, terminal, reward, neighbour = self.__environment.step(action)

		# 4. If invalid action skip to next step
		if neighbour == Entities.VOID.value:
			print('[act] neighbour is VOID')
			return False
		print('[act] learning -> next, action, reward, terminal, neighbour:', next_obs, action, reward, terminal, neighbour)

		# If terminated, reset the environment
		if terminal:
			return self.__environment.play_next_scenario()

		# 6. Save next observation
		obs.get_data_from(next_obs)
		print(f'[act] Quitting. terminal: {terminal}\n')
		return terminal

	def decide(self, state):
		self.__current_steps_sum += 1
		if self.__qtable_path:
			print('action: GREEDY')
			return self.__qtable.get_greedy_action(state)
		if self.__user_interaction_mode == "Auto":
			if random() > self.__current_epsilon:
				print('action: GREEDY')
				return self.__qtable.get_greedy_action(state)
			else:
				print('action: RANDOM')
				return choice(self.__action_space)
		elif self.__user_interaction_mode == "Manual":
			if self.__action_mode == "Greedy":
				print('action: GREEDY')
				return self.__qtable.get_greedy_action(state)
			else:
				print('action: RANDOM')
				return choice(self.__action_space)
	def decide_greedy(self, state):
		print('action: GREEDY')
		return self.__qtable.get_greedy_action(state)

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
			self.__window_ending_causes.appendleft(1) if neighbour == Entities.GOAL.value else self.__window_ending_causes.appendleft(0)
			if self.__episodes_passed % self.__window_size_ending_causes_moving_avg == 0:
				self.__ending_causes.append(sum(self.__window_ending_causes) / len(self.__window_ending_causes))
			print(f'>> Episodes left: {self.__episodes_left}')
			if self.__episodes_passed == self.__episodes:
				self.__finished = True
			max_q = RewardFunction.UNDEFINED
		else:
			max_q = self.__qtable.get_q(next_state, self.__qtable.get_greedy_action(next_state))
		sa = self.__qtable.get_sa(cur_state, action)
		print(f'[learn] {sa}')
		sa.update_q(reward, max_q, terminal)

	def export_results(self):
		# Paths
		parent_path = os.path.join(os.getcwd(), "runs")
		time_string = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M_%S")
		folder_name = "run-" + time_string
		folder_path = os.path.join(parent_path, folder_name)

		# Folder
		os.mkdir(folder_path)

		# Files
		qtable_file = open(os.path.join(folder_path, "table-" + time_string + ".txt"), "w")
		qtable_file.write(str(self.__qtable))

		visited_sa_pairs = 0
		for sa in self.__qtable.table:
			if sa.visited:
				visited_sa_pairs += 1
		metrics_str = 'Visited sa pairs: ' + str(visited_sa_pairs) + ' / ' + str(len(self.__qtable.table)) + '\n'
		metrics_str += 'Visited percentage (%): ' + str(100 * visited_sa_pairs / len(self.__qtable.table)) + '\n'
		metrics_str += 'Moving average reward: ' + str(self.__mov_avg_reward) + '\n'
		metrics_str += 'Moving average steps: ' + str(self.__mov_avg_steps)
		metrics_file = open(os.path.join(folder_path, "metrics-" + time_string + ".txt"), "w")
		metrics_file.write(metrics_str)
		print('Final results:\n', metrics_str, sep = '')

		env_data_str = "width: " + str(self.__environment.w) + '\n'
		env_data_str += "height: " + str(self.__environment.h) + '\n'
		env_data_str += "state space size: " + str(len(self.__environment.state_space)) + '\n'
		env_data_str += "state space raw representation: " + str(self.__environment.state_space.space) + '\n'
		env_data_str += "action space: " + str(self.__action_space)
		env_data_file = open(os.path.join(folder_path, "env_data-" + time_string + ".txt"), "w")
		env_data_file.write(env_data_str)

		hyperparameters_str = "learning rate: " + str(self.__alpha) + '\n'
		hyperparameters_str += "discount factor: " + str(self.__gamma) + '\n'
		hyperparameters_str += "episodes: " + str(self.__episodes) + '\n'
		hyperparameters_str += "initial epsilon: " + str(self.__initial_epsilon) + '\n'
		hyperparameters_str += "final epsilon: " + str(self.__final_epsilon)
		qlearning_hp_file = open(os.path.join(folder_path, "hyperparameters-" + time_string + ".txt"), "w")
		qlearning_hp_file.write(hyperparameters_str)

		with open(os.path.join(folder_path, "table-" + time_string + ".pkl"), 'wb') as q_table_object_file:
			pickle.dump(self.__qtable.table, q_table_object_file)

		return folder_path, time_string

	def get_stats(self):
		return self.__mov_avg_steps, \
			self.__current_gamma, \
			self.__current_epsilon, \
			self.__mov_avg_reward, \
			self.__episodes_left
