from random import randint
from qlearning import Actions
from enum import Enum
import numpy as np

class Entities(Enum):
	OBSTACLE = -1
	EMPTY = 0
	AGENT = 1
	GOAL = 2

class EnvEntity:
	def __init__(self):
		self.__pos = [0,0]

	@property
	def pos(self):
		return self.__pos

	def set_x(self, x):
		self.__pos[0] = x

	def set_y(self, y):
		self.__pos[1] = y

	def go_left(self):
		self.__pos[0] -= 1

	def go_up(self):
		self.__pos[1] -= 1

	def go_right(self):
		self.__pos[0] += 1

	def go_down(self):
		self.__pos[1] += 1

	@property
	def x(self):
		return self.__pos[0]

	@property
	def y(self):
		return self.__pos[1]
		
	def set_pos(self, x, y):
		self.__pos[0], self.__pos[1] = x, y

class RewardFunction:
	def __init__(self):
		pass
	def __call__(self, event = None):
		if event is None:
			return 0
		elif event == Entities.OBSTACLE.value:
			print('COLLIDED')
			return -1
		else:
			print('GOAL')
			return 1

class Environment:
	def __init__(self, w, h, num_obstacles):
		self.__w = w
		self.__h = h
		if num_obstacles >= w * h:
			raise ValueError('Number of obstacles needs to be lower than width * height')
		self.__num_obstacles = num_obstacles
		self.__arena = np.zeros((self.__w+2,self.__h+2))
		self.__arena[:,0] = Entities.OBSTACLE.value
		self.__arena[0,1:] = Entities.OBSTACLE.value
		self.__arena[1:,-1] = Entities.OBSTACLE.value
		self.__arena[-1,1:self.__w+1] = Entities.OBSTACLE.value

		self.__agent = EnvEntity()
		self.__goal = EnvEntity()
		self.__obstacles = []
		for i in range(self.__num_obstacles):
			self.__obstacles.append(EnvEntity())

		self.__entities = {
			Entities.AGENT: self.__agent,
			Entities.OBSTACLE: self.__obstacles,
			Entities.GOAL: self.__goal
		}
		self.__reward_function = RewardFunction()
		self.reset()

	def __repr__(self):
		return str(self.__arena)

	def reset(self, event = None):
		for row in range(1,self.__h + 1):
			self.__arena[row, 1:self.__w + 1] = [Entities.EMPTY.value]
		self.__generate_obstacles(self.__num_obstacles)
		self.__place_entity_random(Entities.AGENT)
		self.__place_entity_random(Entities.GOAL)
		return self.observe(), True, self.__reward_function(event)

	def __generate_obstacles(self, num):
		for i in range(num):
			while True: 
				r, c = randint(1, self.__w), randint(1, self.__h)
				if self.__arena[r,c] == Entities.EMPTY.value:
					self.__arena[r,c] = Entities.OBSTACLE.value
					self.__entities[Entities.OBSTACLE][i].set_pos(c, r)
					break

	def __place_entity_random(self, ent):
		while True:
			r, c = randint(1, self.__w), randint(1, self.__h)
			if self.__arena[r,c] == Entities.EMPTY.value:
				if ent == Entities.AGENT:
					self.__arena[r,c] = Entities.AGENT.value
					self.__agent.set_pos(c, r)
					break
				elif ent == Entities.GOAL:
					self.__arena[r,c] = Entities.GOAL.value
					self.__goal.set_pos(c, r)
					break

	def step(self, action):
		self.__arena[self.__agent.y,self.__agent.x] = Entities.EMPTY.value

		if action == Actions.LEFT:
			neighbour = self.__arena[self.__agent.y, self.__agent.x - 1]
			if neighbour != Entities.EMPTY.value:
				return self.reset(neighbour)
			else:
				self.__agent.go_left()

		elif action == Actions.UP:
			neighbour = self.__arena[self.__agent.y - 1, self.__agent.x]
			if neighbour != Entities.EMPTY.value:
				return self.reset(neighbour)
			else:
				self.__agent.go_up()

		elif action == Actions.RIGHT:
			neighbour = self.__arena[self.__agent.y, self.__agent.x + 1]
			if neighbour != Entities.EMPTY.value:
				return self.reset(neighbour)
			else:
				self.__agent.go_right()

		elif action == Actions.DOWN:
			neighbour = self.__arena[self.__agent.y + 1, self.__agent.x]
			if neighbour != Entities.EMPTY.value:
				return self.reset(neighbour)
			else:
				self.__agent.go_down()
				
		self.__arena[self.__agent.y, self.__agent.x] = Entities.AGENT.value
		return self.observe(), False, self.__reward_function()

	def __generate_los(self):
		# [left, up, right, down]
		return [
			self.__arena[self.__agent.y][self.__agent.x - 1],
			self.__arena[self.__agent.y - 1][self.__agent.x],
			self.__arena[self.__agent.y][self.__agent.x + 1],
			self.__arena[self.__agent.y + 1][self.__agent.x],
		]
	
	def observe(self):
		return (
			self.__agent.x,
			self.__agent.y,
			self.__goal.x,
			self.__goal.y,
			self.__generate_los()
			)
