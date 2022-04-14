import numpy as np
from enum import Enum
from random import randint

class StateNav:
	pass
class StateNavCol:
	pass
class Actions(Enum):
	LEFT = 0
	UP = 1
	RIGHT = 2
	DOWN = 3

class Entities(Enum):
	OBSTACLE = -1
	EMPTY = 0
	AGENT = 1
	GOAL = 2


class Agent:
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

class Environment:
	def __init__(self, w, h):
		self.__w = w
		self.__h = h
		self.__arena = np.zeros((self.__w+2,self.__h+2))
		self.__arena[:,0] = Entities.OBSTACLE.value
		self.__arena[0,1:] = Entities.OBSTACLE.value
		self.__arena[1:,-1] = Entities.OBSTACLE.value
		self.__arena[-1,1:self.__w+1] = Entities.OBSTACLE.value
		self.__agent = Agent()
		self.reset()

	def __repr__(self):
		return str(self.__arena)

	def reset(self):
		for row in range(1,self.__h-1):
			self.__arena[row, 1:self.__w-1] = Entities.EMPTY.value
		r, c = randint(1, self.__w), randint(1, self.__h)
		self.__set_value_arena(r, c, Entities.AGENT.value)
		self.__agent.set_pos(c, r)

	def __collided(self, x, y):
		return self.__arena[x][y] == Entities.OBSTACLE.value

	def __set_value_arena(self, x, y, val):
		self.__arena[x,y] = val

	def step(self, action):
		self.__set_value_arena(self.__agent.y, self.__agent.x, Entities.EMPTY.value)
		if action == Actions.LEFT:
			print('left')
			if not self.__collided(self.__agent.y - 1, self.__agent.x - 1):
				self.__agent.go_left()
			else:
				print('collided')
				self.reset()
				return
		elif action == Actions.UP:
			print('up')
			if not self.__collided(self.__agent.y - 1, self.__agent.x):
				self.__agent.go_up()
			else:
				print('collided')
				self.reset()
				return
		elif action == Actions.RIGHT:
			print('right')
			if not self.__collided(self.__agent.y, self.__agent.x + 1):
				self.__agent.go_right()
			else:
				print('collided')
				self.reset()
				return
		elif action == Actions.DOWN:
			print('down')
			if not self.__collided(self.__agent.y + 1, self.__agent.x):
				self.__agent.go_down()
			else:
				print('collided')
				self.reset()
				return
		print('next position: ', self.__agent.pos)
		self.__set_value_arena(self.__agent.y, self.__agent.x, Entities.AGENT.value)

env = Environment(5,5)
print(env)
env.step(Actions.RIGHT)
print()
print(env)
