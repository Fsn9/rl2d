import numpy as np
from qlearning import *
from environment import *

env = Environment(5,5,3)

print(env)

env.step(Actions.RIGHT)
print(env)

env.step(Actions.DOWN)
print(env)

env.step(Actions.LEFT)
print(env)

env.step(Actions.UP)
print(env)