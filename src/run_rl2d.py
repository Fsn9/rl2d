from qlearning import *
from environment import *
from graphics import GUI
import sys

# Read args
if len(sys.argv) == 1:
	env_dim, action_complexity = 5, 'simple'
elif len(sys.argv) == 2:
	env_dim, action_complexity = int(sys.argv[1]), 'simple'
elif len(sys.argv) == 3:
	env_dim, action_complexity = int(sys.argv[1]), sys.argv[2]

# Empty environment
env = EmptyEnvironment(env_dim, env_dim, action_complexity)
#env = ObstacleEnvironment(3,3,2,'complex','.')
print(env)

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
episodes = 2000
initial_epsilon = 1
final_epsilon = 0.05

# Agent
agent = QLearner(learning_rate, discount_factor, episodes, initial_epsilon, final_epsilon, env)

# Graphics
gui = GUI(agent, env)

gui.mainloop()