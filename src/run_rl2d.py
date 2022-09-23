from qlearning import *
from environment import *
from graphics import GUI
import sys

# Read args
if len(sys.argv) == 1:
	env_dim, action_complexity, episodes = 5, 'simple', 4000
elif len(sys.argv) == 2:
	env_dim, action_complexity, episodes = int(sys.argv[1]), 'simple', 4000
elif len(sys.argv) == 3:
	env_dim, action_complexity, episodes = int(sys.argv[1]), sys.argv[2], 4000
elif len(sys.argv) == 4:
	env_dim, action_complexity, episodes = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])

if env_dim < 3 or env_dim > 9:
	raise ValueError("The dimension of the environment needs to be between 2 < x < 10")

# Empty environment
env = EmptyEnvironment(env_dim, env_dim, action_complexity)
#env = ObstacleEnvironment(env_dim, env_dim, 2, 'complex','.')

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
initial_epsilon = 1
final_epsilon = 0.05

# Agent
agent = QLearner(learning_rate, discount_factor, episodes, initial_epsilon, final_epsilon, env)

# Graphics
gui = GUI(agent, env)

# Start GUI thread
gui.mainloop()