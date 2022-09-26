from qlearning import *
from environment import *
from graphics import GUI
import sys

# Read args
if len(sys.argv) == 1:
	env_type, env_dim, action_type, episodes = 'empty', 5, 'simple', 4000
elif len(sys.argv) == 2:
	env_type, env_dim, action_type, episodes = sys.argv[1], 5, 'simple', 4000
elif len(sys.argv) == 3:
	env_type, env_dim, action_type, episodes = sys.argv[1], int(sys.argv[2]), 'simple', 4000
elif len(sys.argv) == 4:
	env_type, env_dim, action_type, episodes = sys.argv[1], int(sys.argv[2]), sys.argv[3], 4000
elif len(sys.argv) == 5:
	env_type, env_dim, action_type, episodes = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])

if env_dim < 3 or env_dim > 9:
	raise ValueError("The dimension of the environment needs to be between 2 < x < 10")

num_obstacles = 2
los_type = '-'

# Empty environment
if env_type == 'empty':
	env = EmptyEnvironment(env_dim, env_dim, action_type)
else:
	env = ObstacleEnvironment(env_dim, env_dim, num_obstacles, action_type, los_type)

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