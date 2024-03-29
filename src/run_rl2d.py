from qlearning import *
from environment import *
from graphics import GUI
import sys
import argparse

# Initialize args parser
parser = argparse.ArgumentParser(description="rl2d obstacle avoidance simulation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## Environment and action space args
parser.add_argument('--env_type', type=str, default='empty', help='The type of the environment. Can be empty or obstacle')
parser.add_argument('--env_dim', type=int, default=5, help='The environment dimension. It needs to be > 2 and < 10')
parser.add_argument('--num_obstacles', type=int, default=3, help='The number of obstacles in the environment')

## Qlearning args
parser.add_argument('--learning_rate', type=float, default=0.1, help='The learning rate > 0 and <= 1')
parser.add_argument('--discount_factor', type=float, default=0.99, \
	help='The discount factor >= and <= 1. For a value of 1 we have a long-term view agent. For a value of 0 we have a myopic agent.')
parser.add_argument('--episodes', type=int, default=2000, help='The number of learning episodes')
parser.add_argument('--initial_epsilon', type=float, default=1, \
	help='The initial epsilon is the exploration probability in the beggining of the learning process.\
	A value of 1 means a total random agent. A value of 0 is a total greedy agent.')
parser.add_argument('--final_epsilon', type=float, default=0.05, help='The value of the final exploration probability.')
parser.add_argument('--qtable_path', type=str, default="", help='The path of a trained Qtable .pkl file. \
	The algorithm will then run in evaluation/testing mode. Provide the qtable .pkl file name (e.g.,: --qtable_path table-2022_10_04_18_36_02.pkl)')
parser.add_argument('--evaluation', action='store_true')

# Parse args
args = parser.parse_args()
config = vars(args)

# Handle environment dimension
if config['env_dim'] < 3 or config['env_dim'] > 9:
	raise ValueError("The dimension of the environment needs to be between 2 < x < 10")

# Create environment
env_types = ["empty", "obstacle"]
if config['env_type'] not in env_types:
	raise ValueError(f"Invalid env type {config['env_type']}. --env_type should be one of these types: {str(env_types)}")

if config['env_type'] == 'empty':
	env = EmptyEnvironment(config['env_dim'], config['env_dim'], config['evaluation'])
else:
	env = ObstacleEnvironment(config['env_dim'], config['env_dim'], config['num_obstacles'], config['evaluation'])

# Agent
agent = QLearner(config['learning_rate'], config['discount_factor'], config['episodes'], config['initial_epsilon'], config['final_epsilon'], env, config['qtable_path'], config['evaluation'])

# Graphics
gui = GUI(agent, env)

# Start GUI thread
gui.mainloop()