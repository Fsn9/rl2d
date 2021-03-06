from qlearning import *
from environment import *
from graphics import GUI

# Empty environment
env = EmptyEnvironment(5,5)
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