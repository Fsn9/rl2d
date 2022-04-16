import numpy as np
from qlearning import *
from environment import *

# Environment 5x5 and 3 obstacles
env = Environment(5,5,3)

learning_rate = 0.05
discount_factor = 0.99
episodes = 1
initial_epsilon = 1
final_epsilon = 0.05
agent = QLearner('simple', 'simple', learning_rate, discount_factor, episodes, initial_epsilon, final_epsilon)


for ep in range(episodes):
    observation, done = env.reset()
    print('before action')
    print(env)
    env.step(agent.decide(observation))
    print('after action')
    print(env)
