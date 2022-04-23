import numpy as np
from qlearning import *
from environment import *

# Environment 5x5 and 3 obstacles
env = Environment(5,5,2)

learning_rate = 0.05
discount_factor = 0.99
episodes = 2
initial_epsilon = 1
final_epsilon = 0.05
agent = QLearner('simple', 'simple', learning_rate, discount_factor, episodes, initial_epsilon, final_epsilon)

action, obs, next_obs, terminal, reward = [], [], [], False, 0

for ep in range(episodes):
    obs, _, _ = env.reset()
    while not terminal:
        action = agent.decide(obs)
        print('\n',env)
        next_obs, terminal, reward = env.step(action)
        agent.learn(obs, next_obs, action, reward, terminal)
        obs = next_obs
