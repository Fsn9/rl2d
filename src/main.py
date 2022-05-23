import numpy as np
from qlearning import *
from environment import *
from statistics import mean
import matplotlib.pyplot as plt


# Empty environment
env = EmptyEnvironment(5,5)

# Obstacle Environment 5x5 and 3 obstacles
# env = ObstacleEnvironment(5,5,0)

learning_rate = 0.1
discount_factor = 0.99
episodes = 10000
initial_epsilon = 1
final_epsilon = 0.05
agent = QLearner('simple', 'simple', learning_rate, discount_factor, episodes, initial_epsilon, final_epsilon, env)

action, obs, next_obs, terminal, reward, neighbour = [], [], [], False, 0, None

rew_sum = 0
rew_sums = []

for ep in range(episodes):
    obs, _, _, _ = env.reset()
    print('\nEpisode: ', ep, '\n')
    while not terminal:
        # 1. Decide
        action = agent.decide(obs)

        # 2. Act and observe
        next_obs, terminal, reward, neighbour = env.step(action)

        # 3. If invalid action skip to next decision
        if neighbour == Entities.VOID.value:
            #print('continue')
            continue

        # 4. Learn
        agent.learn(obs, next_obs, action, reward, terminal, neighbour)
        #print('obs: ', obs, 'next_obs: ', next_obs, 'action: ', action, 'ter: ', terminal)

        # 5. Save observation for next decision
        obs = next_obs

        # Collect reward
        rew_sum += reward

    terminal = False

    # Save rewards for statistics
    rew_sums.append(rew_sum)
    rew_sum = 0

agent.print_result()
plt.plot([x for x in range(len(rew_sums))], rew_sums)
plt.show()
