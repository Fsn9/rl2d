# rl2d

This code launches a Python Tkinter interface displaying a learning process of a **Q-learning** agent for two navigational tasks each with its own environment. 
* Learning to reach a point in a 2D empty grid environment.
* Learning to reach a point in a 2D obstacle scattered grid environment.

To run the empty grid world one must set the `--env_type empty`. To run the obstacle scattered environment, one must set the launching argument `--env_type obstacle`.

___

## To run
To run with **default** arguments: `python run_rl2d.py`

The arguments available are:
* `--env_type`, **default**='empty'. It is the type of the environment. Can be `'empty'` or `'obstacle'`.
* `--env_dim`, **default**=5. The environment dimension. It needs to be > 2 and < 10.
* `--num_obstacles`, **default**=2. The number of obstacles in the environment.

* `--learning_rate`, **default**=0.1. The learning rate needs to be >= 0 and <= 1.
* `--discount_factor`, **default**=0.99. The discount factor >= and <= 1. In the extreme, for a value of 1 we have a long-term view agent. For a value of 0 we have a myopic agent.
* `--episodes`, **default**=4000. The number of learning episodes.
* `--initial_epsilon`, **default**=1. The initial epsilon is the exploration probability in the beggining of the learning process. A value of 1 means a total random agent. A value of 0 is a total greedy agent.
* `--final_epsilon`, **default**=0.05. The value of the final exploration probability.