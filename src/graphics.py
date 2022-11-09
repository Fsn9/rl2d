import tkinter as tk    
import matplotlib.pyplot as plt
from math import sin, cos
from environment import EmptyEnvironment, ObstacleEnvironment, Entities
import os
import time
import numpy as np

# REFRESH RATE (milisseconds)
LEARNING_REFRESH_RATE = 1
EVALUATION_REFRESH_RATE = 750

def compute_pixel_size(grid_width, grid_height, big_side):
    if grid_width > grid_height:
        factor = grid_width / grid_height
        width = big_side
        height = int(big_side / factor)
        pixel_size = width / grid_width

    elif grid_width < grid_height:
        factor = grid_height / grid_width
        height = big_side
        width = int(big_side / factor)
        pixel_size = height / grid_height
        
    else:
        width = big_side
        height = big_side
        pixel_size = big_side / grid_width

    return pixel_size, width, height

class GUI(tk.Tk):
    def __init__(self, learner, environment):
        # initialize a Tk object
        tk.Tk.__init__(self)

        # save reinforcement learning object
        self.__learner = learner

        # save environment
        self.__environment = environment

        # define objects containing rectangles of food and snake
        self.__agent_gui = []
        self.__agent_orientation_gui = []
        self.__goal_gui = []
        self.__obstacles_gui = []

        # @TODO: put this in json or yaml
        num_obstacles = 0
        self.__grid_width, self.__grid_height, self.big_side = self.__environment.w, self.__environment.h, 400
        self.__pixel_size, self.width, self.height = compute_pixel_size(self.__grid_width,
            self.__grid_height,
            self.big_side
        )

        # set title
        self.title("rl2d")

        # set geometry
        #print('Graphics w,h:', self.width, self.height)
        self.extra_length = 300
        self.geometry(str(self.width) + "x" + str(self.height + self.extra_length))

        # string length limit
        self.__str_len_limit = 6
        self.__sim_time = tk.IntVar()

        # draw canvas
        self.__canvas = self.create_canvas(self.width, self.height, "white")
        self.__canvas.grid(row = 0, column = 0, columnspan = 2, sticky = tk.W + tk.E + tk.N + tk.S)
        for r in range(self.__grid_height):
            for c in range(self.__grid_width):
                self.draw_rectangle(c,r,"white")

        # draw labels
        if not self.__learner.evaluation:
            self.label_reward = self.create_label('Average reward:',1,0)
            self.label_reward_val = self.create_label(str(self.__learner.mov_avg_reward),1,1)
            self.label_steps = self.create_label('Average steps:',2,0)
            self.label_steps_val = self.create_label(str(self.__learner.mov_avg_steps),2,1)
            self.label_epsilon = self.create_label('Epsilon:',3,0)
            self.label_epsilon_val = self.create_label(str(self.__learner.current_epsilon),3,1)
            self.label_gamma = self.create_label('Gamma:',4,0)
            self.label_gamma_val = self.create_label(str(self.__learner.current_gamma),4,1)
            self.label_alpha = self.create_label('Alpha:',5,0)
            self.label_alpha_val = self.create_label(str(self.__learner.alpha),5,1)
            self.label_episodes_left = self.create_label('Episodes left:',6,0)
            self.label_episodes_left_val = self.create_label(str(self.__learner.episodes_left),6,1)
            self.label_slider_time = self.create_label("Simulation period (ms):", 7,0)
            self.label_slider_time_slider = tk.Scale(from_ = 1, to = 10000, variable = self.__sim_time, orient = tk.HORIZONTAL, command = self.__change_sim_time)
            self.label_slider_time_slider.grid(row = 7, column = 1)
            self.__sim_time.set(LEARNING_REFRESH_RATE)
        else:
            self.label_scene = self.create_label('Scene:',1,0)
            self.label_scene_name = self.create_label("",1,1)
            self.label_slider_time = self.create_label("Simulation period (ms):", 2,0)
            self.label_slider_time_slider = tk.Scale(from_ = 1, to = 10000, variable = self.__sim_time, orient = tk.HORIZONTAL, command = self.__change_sim_time)
            self.label_slider_time_slider.grid(row = 2, column = 1)
            self.__sim_time.set(EVALUATION_REFRESH_RATE)

        # First drawing
        self.draw = self.draw_simple
        self.clear()
        if isinstance(self.__environment,ObstacleEnvironment):
            self.draw = self.draw_complex
        self.draw()

        # start learning process
        self.run_evaluation() if self.__learner.evaluation else self.run_learning()

    def __str__(self):
        return '--GUI--' + '\n' + 'width = ' + str(self.width) + ' pixels' + '\nheight = ' + str(
            self.height) + ' pixels' + '\npixel_size = ' + str(int(self.__pixel_size)) + ' pixels'

    def create_label(self,text, row, col):
        label = tk.Label(text = text, anchor = "w")
        label.grid(row = row, column = col)
        return label

    def create_canvas(self, width, height, color):
        canvas = tk.Canvas(width = width, height = height, bg = color)
        return canvas

    def create_button(self, text, row, col, callback, state = tk.DISABLED):
        button = tk.Button(text = text, command = callback, state = state)
        button.grid(row = row, column = col)
        return button

    def draw_rectangle(self, x, y, color, outline = "black"):
        rectangle = self.__canvas.create_rectangle(x * self.__pixel_size, y * self.__pixel_size, (1 + x) * self.__pixel_size,
        (1 + y) * self.__pixel_size, fill = color, outline = outline)
        return rectangle

    def draw_circle(self, x, y, r, color, outline = "black"):
        circle = self.__canvas.create_oval(self.__pixel_size * (x + 0.5 - r), self.__pixel_size * (y + 0.5 - r),
        self.__pixel_size * (x + 0.5 + r), self.__pixel_size * (y + 0.5 + r), fill = color, outline = outline, width = 1)
        return circle

    def draw_arrow(self, x0, y0, x1, y1):
        arrow = self.__canvas.create_line(x0 * self.__pixel_size, y0 * self.__pixel_size,
            x1 * self.__pixel_size, y1 * self.__pixel_size, arrow = tk.LAST, fill = "black", width = 5
        )
        return arrow

    def draw_agent(self):
        agent_pos = self.__environment.entities[Entities.AGENT].pos
        agent_ori = self.__environment.entities[Entities.AGENT].theta
        mid_pos_translated = agent_pos[0] + 0.5, agent_pos[1] + 0.5
        self.__agent_gui = self.draw_circle(agent_pos[0], agent_pos[1], 0.5, 'gray')
        self.__agent_orientation_gui = self.draw_arrow(mid_pos_translated[0], mid_pos_translated[1], mid_pos_translated[0] + 0.5 * cos(agent_ori), mid_pos_translated[1] + 0.5 * sin(agent_ori))

    def draw_goal(self):
        goal_pos = self.__environment.entities[Entities.GOAL].pos
        self.__goal_gui = self.draw_circle(goal_pos[0], goal_pos[1], 0.5, 'red')

    def draw_obstacles(self):
        obstacles = self.__environment.entities[Entities.OBSTACLE]
        for obstacle in obstacles:
            self.__obstacles_gui.append(self.draw_circle(obstacle.pos[0], obstacle.pos[1], 0.5, 'black'))

    def draw_learning_stats(self):
        steps, gamma, epsilon, reward, episodes = self.__learner.get_stats()
        self.label_steps_val.config(text=str(steps)[0:self.__str_len_limit])
        self.label_gamma_val.config(text=str(gamma)[0:self.__str_len_limit])
        self.label_epsilon_val.config(text=str(epsilon)[0:self.__str_len_limit])
        self.label_reward_val.config(text=str(reward)[0:self.__str_len_limit])
        self.label_episodes_left_val.config(text=str(episodes)[0:self.__str_len_limit])
        #self.label_collisions_wall_val.config(text=str(wall_collisions)[0:6])

    def draw_evaluation_stats(self):
        self.label_scene_name.config(text=str(self.__environment.cur_scene_name)[0:self.__str_len_limit + 1])

    def draw_simple(self):
        self.draw_agent()
        self.draw_goal()
        if not self.__learner.evaluation:
            self.draw_learning_stats()
        else:
            self.draw_evaluation_stats()

    def draw_complex(self):
        self.draw_agent()
        self.draw_goal()
        self.draw_obstacles()
        if not self.__learner.evaluation:
            self.draw_learning_stats()
        else:
            self.draw_evaluation_stats()

    def clear(self):
        self.__canvas.delete(self.__agent_gui)
        self.__canvas.delete(self.__goal_gui)
        self.__canvas.delete(self.__agent_orientation_gui)
        for obs_gui in self.__obstacles_gui:
            self.__canvas.delete(obs_gui)
        self.__canvas.delete(self.__obstacles_gui)
        self.__obstacles_gui.clear()

    def repaint(self):
        self.clear()
        self.draw()

    def __change_sim_time(self, val):
        self.__sim_time.set(val)

    def run_learning(self):
        if self.__learner.finished:
            print('Learning is finished!')
            folder_path, time_string = self.__learner.export_results()

            plt.title('Average cumulative reward per epoch', fontsize = 14)
            plt.plot([x for x in range(len(self.__learner.reward_sums))], self.__learner.reward_sums)
            plt.xlabel("Epochs (Window of " + str(self.__learner.window_size_reward_moving_avg) + " episodes)", fontsize = 14)
            plt.ylabel("Average cumulative reward", fontsize = 14)
            plt.savefig(os.path.join(folder_path, 'reward-' + time_string + '.eps'), bbox_inches='tight', format = 'eps')
            plt.show()

            plt.title('Average number of steps per epoch', fontsize = 14)
            plt.plot([x for x in range(len(self.__learner.steps))], self.__learner.steps)
            plt.xlabel("Epochs (Window of " + str(self.__learner.window_size_steps_moving_avg) + " episodes)", fontsize = 14)
            plt.ylabel("Average steps", fontsize = 14)
            plt.savefig(os.path.join(folder_path, 'steps-' + time_string + '.eps'), bbox_inches='tight', format = 'eps')
            plt.show()

            if isinstance(self.__environment, ObstacleEnvironment):
                plt.title('Episode ending cause average per epoch', fontsize = 14)
                plt.plot([x for x in range(len(self.__learner.ending_causes))], self.__learner.ending_causes)
                plt.xlabel("Epochs (Window of " + str(self.__learner.window_size_ending_causes_moving_avg) + " episodes)", fontsize = 14)
                plt.ylabel("Average ending cause", fontsize = 14)
                plt.savefig(os.path.join(folder_path, 'ending_causes-' + time_string + '.eps'), bbox_inches='tight', format = 'eps')
                plt.show()

            exit()
        else:
            terminal = self.__learner.act()
            if terminal[0] or terminal[1] == "skip":
                self.__environment.reset()
            self.repaint()
        self.after(self.__sim_time.get(), self.run_learning)

    def run_evaluation(self):
        terminal = self.__learner.act_eval()
        next_scenario = True
        if terminal[0]:
            next_scenario = self.__environment.play_next_scenario()
        if not next_scenario:
            print('Evaluation is finished!')
            for scenario_idx, trajectory in self.__learner.trajectories.items():
                plt.title(f'Trajectory for scenario {scenario_idx}')
                agent_x, agent_y = (np.array(trajectory['x']) + 0.5)[0], (np.array(trajectory['y']) + 0.5)[0]
                plt.annotate(text = "Start", xy = (agent_x, agent_y), xytext = (agent_x - 0.25, agent_y + 0.1), size = 12, weight = 'bold', color = "white")
                plt.annotate(text = "End", xy = (trajectory['goal']['x'] + 0.5, trajectory['goal']['y'] + 0.5) , xytext = (trajectory['goal']['x'] + 0.5 - 0.25, trajectory['goal']['y'] + 0.5 + 0.1), size = 12, weight = 'bold', color = "white")
                plt.plot(np.array(trajectory['x']) + 0.5, np.array(trajectory['y']) + 0.5, color = "black", linestyle = "dashed")
                plt.scatter(np.array(trajectory['x'][0]) + 0.5, np.array(trajectory['y'][0]) + 0.5, color = "blue", s = 2600)
                plt.scatter(np.array(trajectory['x'][1:]) + 0.5, np.array(trajectory['y'][1:]) + 0.5, color = "grey", s = 2600)
                plt.scatter(np.array(trajectory['obstacles']['x']) + 0.5, np.array(trajectory['obstacles']['y']) + 0.5, color = "black", s = 2600)
                plt.scatter(np.array(trajectory['goal']['x']) + 0.5, np.array(trajectory['goal']['y']) + 0.5, color = "red", s = 2600)
                # Plot trajectory orientation
                xprev, yprev = trajectory['x'][0], trajectory['y'][0]
                for idx, (x,y) in enumerate(zip(trajectory['x'], trajectory['y'])):
                    if idx > 0:
                        plt.arrow(xprev + 0.5, yprev + 0.5, (x - xprev) * 0.25, (y - yprev) * 0.25, shape = "full", width = 0.05, color = "black", fill = True)
                        xprev, yprev = x, y
                plt.xlabel("x", fontsize = 18)
                plt.ylabel("y", fontsize = 18)
                plt.xlim([0, self.__environment.w])
                plt.ylim([0, self.__environment.h])
                plt.grid()
                plt.savefig(os.path.join(self.__learner.run_folder_path, 'trajectory-' + \
                    str(scenario_idx) + '-' + self.__learner.run_timestamp + '.eps'), bbox_inches='tight', format = 'eps')
                plt.clf()
            time.sleep(1)
            exit()
        self.repaint()
        self.after(self.__sim_time.get(), self.run_evaluation)