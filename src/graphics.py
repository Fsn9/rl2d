import tkinter as tk    
import matplotlib.pyplot as plt
from math import sin, cos
from environment import EmptyEnvironment, ObstacleEnvironment, Entities
import os
import time

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
        self.__env_with_walls = False
        if isinstance(self.__environment, ObstacleEnvironment):
            self.__env_with_walls = True

        # define objects containing rectangles of food and snake
        self.__agent_gui = []
        self.__agent_orientation_gui = []
        self.__goal_gui = []
        self.__obstacles_gui = []

        # @TODO: put this in json or yaml
        num_obstacles = 0
        #print('Graphics, envw, envh:',self.__environment.w, self.__environment.h)
        if self.__env_with_walls:
            #print('Graphics, obstacleenv')
            self.__grid_width, self.__grid_height, self.big_side = self.__environment.w + 2, self.__environment.h + 2, 400
        else:
            #print('Graphics, emptyenv')
            self.__grid_width, self.__grid_height, self.big_side = self.__environment.w, self.__environment.h, 400
        #print('Graphics, gw, gh:',self.__grid_width, self.__grid_height)
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
            self.label_episodes_left = self.create_label('EpisodesLeft:',5,0)
            self.label_episodes_left_val = self.create_label(str(self.__learner.episodes_left),5,1)
            self.label_slider_time = self.create_label("Simulation period (ms):", 6,0)
            self.label_slider_time_slider = tk.Scale(from_ = 1, to = 10000, variable = self.__sim_time, orient = tk.HORIZONTAL, command = self.__change_sim_time)
            self.label_slider_time_slider.grid(row = 6, column = 1)
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
            self.draw_walls()
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

    def draw_walls(self):
        for x in range(self.__grid_width + 2):
            self.draw_circle(x, 0, 0.5, 'black')
        for y in range(1, self.__grid_height + 2):
            self.draw_circle(0, y, 0.5, 'black')
        for x in range(1, self.__grid_width + 2):
            self.draw_circle(x, self.__grid_height - 1, 0.5, 'black')
        for y in range(1, self.__grid_height + 2):
            self.draw_circle(self.__grid_width - 1, y, 0.5, 'black')

    def draw_agent(self):
        agent_pos = self.__environment.entities[Entities.AGENT].pos
        agent_ori = self.__environment.entities[Entities.AGENT].theta
        if self.__env_with_walls:
            mid_pos_translated = agent_pos[0] + 1.5, agent_pos[1] + 1.5
            self.__agent_gui = self.draw_circle(agent_pos[0] + 1, agent_pos[1] + 1, 0.5, 'gray')
            #print('Graphics, agent_pos: ', agent_pos[0] + 1, agent_pos[1] + 1)
        else:
            mid_pos_translated = agent_pos[0] + 0.5, agent_pos[1] + 0.5
            self.__agent_gui = self.draw_circle(agent_pos[0], agent_pos[1], 0.5, 'gray')
        self.__agent_orientation_gui = self.draw_arrow(mid_pos_translated[0], mid_pos_translated[1], mid_pos_translated[0] + 0.5 * cos(agent_ori), mid_pos_translated[1] + 0.5 * sin(agent_ori))

    def draw_goal(self):
        goal_pos = self.__environment.entities[Entities.GOAL].pos
        if self.__env_with_walls:
            self.__goal_gui = self.draw_circle(goal_pos[0] + 1, goal_pos[1] + 1, 0.5, 'red')
        else:
            self.__goal_gui = self.draw_circle(goal_pos[0], goal_pos[1], 0.5, 'red')

    def draw_obstacles(self):
        obstacles = self.__environment.entities[Entities.OBSTACLE]
        for obstacle in obstacles:
            self.__obstacles_gui.append(self.draw_circle(obstacle.pos[0] + 1, obstacle.pos[1] + 1, 0.5, 'black'))

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

            plt.title('Reward')
            plt.plot([x for x in range(len(self.__learner.steps))], self.__learner.steps)
            plt.xlabel("Epochs (Window of " + str(self.__learner.window_size_reward_moving_avg) + " episodes)")
            plt.ylabel("Value")
            plt.savefig(os.path.join(folder_path, 'reward-' + time_string + '.png'), bbox_inches='tight')
            plt.show()

            plt.title('Steps')
            plt.plot([x for x in range(len(self.__learner.reward_sums))], self.__learner.reward_sums)
            plt.xlabel("Epochs (Window of " + str(self.__learner.window_size_steps_moving_avg) + " episodes)")
            plt.ylabel("Value")
            plt.savefig(os.path.join(folder_path, 'steps-' + time_string + '.png'), bbox_inches='tight')
            plt.show()

            if isinstance(self.__environment, ObstacleEnvironment):
                plt.title('Episode ending cause average (1 : goal, 0 : collision)')
                plt.plot([x for x in range(len(self.__learner.ending_causes))], self.__learner.ending_causes)
                plt.xlabel("Epochs (Window of " + str(self.__learner.window_size_ending_causes_moving_avg) + " episodes)")
                plt.ylabel("Ending cause")
                plt.savefig(os.path.join(folder_path, 'ending_causes-' + time_string + '.png'), bbox_inches='tight')
                plt.show()

            exit()
        else:
            terminal = self.__learner.act()
            if terminal[0] or terminal[1] == "skip":
                self.__environment.reset()
            self.repaint()
        self.after(self.__sim_time.get(), self.run_learning)

    def run_evaluation(self):
        if self.__learner.act_eval() is None: 
            print('Evaluation is finished!')
            self.repaint()
            time.sleep(2)
            exit()
        self.repaint()
        self.after(self.__sim_time.get(), self.run_evaluation)