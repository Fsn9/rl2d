import tkinter as tk    
import matplotlib.pyplot as plt
from math import sin, cos
from environment import EmptyEnvironment, ObstacleEnvironment, Entities

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
        self.geometry(str(self.width) + "x" + str(self.height))

        # string length limit
        self.__str_len_limit = 6
        self.__sim_time = tk.IntVar()
        self.__sim_time.set(1)

        # draw canvas
        self.__canvas = self.create_canvas(self.width, self.height, "black")
        self.__canvas.grid(row = 0, column = 0, columnspan = 2, sticky = tk.W + tk.E + tk.N + tk.S)

        # draw labels
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
        self.label_slider_time = self.create_label("Simulation time:", 6,0)
        self.label_slider_time_slider = tk.Scale(from_ = 1, to = 5000, variable = self.__sim_time, orient = tk.HORIZONTAL, command = self.__change_sim_time)
        self.label_slider_time_slider.grid(row = 6, column = 1)
        #self.label_collisions_wall = self.create_label('Wall collisions:',6,0)
        #self.label_collisions_wall_val = self.create_label(str(self.__learner.counter_collisions_with_wall),6,1)
        #self.label_collisions_itself = self.create_label('Self collisions:',7,0)
        #self.label_collisions_itself_val = self.create_label(str(self.__learner.counter_collisions_with_itself),7,1)

        # First drawing
        self.draw = self.draw_simple
        self.clear()
        if isinstance(self.__environment,ObstacleEnvironment):
            self.draw = self.draw_complex
            self.draw_walls()
        self.draw()

        # start learning process
        self.run_rl()

    def __str__(self):
        return '--GUI--' + '\n' + 'width = ' + str(self.width) + ' pixels' + '\nheight = ' + str(
            self.height) + ' pixels' + '\npixel_size = ' + str(int(self.__pixel_size)) + ' pixels'

    def create_label(self,text, row, col):
        label = tk.Label(text = text)
        label.grid(row = row, column = col)
        return label   

    def create_canvas(self, width, height, color):
        canvas = tk.Canvas(width = width, height = height, bg = color)
        return canvas

    def draw_rectangle(self, x, y, color):
        rectangle = self.__canvas.create_rectangle(x * self.__pixel_size, y * self.__pixel_size, (1 + x) * self.__pixel_size,
                                            (1 + y) * self.__pixel_size, fill = color)
        return rectangle

    def draw_arrow(self, x0, y0, x1, y1):
        arrow = self.__canvas.create_line(x0 * self.__pixel_size, y0 * self.__pixel_size,
            x1 * self.__pixel_size, y1 * self.__pixel_size, arrow = tk.LAST, fill = "blue", width = 5
        )
        return arrow

    def draw_walls(self):
        for x in range(self.__grid_width + 2):
            self.draw_rectangle(x, 0, 'brown')
        for y in range(1, self.__grid_height + 2):
            self.draw_rectangle(0, y, 'brown')
        for x in range(1, self.__grid_width + 2):
            self.draw_rectangle(x, self.__grid_height - 1, 'brown')
        for y in range(1, self.__grid_height + 2):
            self.draw_rectangle(self.__grid_width - 1, y, 'brown')

    def draw_agent(self):
        agent_pos = self.__environment.entities[Entities.AGENT].pos
        agent_ori = self.__environment.entities[Entities.AGENT].theta
        if self.__env_with_walls:
            #print('Graphics: drawing agent with walls')
            mid_pos_translated = agent_pos[0] + 1.5, agent_pos[1] + 1.5
            self.__agent_gui = self.draw_rectangle(agent_pos[0] + 1, agent_pos[1] + 1, 'white')
            #print('Graphics, agent_pos: ', agent_pos[0] + 1, agent_pos[1] + 1)
        else:
            mid_pos_translated = agent_pos[0] + 0.5, agent_pos[1] + 0.5
            self.__agent_gui = self.draw_rectangle(agent_pos[0], agent_pos[1], 'white')
        self.__agent_orientation_gui = self.draw_arrow(mid_pos_translated[0], mid_pos_translated[1], mid_pos_translated[0] + 0.5 * cos(agent_ori), mid_pos_translated[1] + 0.5 * sin(agent_ori))

    def draw_goal(self):
        goal_pos = self.__environment.entities[Entities.GOAL].pos
        #print('Graphics, goal_pos: ', goal_pos)
        if self.__env_with_walls:
            self.__goal_gui = self.draw_rectangle(goal_pos[0] + 1, goal_pos[1] + 1, 'red')
        else:
            self.__goal_gui = self.draw_rectangle(goal_pos[0], goal_pos[1], 'red')

    def draw_obstacles(self):
        # Obstacles
        obstacles = self.__environment.entities[Entities.OBSTACLE]
        for obstacle in obstacles:
            self.__obstacles_gui.append(self.draw_rectangle(obstacle.pos[0] + 1, obstacle.pos[1] + 1, 'brown'))

    def draw_stats(self):
        # Statistics
        steps, gamma, epsilon, reward, episodes = self.__learner.get_stats()
        self.label_steps_val.config(text=str(steps)[0:self.__str_len_limit])
        self.label_gamma_val.config(text=str(gamma)[0:self.__str_len_limit])
        self.label_epsilon_val.config(text=str(epsilon)[0:self.__str_len_limit])
        self.label_reward_val.config(text=str(reward)[0:self.__str_len_limit])
        self.label_episodes_left_val.config(text=str(episodes)[0:self.__str_len_limit])
        #self.label_collisions_itself_val.config(text=str(self_collisions)[0:6])
        #self.label_collisions_wall_val.config(text=str(wall_collisions)[0:6])

    def draw_simple(self):
        self.draw_agent()
        self.draw_goal()
        self.draw_stats()

    def draw_complex(self):
        self.draw_simple()
        self.draw_obstacles()
        self.draw_stats()

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

    def run_rl(self):
        if self.__learner.finished:
            print('Learning is finished!')
            self.__learner.print_result()
            plt.title('Reward')
            plt.plot([x for x in range(len(self.__learner.mov_avgs_reward))], self.__learner.mov_avgs_reward)
            plt.show()
            plt.title('Steps')
            plt.plot([x for x in range(len(self.__learner.mov_avgs_steps))], self.__learner.mov_avgs_steps)
            plt.show()
            exit()
        else:
            terminal = self.__learner.act()
            if terminal:
                print('episodes: ', self.__learner.episodes_passed)
            self.repaint()
        self.after(self.__sim_time.get(), self.run_rl)