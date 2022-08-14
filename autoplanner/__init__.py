import random 
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from .agents import *
from .net import *
from .land import *

class World:
    def __init__(self, map):
        self.map = map
        self.original_map = deepcopy(map)

        self.agent = None
        return

    def reward(self):
        reward = 0
        cells = self.map.cells.flatten()
        for cell in cells:
            reward += cell.score
        return reward
    
    # Prints info about the world
    def showParams(self):
        return

    # Returns matrix representing the current state of the world
    def getState(self):
        state_matrix = np.matrix([[c.type+1 for c in line] for line in self.map.cells])
        if (self.agent != None):
            # Current agent position is represented by adding 6 to the entry where it stands
            state_matrix[self.agent.x, self.agent.y] += 6
        return state_matrix

    # Resets the map to its starting state
    def resetMap(self):
        self.map = deepcopy(self.original_map)

    # Plots world
    def show(self, land_use=True, agent=False, accessibility=False, map_links=False):
        self.map.plotGrid(accessibility=accessibility, links=map_links)

        self.map.net.plotNet()

        if (land_use):
            for i in range(self.map.lines):
                for j in range(self.map.columns):
                    if (self.map.cells[i,j].undeveloped==False):
                        plt.gca().add_patch(plt.Rectangle((j*self.map.cell_size, i*self.map.cell_size), 
                                            self.map.cell_size, self.map.cell_size, ec="gray", fc=self.map.cells[i,j].type_color_rgb, alpha=.7))

        if (agent and self.agent!=None):
            x, y = self.agent.cell.position
            plt.scatter(x, y, color="red")
        plt.show()

    # Runs the simulation, storing state of the world at the end
    def run(self, episodes, steps=1000, epsilon=1, epsilon_decay=0.99975, min_epsilon=0.001):
        self.agent = Agent(self)
        cells = self.map.cells.flatten() # Randomly placing agent in the map
        
        rewards = []

        for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes'):
            starting_cell = random.choice(cells)
            self.agent.moveTo(starting_cell)
            self.resetMap()

            episode_reward = 0
            step = 1
            
            current_state = self.getState()
            for i in range(1, steps):
                action = self.agent.interact(epsilon=epsilon)
                new_state = self.getState()
                reward = self.reward()
                episode_reward += reward

                self.agent.updateReplayMemory((current_state, action, reward, new_state, False))
                self.agent.train(terminal_state=False, step=step)
                current_state = new_state
                step += 1

            action = self.agent.interact(epsilon=0.5)
            new_state = self.getState()
            reward = self.reward()
            episode_reward += reward

            self.agent.updateReplayMemory((current_state, action, reward, new_state, True))
            self.agent.train(terminal_state=True, step=step)
            step += 1

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
                epsilon = max(min_epsilon, epsilon)

            #self.show(land_use=True, agent=True, accessibility=False, map_links=False)
            rewards.append(episode_reward)
    
        plt.plot(np.arange(1, len(rewards)+1), rewards)
        plt.show()
        return  