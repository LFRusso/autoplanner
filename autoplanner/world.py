import random 
import os
import shutil
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import sys

from .agents import *
from .net import *
from .land import *

class World:
    def __init__(self, view_radius=10):
        self.map = map
        self.agent = None
        self.view_radius = view_radius
        
    def add_map(self, map):
        self.map = map

    def load(self, map_name):
        with open(map_name+"/land.pkl", 'rb') as land_file:
            self.map = pickle.load(land_file)

    def save(self, map_name):
        sys.setrecursionlimit(5000)
        if os.path.exists(map_name):
            shutil.rmtree(map_name)
        os.mkdir(map_name)

        land_uses = np.matrix([[c.type for c in line] for line in self.map.cells], dtype=int)
        with open(map_name+"/land.pkl", 'wb') as land_file:
            pickle.dump(self.map, land_file, pickle.HIGHEST_PROTOCOL)
        np.savetxt(map_name+"/land_uses.dat", land_uses, delimiter=' ', fmt='%i')

    def saveMap(self, filename):
        land_uses = np.matrix([[c.type for c in line] for line in self.map.cells], dtype=int)
        np.savetxt(filename, land_uses, delimiter=' ', fmt='%i')

    def loadMap(self, filename):
        land_uses = np.loadtxt(filename)
        for i in range(land_uses.shape[0]):
            for j in range(land_uses.shape[1]):
                if (land_uses[i,j]!=-1):
                    self.map.cells[i][j].setDeveloped(dev_type=land_uses[i,j], map=self.map)
                else:    
                    self.map.cells[i][j].setUndeveloped()

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
        padded_cells = np.pad(self.map.cells, 
                        ((self.view_radius, self.view_radius), (self.view_radius, self.view_radius)))

        x, y = self.agent.x + self.view_radius, self.agent.y + self.view_radius
        observed_map = s = padded_cells[x - self.view_radius:x + self.view_radius+1, y - self.view_radius:y + self.view_radius+1]

        accessibility_matrix = np.matrix([[c.norm_accessibility if c!=0 else 0 for c in line] for line in observed_map])
        road_matrix = np.matrix([[1 if c != 0 and c.type==0 else 0 for c in line] for line in observed_map])
        residential_matrix = np.matrix([[1 if c != 0 and c.type==1 else 0 for c in line] for line in observed_map])
        commercial_matrix = np.matrix([[1 if c != 0 and c.type==2  else 0 for c in line] for line in observed_map])
        industrial_matrix = np.matrix([[1 if c != 0 and c.type==3 else 0 for c in line] for line in observed_map])
        recreational_matrix = np.matrix([[1 if c != 0 and c.type==4 else 0 for c in line] for line in observed_map])

        state_matrix = np.array([road_matrix, residential_matrix, commercial_matrix, industrial_matrix, recreational_matrix, accessibility_matrix]).T
        return state_matrix

    # Resets the map to its starting state
    def resetMap(self):
        for cell in self.map.cells.flatten():
            cell.setUndeveloped()

        # Reseting agent development queue
        self.agent.devel_queue = self.agent.getDevelQueue()
        starting_cell = self.agent.devel_queue.pop(0)
        self.agent.moveTo(starting_cell)

    # Plots world
    def show(self, land_use=True, agent=False, accessibility=False, map_links=False, color=False, net=False, legends=True):
        plt.axis("equal")
        self.map.plotGrid(accessibility=accessibility, links=map_links)

        # Show original road network on top of map
        if(net):
            self.map.net.plotNet()

        # Color cells based on land use
        if (land_use):
            for i in range(self.map.lines):
                for j in range(self.map.columns):
                    if (self.map.cells[i,j].undeveloped==False):
                        if (color or self.map.cells[i,j].type==0):
                            plt.gca().add_patch(plt.Rectangle((j*self.map.cell_size, i*self.map.cell_size), 
                                                self.map.cell_size, self.map.cell_size, ec="gray", fc=self.map.cells[i,j].type_color_rgb, alpha=.7))
                        else:
                            plt.gca().add_patch(plt.Rectangle((j*self.map.cell_size, i*self.map.cell_size), 
                                                self.map.cell_size, self.map.cell_size, ec="k", fill=False, hatch=self.map.cells[i,j].hatch))

        # Adding custom legends
        if (legends):
            leg = plt.legend(['Undeveloped', 'Residential', 'Commercial', 'Industrial', 'Recreational', 'Road'], framealpha=1,
                            fontsize=10, frameon=False, handleheight=2.6, loc='lower right', bbox_to_anchor=(0.85, 0))
            if (color==False):
                hatches=[None, '...', '///', 'xx', 'OO', None]
                for i, lh in enumerate(leg.legendHandles):
                    lh.set_hatch(hatches[i])
                    lh.set_alpha(1)
            else:
                colors=['white', 'blue', 'yellow', 'red', 'limegreen', None]
                for i, lh in enumerate(leg.legendHandles):
                    lh.set_color(colors[i])
                    lh.set_edgecolor('gray')
                    lh.set_alpha(1)

            lh = leg.legendHandles[-1]
            lh.set_color('k')

        # Plot agent location
        if (agent and self.agent!=None):
            x, y = self.agent.cell.position
            plt.scatter(x, y, color="red")
        
        plt.title(f"Reward: {self.reward()}")
        plt.autoscale()
        plt.margins(0)
        plt.axis('off')
        plt.show()

    # Runs the simulation, storing state of the world at the end
    def run(self, episodes, steps=1000, epsilon=1, epsilon_decay=0.99975, min_epsilon=0.001, model=None):
        self.agent = Agent(self, model)
        
        rewards = []
        for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes'):
            self.resetMap()

            episode_reward = 0
            step = 1
            
            current_state = self.getState()
            for i in range(1, steps):
                action, terminal_state = self.agent.interact(epsilon=epsilon)
                new_state = self.getState()
                reward = self.reward()
                projected_rewards = [self.agent.prospectReward(act) for act in range(8)]
                episode_reward += reward

                self.agent.updateReplayMemory((current_state, action, reward, new_state, False, projected_rewards))
                self.agent.train(terminal_state=terminal_state, step=step)
                current_state = new_state
                step += 1

                if terminal_state:
                    break

            if not terminal_state: # Agent did not reach terminal state by completing map; terminal by number of iterations

                action, terminal_state = self.agent.interact(epsilon=epsilon)
                new_state = self.getState()
                reward = self.reward()
                projected_rewards = [self.agent.prospectReward(act) for act in range(8)]
                episode_reward += reward

                self.agent.updateReplayMemory((current_state, action, reward, new_state, True, projected_rewards))
                self.agent.train(terminal_state=True, step=step)
                step += 1

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
                epsilon = max(min_epsilon, epsilon)

            rewards.append(reward)
    
        return self.agent.model, self.agent.target_model


    # Runs always choosing the best immediate action
    def runGreedy(self, steps=1000):
        self.agent = Agent(self)

        for i in range(1, steps):
            action, terminal_state = self.agent.greedyInteract()
            if (terminal_state):
                break
        return

    # Runs choosing random action
    def runRandom(self, steps=1000):
        self.agent = Agent(self)

        for i in range(1, steps):
            action, terminal_state = self.agent.randomInteract()
            if (terminal_state):
                break
        return