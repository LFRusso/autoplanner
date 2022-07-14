import random 
import numpy as np
import matplotlib.pyplot as plt

from .agents import Agent
from .net import *
from .land import *

class World:
    def __init__(self, map, agents=10, agent_view_radius=5):
        self.map = map
        self.agents = []

        self._initAgents(agents, agent_view_radius)
        return
    
    # Initialize the agents by randomly selecting their starting posion
    def _initAgents(self, n_agents, view_radius=5):
        cells = self.map.cells.flatten()
        for i in range(n_agents):
            selected_cell = random.choice(cells)
            new_agent = Agent(selected_cell, view_radius)
            self.agents.append(new_agent)
        return
    
    # Prints info about the world
    def showParams(self):
        return

    # Ramdomly adds more agents to the world with default parameters
    def addRandomAgents(self, n=1, view_radius=5):
        cells = self.map.cells.flatten()
        for i in range(n):
            selected_cell = random.choice(cells)
            new_agent = Agent(selected_cell, view_radius)
            self.agents.append(new_agent)
        return

    # Adds previously specified, custom agents to the world
    def addAgents(self, agents):
        self.agents.append(agents)
        return
    
    # Runs the simulation, storing state of the world at the end
    def run(self, max_iter=1000):
        for i in range(max_iter):
            for agent in self.agents:
                agent.interact(self)
        return   

    # Plots world
    def show(self, land_use=True, agents=False, accessibility=False, map_links=False):
        self.map.plotGrid(accessibility=accessibility, links=map_links)
        self.map.net.plotNet()
        if (land_use):
            for i in range(self.map.lines):
                for j in range(self.map.columns):
                    if (self.map.cells[i,j].undeveloped==False):
                        plt.gca().add_patch(plt.Rectangle((j*self.map.cell_size, i*self.map.cell_size), 
                                            self.map.cell_size, self.map.cell_size, ec="gray", fc=(0,0,1), alpha=.7))

        if (agents):
            x, y = np.transpose([agent.cell.position for agent in self.agents])
            plt.scatter(x, y, color="red")
        plt.show()
        
