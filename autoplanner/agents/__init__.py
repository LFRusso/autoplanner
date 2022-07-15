import numpy as np
from math import ceil, floor

class Agent:
    def __init__(self, cell, view_radius):
        self.cell = cell
        self.view_radius = view_radius
        self.x, self.y = cell.idx[0], cell.idx[1]

        self.dev_sites = []
        self.dev_sites_scores = []
        self.type_id = -1
        self._setAgentTypeID()
        return
    
    # To be implemented by different types of agents
    def _setAgentTypeID(self):
        return

    # To be implemented by different types of agents 
    def build(self):
        self.cell.setDeveloped()
        return

    # To be implemented by different types of agents
    def getScore(self, cell):
        return cell.score

    # Execute explore-build behavior
    def interact(self, world):
        neighborhood = self.lookAround(world.map).flatten()
        self.updateDevSites(neighborhood)
        next_cell, proffit = self.getNextCell(neighborhood)

        if (next_cell==self.cell):
            # build
            self.build()
        else:
            # move to next
            self.moveTo(next_cell)
        return

    # Selece neighboring cells from the one the agent is currently on
    def lookAround(self, map):
        close_cells = map.cells[max(self.x-self.view_radius,0) : min(self.x+self.view_radius+1, map.lines),
                                max(self.y-self.view_radius,0) : min(self.y+self.view_radius+1, map.columns)]
        return close_cells
    
    # Get the next cell by popping the top of the seen cells list
    def getNextCell(self, cell_list):
        next_cell, self.dev_sites = self.dev_sites[0], self.dev_sites[1:]
        score, self.dev_sites_scores = self.dev_sites_scores[0], self.dev_sites_scores[1:]

        return next_cell, score

    # Changes the agent current cell
    def moveTo(self, cell):
        self.cell = cell
        self.x, self.y = cell.idx[0], cell.idx[1]
        return

    # Adds more cells to the agents previously seen sites along with their score when last seen
    def updateDevSites(self, new_sites):
        new_sites = [site for site in new_sites if site.undeveloped]
        for site in new_sites:
            if (site not in self.dev_sites):
                self.dev_sites = np.append(self.dev_sites, site)
                self.dev_sites_scores = np.append(self.dev_sites_scores, self.getScore(site))

        # Sort sites by score
        sorted_idx = np.argsort(self.dev_sites_scores)[::-1]
        self.dev_sites = self.dev_sites[sorted_idx]
        self.dev_sites_scores = self.dev_sites_scores[sorted_idx]

        # Removing bottom 10%
        slc = ceil(len(self.dev_sites)*0.9)
        self.dev_sites = self.dev_sites[:slc]
        self.dev_sites_scores = self.dev_sites_scores[:slc]
        return
    
class ResidentialBuilder(Agent):
    def _setAgentTypeID(self):
        self.type_id = 0
        return

    def getScore(self, cell):
        return cell.score

    def build(self):
        self.cell.type_color_rgb = (0,0,1)
        self.cell.type = 0
        self.cell.setDeveloped()
        return

class ComercialBuilder(Agent):
    def _setAgentTypeID(self):
        self.type_id = 1
        return

    def getScore(self, cell):
        return cell.score

    def build(self):
        self.cell.type_color_rgb = (1,1,0)
        self.cell.type = 1
        self.cell.setDeveloped()
        return

class IndustrialBuilder(Agent):
    def _setAgentTypeID(self):
        self.type_id = 2
        return

    def getScore(self, cell):
        return cell.score

    def build(self):
        self.cell.type_color_rgb = (1,0,0)
        self.cell.type = 2
        self.cell.setDeveloped()
        return