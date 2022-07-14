import numpy as np

class Agent:
    def __init__(self, cell, view_radius):
        self.cell = cell
        self.view_radius = view_radius
        self.x, self.y = cell.idx[0], cell.idx[1]
        return
    
    # Execute explore-build behavior
    def interact(self, world):
        neighborhood = self.lookAround(world.map).flatten()
        next_cell = self.getNextCell(neighborhood)

        if (next_cell==self.cell):
            # build
            self.cell.setDeveloped()
        else:
            # move to next
            self.moveTo(next_cell)
        return

    # Selece neighboring cells from the one the agent is currently on
    def lookAround(self, map):
        close_cells = map.cells[max(self.x-self.view_radius,0) : min(self.x+self.view_radius+1, map.lines),
                                max(self.y-self.view_radius,0) : min(self.y+self.view_radius+1, map.columns)]
        return close_cells
    
    # Check if there is a more proffitable, undeveloped cell within the list
    def getNextCell(self, cell_list):
        cell_list = [c for c in cell_list if c.undeveloped]
        next_cell = cell_list[0]
        for c in cell_list:
            if (self.getCellProffit(c) > self.getCellProffit(next_cell)):
                next_cell = c
        return next_cell

    def getCellProffit(self, cell):
        return cell.score


    def moveTo(self, cell):
        self.cell = cell
        self.x, self.y = cell.idx[0], cell.idx[1]
        return