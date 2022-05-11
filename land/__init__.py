import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
from tqdm import tqdm

from .cell import Cell

class Grid:
    def __init__(self, width, height, cell_size, network=None):
        start = time.time()
        print("Building cells...")
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.net = network
        self.cell_coords = []
        self.edge_sets = {}

        # Number of lines and columns based on cell size
        self.lines = int(np.ceil(height/cell_size))
        self.columns = int(np.ceil(width/cell_size))
        
        # Initializing cells
        self.cells = np.empty((self.lines, self.columns), dtype=Cell)
        for i in range(self.lines):
            for j in range(self.columns):
                self.cells[i,j] = Cell((j*cell_size + cell_size/2, i*cell_size + cell_size/2))
        self.getMeshDistances()
        print(f"{self.lines*self.columns} ({self.lines}x{self.columns}) cells built in {time.time() - start}s")
        return
        
    def plotGrid(self, links=False):
        for i in range(self.lines):
            for j in range(self.columns):
                plt.gca().add_patch(plt.Rectangle((j*self.cell_size, i*self.cell_size), 
                                    self.cell_size, self.cell_size, ec="gray", fc="white", alpha=.2))
        
        if (links):
            for i in range(self.lines):
                for j in range(self.columns):
                    plt.plot([self.cells[i,j].x, self.cells[i,j].linked_position[0]], [self.cells[i,j].y, self.cells[i,j].linked_position[1]])
        return

    def getMeshDistances(self): # Approximation using closest node
        print("Calculating cell-network distances...")
        flattened_cells = self.cells.flatten()
        cell_coords = [c.position for c in flattened_cells]
        node_coords = [n.position for n in self.net.nodes.values()]

        node_tree = cKDTree(node_coords)
        d, idx = node_tree.query(cell_coords, 1)
        
        selected_nodes = np.array(list(self.net.nodes.values()))[idx]
        for c, n in tqdm(zip(flattened_cells, selected_nodes), total=len(flattened_cells)):
            neighboring_edges = n.out_edges + n.in_edges
            x, y, d, e = self.closestEdgeDistance(c, neighboring_edges)
            
            if e not in self.edge_sets.keys():
                self.edge_sets[e] = [c]
            else:
                self.edge_sets[e].append(c)
            c.setMeshLink((x, y), e, d)
        return

    def edgeDistance(self, edge, cell):
        x, y = cell.position
        p1 = edge.from_node.position
        p2 = edge.to_node.position

        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2-x1, y2-y1
        det = dx*dx + dy*dy       
        a = ((x-x1)*dx + (y-y1)*dy) / det

        # Assuring points belong to the segment
        a = min(1, max(0, a))
        Px, Py = x1+a*dx, y1+a*dy
        d = self.distance((Px, Py), (x, y))
    
        return Px, Py, d

    def closestEdgeDistance(self, cell, edges): 
        x, y = cell.position
        Px, Py = -1, -1
        d = np.Inf
        edge_list = edges
        
        edgeDistanceV = np.vectorize(self.edgeDistance)
        D = edgeDistanceV(edge_list, cell)
        index = np.argmin(D[2]) 
        return [*np.transpose(D)[index], edge_list[index]] 

    def distance(self, u, v):
        return np.linalg.norm(np.array(u)-np.array(v))