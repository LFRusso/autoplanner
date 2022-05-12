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
        self.edge_travel_time = {}

        # Number of lines and columns based on cell size
        self.lines = int(np.ceil(height/cell_size))
        self.columns = int(np.ceil(width/cell_size))
        
        # Initializing cells
        self.cells = np.empty((self.lines, self.columns), dtype=Cell)
        for i in range(self.lines):
            for j in range(self.columns):
                self.cells[i,j] = Cell((j*cell_size + cell_size/2, i*cell_size + cell_size/2))
        for e in self.net.edges.values():
            self.edge_sets[e] = []
        self._getMeshDistances()
        self._getMeshTravelTimes()
        print(f"{self.lines*self.columns} ({self.lines}x{self.columns}) cells built in {time.time() - start}s")
        print("Calculating cell scores...")
        start = time.time()
        self._getCellScores()
        print(f"Finished calculating scores in {time.time()-start}s")
        return
        
    def plotGrid(self, links=False):
        max_score = min([c.score for c in self.cells.flatten()])
        for i in range(self.lines):
            for j in range(self.columns):
                plt.gca().add_patch(plt.Rectangle((j*self.cell_size, i*self.cell_size), 
                                    self.cell_size, self.cell_size, ec="gray", fc=(1.-max_score/self.cells[i,j].score,1.,1.-max_score/self.cells[i,j].score), alpha=.5))
        
        if (links):
            for i in range(self.lines):
                for j in range(self.columns):
                    plt.plot([self.cells[i,j].x, self.cells[i,j].linked_position[0]], [self.cells[i,j].y, self.cells[i,j].linked_position[1]])
        return

    def _getCellScores(self):
        flattened_cells = self.cells.flatten()
        scores = np.zeros(len(flattened_cells))
        sum_Te = sum(self.edge_travel_time.values())
        card_C = len(flattened_cells)
        for c in tqdm(flattened_cells, total=card_C):
            c_score = 0

            card_Cec = len(self.edge_sets[c.linked_edge])
            c_score += (card_Cec + 1) * c.mesh_distance
            c_score += self.distance(c.linked_position, c.linked_edge.to_node.position) / c.linked_edge.speed
            
            sum_terms = sum_Te - self.edge_travel_time[c.linked_edge]
            for e, e_set in self.edge_sets.items():
                if e != c.linked_edge:
                    term = self.net.dist_matrix[self.net.node_map[c.linked_edge.to_node.label], self.net.node_map[e.from_node.label]]
                    if (term != np.Inf and term != 0):
                        term *= len(e_set)
                        sum_terms += term
       
            sum_terms = ( 1/(card_C - card_Cec) ) * sum_terms
            c_score += sum_terms
            c.setScore(c_score)
        return

    def _getMeshTravelTimes(self):
        for e in self.net.edges.values():
            Ce = self.edge_sets[e]
            Te = np.sum([self.distance(e.from_node.position, c.linked_position)/e.speed + c.mesh_distance for c in Ce])
            self.edge_travel_time[e] = Te
        return

    def _getMeshDistances(self): # Approximation using closest node
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