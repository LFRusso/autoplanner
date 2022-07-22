import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
from tqdm import tqdm

from .cell import Cell

class Grid:
    def __init__(self, width, height, cell_size, network=None, walking_weight_from=1, walking_weight_to=1, searched_nodes=1):
        start = time.time()
        print("Building cells...")
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.net = network
        self.cell_coords = []
        self.edge_sets = {}
        self.edge_travel_time = {}
        self.walking_weight_from = walking_weight_from
        self.walking_weight_to = walking_weight_to


        # Number of lines and columns in the grid based on cell size
        self.lines = int(np.ceil(height/cell_size))
        self.columns = int(np.ceil(width/cell_size))
        
        # Initializing cells
        self.cells = np.empty((self.lines, self.columns), dtype=Cell)
        for i in range(self.lines):
            for j in range(self.columns):
                self.cells[i,j] = Cell(cell_size, (j*cell_size + cell_size/2, i*cell_size + cell_size/2), (i,j)) # 'Position' of the cell is set in its center
        for e in self.net.edges.values():
            self.edge_sets[e] = []
        self._getMeshDistances(searched_nodes) # Build edge_sets (cell neighborhoods) based on distance of each sell to each edge
        self._getMeshTravelTimes()
        print(f"{self.lines*self.columns} ({self.lines}x{self.columns}) cells built in {time.time() - start}s")
        print("Calculating cell scores...")
        start = time.time()
        self._getCellScores()
        print(f"Finished calculating scores in {time.time()-start}s")
        return
        
    def plotGrid(self, accessibility=False, links=False):
        max_score = max([c.score for c in self.cells.flatten()])
        if (accessibility):
            for i in range(self.lines):
                for j in range(self.columns):
                    plt.gca().add_patch(plt.Rectangle((j*self.cell_size, i*self.cell_size), 
                                        self.cell_size, self.cell_size, ec="gray", fc=(1.-self.cells[i,j].score/max_score,1.,1.-self.cells[i,j].score/max_score), alpha=.5))
        else:
            for i in range(self.lines):
                for j in range(self.columns):
                    plt.gca().add_patch(plt.Rectangle((j*self.cell_size, i*self.cell_size), 
                                        self.cell_size, self.cell_size, ec="gray", fc=(1,1,1), alpha=.5))

        if (links):
            for i in range(self.lines):
                for j in range(self.columns):
                    plt.plot([self.cells[i,j].x, self.cells[i,j].linked_position[0]], [self.cells[i,j].y, self.cells[i,j].linked_position[1]])
                    #plt.text(self.cells[i,j].x, self.cells[i,j].y, f"{round(self.cells[i,j].score, 2)}", fontsize="x-small")

        return

    def _getCellScores(self):
        flattened_cells = self.cells.flatten()
        scores = np.zeros(len(flattened_cells))
        sum_Te = sum(self.edge_travel_time.values())
        card_C = len(flattened_cells)
        set_sizes = np.array([len(e_set) for e_set in self.edge_sets.values()])
        for c in tqdm(flattened_cells, total=card_C):
            c_avg_time = 0

            card_Cec = len(self.edge_sets[c.linked_edge])
            c_avg_time += card_C * c.mesh_distance**self.walking_weight_from
            c_avg_time += (card_C - card_Cec) * self.distance(c.linked_position, c.linked_edge.to_node.position) / c.linked_edge.speed
            
            sum_terms = sum_Te
            net_distances = np.array([self.net.dist_matrix[self.net.node_map[c.linked_edge.to_node.label], self.net.node_map[e.from_node.label]] for e in self.net.edges.values()])
            sum_terms += (set_sizes*net_distances).sum()
            c_avg_time += sum_terms
            
            c_avg_time -= len(self.edge_sets[c.linked_edge]) * self.net.dist_matrix[self.net.node_map[c.linked_edge.to_node.label], self.net.node_map[c.linked_edge.from_node.label]]

            # Slowest part 
            c_avg_time += sum([self.distance(c.linked_position, ci.linked_position) / c.linked_edge.speed - self.distance(ci.linked_position, c.linked_edge.from_node.position) / c.linked_edge.speed for ci in self.edge_sets[c.linked_edge]])
            
            c_avg_time = c_avg_time/card_C
            c.setAvgTime(c_avg_time)
            c.setScore(1/c_avg_time)
        return

    # Total travel time spent inside a given edge when travelling to all cells linked to it 
    def _getMeshTravelTimes(self):
        for e in self.net.edges.values():
            Ce = self.edge_sets[e]
            Te = np.sum([self.distance(e.from_node.position, c.linked_position)/e.speed + c.mesh_distance**self.walking_weight_to/self.net.v0 for c in Ce])
            self.edge_travel_time[e] = Te
        return

    # Calculates the distance of each cell in relation to the network by finding the closest edge to its center
    def _getMeshDistances(self, n_nodes=1): 
        print("Calculating cell-network distances...")
        flattened_cells = self.cells.flatten()
        cell_coords = [c.position for c in flattened_cells]
        node_coords = [n.position for n in self.net.nodes.values()]

        # Tries to reduce the searched edges by looking only at those that connect the closest nodes
        node_tree = cKDTree(node_coords) 
        _, idx = node_tree.query(cell_coords, k=n_nodes, workers=-1)
        
        selected_nodes = np.array(list(self.net.nodes.values()))[idx]
        for c, n_list in tqdm(zip(flattened_cells, selected_nodes), total=len(flattened_cells)):
            if(isinstance(n_list, np.ndarray)): # When only one node is selected (n_nodes=1), the returned structure is not an array
                neighboring_edges = []
                for n in n_list:
                    neighboring_edges += n.out_edges + n.in_edges
            else:
                neighboring_edges = n_list.out_edges + n_list.in_edges

            neighboring_edges = set([e for e in neighboring_edges if 'rev_' not in e.label]) # Filtering out reverse edges made for the sake of making sure graph is strongly connected
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
        edge_list = list(edges)
        
        edgeDistanceV = np.vectorize(self.edgeDistance)
        D = edgeDistanceV(edge_list, cell)
        index = np.argmin(D[2]) 
        return [*np.transpose(D)[index], edge_list[index]] 

    def distance(self, u, v):
        return np.linalg.norm(np.array(u)-np.array(v))

    def cell2CellDist(self, cell_i, cell_j):

        return