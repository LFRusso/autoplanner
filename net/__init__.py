import sumolib
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall, shortest_path, dijkstra
import networkx as nx
import time

from .node import Node
from .edge import Edge

class Net:
    def __init__(self, offset, boundary, projection_parameters):
        self.offset = offset
        self.boundary = boundary
        self.projection = pyproj.Proj(projparams=projection_parameters)

        self.nodes = {}
        self.edges = {}
        self.node_map = {}
        self.graph = []
        self.dist_matrix = []
        return
        
    def addNode(self, pos, label, geometry_point):
        self.nodes[label] = Node(pos, label, geometry_point)
        self.node_map[label] = len(self.node_map) 
        return

    def addEdge(self, from_node, to_node, lenght, speed, label):
        self.edges[label] = Edge(from_node, to_node, lenght, speed, label)
        from_node.setOutEdge(self.edges[label])
        to_node.setInEdge(self.edges[label])        
        self.graph.append([self.node_map[from_node.label], self.node_map[to_node.label], lenght/speed])
        return
    
    def getDistMatrix(self):
        [row, col, data] = np.transpose(self.graph)
        matrix_form = csr_matrix( (data, (row, col)), shape=( len(self.nodes), len(self.nodes) ) )
        
        self.dist_matrix = dijkstra(csgraph=matrix_form, directed=True, return_predecessors=False)
        return self.dist_matrix

    def XY2LonLat(self, x, y):
        x_off, y_off = self.offset
        x -= x_off
        y -= y_off
        return self.projection(x, y, inverse=True)

    def LonLat2XY(self, lon, lat):
        return self.projection(lon, lat)

    def plotNet(self):
        for e in self.edges.values():
            plt.plot([e.from_node.x, e.to_node.x], [e.from_node.y, e.to_node.y], color="black")
        plt.axis("equal")


def distance(u, v):
    return np.linalg.norm(np.array(u)-np.array(v))

def loadNetSumo(netfile):
    start = time.time()
    print("Building road network...")

    sumo_net = sumolib.net.readNet(netfile)
    net = Net(sumo_net.getLocationOffset(), sumo_net.getBoundary(), sumo_net._location["projParameter"])    
    
    node_objs = sumo_net.getNodes()
    edge_objs = sumo_net.getEdges()

    for n in node_objs:
        net.addNode(n.getCoord(), n.getID(), geometry_point=False)

    # Incorporating geometry points to the edge list
    removed_edges = []
    for e in edge_objs:
        shape = e.getRawShape()
        if (len(shape)>2):
            new_nodes_labels = [f"gnode_{e.getID()}_{i}" for i in range(len(shape[1:-1]))]
            for n_label, n_pos in zip(new_nodes_labels, shape[1:-1]):
                net.addNode(n_pos, n_label, geometry_point=True)

            lenghts = [distance(shape[i], shape[i+1]) for i in range(len(shape)-1)]
            speeds = [e.getSpeed() for i in range(len(shape)-1)]
            
            new_edges = [(new_nodes_labels[i],new_nodes_labels[i+1]) for i in 
                        range(len(new_nodes_labels)-1)]
            new_edges = [(e.getFromNode().getID(), new_nodes_labels[0])] + new_edges +[(new_nodes_labels[-1], e.getToNode().getID())]
            
            for i in range(len(new_edges)):
                net.addEdge(net.nodes[new_edges[i][0]], net.nodes[new_edges[i][1]], lenghts[i],
                            speeds[i], f"gedge_{e.getID()}_{i}")
            
            removed_edges.append(e)
    edge_objs = [e for e in edge_objs if e not in removed_edges]

    for e in edge_objs:
        net.addEdge(net.nodes[e.getFromNode().getID()], net.nodes[e.getToNode().getID()],
                    e.getLength(), e.getSpeed(), e.getID())
    print(f"Building road network completed in {time.time() - start}s!")
    
    start = time.time()
    print(f"{net.boundary[-2]}m x {net.boundary[-1]}m")
    print(len(net.edges.values()), "edges")
    print(len(net.nodes.values()), "nodes")
    print("Building distances matrix..")

    net.getDistMatrix()
    print(f"Building distances matrix completed in {time.time() - start}s!")
    return net