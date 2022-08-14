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
    def __init__(self, offset, boundary, projection_parameters, v0=5):
        self.offset = offset
        self.boundary = boundary
        self.projection = pyproj.Proj(projparams=projection_parameters)
        self.v0 = v0

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
        new_edge = Edge(from_node, to_node, lenght, speed, label)
        if (new_edge in self.edges.values()):
            return

        self.edges[label] = new_edge
        from_node.setOutEdge(self.edges[label])
        to_node.setInEdge(self.edges[label])        
        self.graph.append([self.node_map[from_node.label], self.node_map[to_node.label], lenght/speed])
        return
    
    def getDistMatrix(self):
        [row, col, data] = np.transpose(self.graph)
        matrix_form = csr_matrix( (data, (row, col)), shape=( len(self.nodes), len(self.nodes) ) )
        
        self.dist_matrix = dijkstra(csgraph=matrix_form, directed=True, return_predecessors=False)

        return self.dist_matrix

    # Adds auxiliary edges to ensure for all vertices v_i, v_j in V, if e_m = (v_i, v_j) is an edge,
    # there has to be an edge of opposite direction e_n = (v_j, v_i) (network is strongly connected)
    def enforceStronglyConnected(self):
        for vi in self.nodes.values():
            for em in vi.out_edges:
                vj = em.to_node
                found_return = False
                for en in vj.out_edges:
                    if (en.to_node == vi): # Return edge found, nothing added
                        found_return = True
                # No return edge found, adding
                if (found_return == False):
                    self.addEdge(vj, vi, em.lenght, self.v0, f"rev_{em.label}")

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


def loadNetSumo(netfile, geometry_nodes=True, v0=1):
    start = time.time()
    print("Building road network...")

    sumo_net = sumolib.net.readNet(netfile)
    net = Net(sumo_net.getLocationOffset(), sumo_net.getBoundary(), sumo_net._location["projParameter"], v0=v0)    
    
    node_objs = sumo_net.getNodes()
    edge_objs = sumo_net.getEdges()

    # All original nodes are added
    for n in node_objs:
        net.addNode(n.getCoord(), n.getID(), geometry_point=False)

    # Incorporating geometry points to the edge list 
    # Edges with geometry are replaced with a set of edge and geometry points are added
    removed_edges = []
    if geometry_nodes:
        for e in edge_objs:
            shape = e.getRawShape()
            if (len(shape)>2): # contains more than one line segment

                # Points of the edge geometry are added as new geometry nodes
                new_nodes_labels = [f"gnode_{e.getID()}_{i}" for i in range(len(shape[1:-1]))]
                for n_label, n_pos in zip(new_nodes_labels, shape[1:-1]):
                    net.addNode(n_pos, n_label, geometry_point=True)

                # Creating new edges by dividing the pre existing edge with geometry
                # New edges have only one line segment
                lenghts = [np.linalg.norm(np.array(shape[i]) - np.array(shape[i+1])) for i in range(len(shape)-1)]
                speeds = [e.getSpeed() for i in range(len(shape)-1)]
                new_edges = [(new_nodes_labels[i],new_nodes_labels[i+1]) for i in 
                            range(len(new_nodes_labels)-1)]
                new_edges = [(e.getFromNode().getID(), new_nodes_labels[0])] + new_edges +[(new_nodes_labels[-1], e.getToNode().getID())]
                
                for i in range(len(new_edges)):
                    net.addEdge(net.nodes[new_edges[i][0]], net.nodes[new_edges[i][1]], lenghts[i],
                                speeds[i], f"gedge_{e.getID()}_{i}")
                
                # After being splitted, original edge is removed
                removed_edges.append(e)
    edge_objs = [e for e in edge_objs if e not in removed_edges]

    for e in edge_objs:
        net.addEdge(net.nodes[e.getFromNode().getID()], net.nodes[e.getToNode().getID()],
                    e.getLength(), e.getSpeed(), e.getID())

    net.enforceStronglyConnected()
    print(f"Building road network completed in {time.time() - start}s!")
    
    start = time.time()
    print(f"{net.boundary[-2]}m x {net.boundary[-1]}m")
    print(len(net.edges.values()), "edges")
    print(len(net.nodes.values()), "nodes")
    print("Building distances matrix..")

    net.getDistMatrix()
    print(f"Building distances matrix completed in {time.time() - start}s!")
    return net