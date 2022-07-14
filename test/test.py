import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
import time

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autoplanner as ap

def plotGraph(net):
    G=nx.DiGraph(directed=True)
    for n in net.nodes.values():
        G.add_node(n.label, pos=n.position)
    for e in net.edges.values():
        G.add_edge(e.from_node.label, e.to_node.label, weight=np.round(e.lenght,2),key=e.label)

    plt.figure(3,figsize=(12,12))
    nx.draw(G, pos=nx.get_node_attributes(G,'pos'),
            node_size=1, edge_color='black', node_color='red', width=1, connectionstyle='arc3, rad = 0.001',)
    nx.draw_networkx_edge_labels(G,nx.get_node_attributes(G,'pos'),edge_labels=nx.get_edge_attributes(G,'weight'), font_size=6)
    plt.show()



net = ap.net.loadNetSumo("data/usp.net.xml", geometry_nodes=True, v0=5)
w, h = net.boundary[-2:]

land = ap.land.Grid(w, h, 10, network=net, walking_weight_from=2, walking_weight_to=2, searched_nodes=50)

world = ap.World(land, agents=100, agent_view_radius=3)
world.run(max_iter=1)
world.show(land_use=False, agents=False, accessibility=True, map_links=False)
