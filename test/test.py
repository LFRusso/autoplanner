import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autoplanner as ap

net = ap.net.loadNetSumo("data/usp.net.xml", geometry_nodes=True, v0=5)
w, h = net.boundary[-2:]

land = ap.land.Grid(w, h, 20, network=net, walking_weight_from=2, walking_weight_to=2, searched_nodes=50)

world = ap.World(land)
world.run(episodes=5, steps=2000, epsilon=1, epsilon_decay=0.9, min_epsilon=0.001)
world.show(land_use=True, agent=True, accessibility=False, map_links=False)