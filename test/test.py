import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autoplanner as ap

net = ap.net.loadNetSumo("data/usp.net.xml", geometry_nodes=True, v0=5)
w, h = net.boundary[-2:]

land = ap.land.Grid(w, h, 10, network=net, walking_weight_from=2, walking_weight_to=2, searched_nodes=50)

world = ap.World(land)
world.addRandomAgents('residential_builder', n=10, view_radius=3)
world.addRandomAgents('comercial_builder', n=5, view_radius=3)
world.addRandomAgents('industrial_builder', n=3, view_radius=3)
world.run(max_iter=400)
world.show(land_use=True, agents=False, accessibility=False, map_links=False)