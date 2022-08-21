import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autoplanner as ap

net = ap.net.loadNetSumo("data/usp.net.xml", geometry_nodes=True, v0=5)
w, h = net.boundary[-2:]


W = {"Whh": 2, "Whc": 1, "Whi": -3, "Whr": 2, 
          "Wch": 3, "Wcc": 3,
          "Wii": 2}
land = ap.land.Grid(w, h, 20, network=net, walking_weight_from=1, 
                    walking_weight_to=1, searched_nodes=50, K=5, weights=W)

world = ap.World(land)
model, target = world.run(episodes=100, steps=100, epsilon=1, epsilon_decay=0.95, min_epsilon=0.001)
#model.save('model_1')
#target.save('target_model_1')

world.show(land_use=True, agent=True, accessibility=False, map_links=False)
