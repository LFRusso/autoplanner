import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os,sys
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autoplanner as ap

MODE = 1

if MODE==0: # SAVE
    net = ap.net.loadNetSumo("data/map2.net.xml", geometry_nodes=True, v0=5)
    w, h = net.boundary[-2:]

    W = {"Whh": 1, "Whc": 2, "Whi": -3, "Whr": 6, 
            "Wch": 3, "Wcc": 4,
            "Wii": 5}
    land = ap.land.Grid(w, h, 21, network=net, walking_weight_from=1, 
                        walking_weight_to=1, searched_nodes=50, K=5, weights=W)

    world = ap.World(view_radius=5)
    world.add_map(land)
    world.runGreedy(steps=10000)
    world.save("grid_map")
    #world.show(land_use=True, agent=False, accessibility=False, map_links=False, color=True)
    
elif MODE==1: # LOAD:
    world = ap.World(view_radius=5)
    world.load("maps/usp_map")
    world.show(land_use=True, agent=False, accessibility=False, map_links=False, color=True)
