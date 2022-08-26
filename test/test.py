import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os,sys
from tensorflow.keras.models import load_model

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

loaded_model = load_model('target_model_plain')
#loaded_model = None
model, target = world.run(episodes=1, steps=5000, epsilon=0.1, epsilon_decay=1, min_epsilon=0.001, model=loaded_model)
#model.save('model_plain')
#target.save('target_model_plain')

world.show(land_use=True, agent=True, accessibility=False, map_links=False)
