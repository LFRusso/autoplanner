import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os,sys
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import autoplanner as ap

net = ap.net.loadNetSumo("data/map2.net.xml", geometry_nodes=True, v0=5)
w, h = net.boundary[-2:]


W = {"Whh": 1, "Whc": 2, "Whi": -3, "Whr": 6, 
          "Wch": 3, "Wcc": 4,
          "Wii": 5}
land = ap.land.Grid(w, h, 21, network=net, walking_weight_from=1, 
                    walking_weight_to=1, searched_nodes=50, K=5, weights=W)

world = ap.World(land, view_radius=5)

mode = 2
if mode==0: #Train
    loaded_model = load_model('models/target_model_nopool')
    #loaded_model = None
    model, target = world.run(episodes=200, steps=5000, epsilon=1, epsilon_decay=.99, min_epsilon=0.001, model=loaded_model)
    model.save('models/model_automove_nopool')
    target.save('models/target_model_nopool')
    world.show(land_use=True, agent=False, accessibility=False, map_links=False, color=True)

elif mode==1: # Test
    loaded_model = load_model('models/target_model_nopool')
    model, target = world.run(episodes=1, steps=10000, epsilon=0, epsilon_decay=1, min_epsilon=0.001, model=loaded_model)
    world.show(land_use=True, agent=False, accessibility=False, map_links=False, color=True)

elif mode==2: # Max imediate reward
    world.runGreedy(steps=100000)
    world.show(land_use=True, agent=False, accessibility=False, map_links=False, color=True)

elif mode==3: # Random
    world.runRandom(steps=100000)
    world.show(land_use=True, agent=False, accessibility=False, map_links=False, color=True)
    
elif mode==4: # Write
    world.runGreedy(steps=10000)
    world.show(land_use=True, agent=False, accessibility=False, map_links=False, color=True)
    R = world.reward()
    world.saveMap("map.dat")
    
elif mode==5: # Read
    world.loadMap("map.dat")
    world.show(land_use=True, agent=False, accessibility=False, map_links=False, color=True)
    R = world.reward()
    print(R)