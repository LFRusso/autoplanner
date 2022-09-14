import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoplanner.interactive import run_in_browser
#import autoplanner as ap

run_in_browser()

'''
net = ap.net.loadNetSumo("data/usp.net.xml", geometry_nodes=True, v0=5)
w, h = net.boundary[-2:]


W = {"Whh": 1, "Whc": 2, "Whi": -3, "Whr": 6, 
          "Wch": 3, "Wcc": 4,
          "Wii": 5}
land = ap.land.Grid(w, h, 21, network=net, walking_weight_from=1, 
                    walking_weight_to=1, searched_nodes=50, K=5, weights=W)

world = ap.World(land, view_radius=5)
'''