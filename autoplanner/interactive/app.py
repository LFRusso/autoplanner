import sys
import os
from flask import Flask, render_template, url_for, request, redirect, flash, jsonify
import json
import urllib.parse as parse
import requests
import numpy as np

sys.path.append('../../')
import autoplanner as ap


app = Flask(__name__)
world = ap.World(view_radius=5)
world.load("maps/usp_map")


def run_in_browser():
    app.run(host="0.0.0.0", debug=True, port=5000)
    return

@app.route('/', methods=["GET"])
def index():
    return render_template("index.html", len=0)

@app.route('/edit', methods=["GET", "POST"])
def editor():
    land_uses = [c.type for line in world.map.cells for c in line]
    scores = [c.score for line in world.map.cells for c in line]
    accessibilities = [c.accessibility for line in world.map.cells for c in line]
    norm_accessibilities = [c.norm_accessibility for line in world.map.cells for c in line]
    mesh_distances = [c.mesh_distance for line in world.map.cells for c in line]
    view_radius = [c.K for line in world.map.cells for c in line]
    weights = world.map.cells[0,0].W
    [height, width] = world.map.cells.shape
    return render_template("editor.html", map=land_uses, scores=scores, accessibilities=accessibilities,
                            norm_accessibilities=norm_accessibilities, mesh_distances=mesh_distances,
                            view_radius=view_radius, weights=weights, width=width, height=height, cell_size=10)
