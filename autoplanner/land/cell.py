class Cell:
    def __init__(self, pos, idx):
        self.x = pos[0]
        self.y = pos[1]
        self.position = pos
        self.idx = idx
        
        self.score = 0
        self.avg_travel_time = 0

        self.undeveloped = True
        self.type = -1
        self.type_color_rgb = (1,1,1)
        return

    # Links the cell to the road network by a given point and the edge it is contained in (by default closest point to the cell)
    def setMeshLink(self, P, e, d=None):
        self.mesh_distance = d
        if d==None:
            self.mesh_distance = self.distance(P, self.position)
        
        self.linked_position = P
        self.linked_edge = e
        return

    def setAvgTime(self, avg_time):
        self.avg_travel_time = avg_time
        return

    def setScore(self, score):
        self.score = score
        return

    def distance(self, u, v):
        return np.linalg.norm(np.array(u)-np.array(v))

    def setDeveloped(self):
        self.undeveloped = False
        return