class Cell:
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.position = pos
        self.score = 0
        return

    def setMeshLink(self, P, e, d=None):
        self.mesh_distance = d
        if d==None:
            self.mesh_distance = self.distance(P, self.position)
        
        self.linked_position = P
        self.linked_edge = e
        return

    def setScore(self, score):
        self.score = score
        return

    def distance(self, u, v):
        return np.linalg.norm(np.array(u)-np.array(v))