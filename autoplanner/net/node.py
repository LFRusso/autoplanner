class Node:
    def __init__(self, pos, label, geometry_point=False):
        self.x = pos[0]
        self.y = pos[1]
        self.label = label
        self.position = pos

        self.out_edges = []
        self.in_edges = []
        return

    def setOutEdge(self, edge):
        self.out_edges.append(edge)
        return
    
    def setInEdge(self, edge):
        self.in_edges.append(edge)
        return