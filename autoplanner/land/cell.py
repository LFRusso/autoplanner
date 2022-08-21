from numpy import sqrt, exp

class Cell:
    def __init__(self, size, pos, idx, K=5, weights="default"):
        self.x = pos[0]
        self.y = pos[1]
        self.position = pos
        self.idx = idx
        self.size = size
        self.K = K
        
        self.accessibility = 0
        self.norm_accessibility = 0
        self.score = 0
        self.avg_travel_time = 0

        self.undeveloped = True
        self.type = -1
        self.type_color_rgb = (1,1,1)

        if (weights == "default"):
            self.W = {"Whh": 1, "Whc": 1, "Whi": -1, "Whr": 1, 
                      "Wch": 1, "Wcc": 1,
                      "Wii": 1}
        else:
            self.W = weights

        return

    def _updateAccessibility(self):
        self.accessibility = 1 / self.avg_travel_time
        return

    # Links the cell to the road network by a given point and the edge it is contained in (by default closest point to the cell)
    def setMeshLink(self, P, e, d=None):
        self.mesh_distance = d
        if d==None:
            self.mesh_distance = self.distance(P, self.position)

        if d < sqrt(2*self.size**2)/2:
            self.setRoad()

        self.linked_position = P
        self.linked_edge = e
        return

    def setAvgTime(self, avg_time):
        self.avg_travel_time = avg_time
        return

    def distance(self, u, v):
        return np.linalg.norm(np.array(u)-np.array(v))

    def setRoad(self):
        self.type = 0
        self.type_color_rgb = (0,0,0) 
        self.undeveloped = False
        return

    def getResidentialScore(self, vicinity):
        flattened_vicinity = vicinity.flatten()

        res, com, ind, rec = 0, 0, 0, 0
        for c in flattened_vicinity:
            res += 1 if c.type==1 else 0
            com += 1 if c.type==2 else 0
            ind += 1 if c.type==3 else 0
            rec += 1 if c.type==4 else 0

        sh = self.norm_accessibility*(self.W["Whh"]*res + self.W["Whc"]*com + self.W["Whi"]*ind + self.W["Whr"]*rec)/self.K**2
        return sh

    def getCommercialScore(self, vicinity):
        flattened_vicinity = vicinity.flatten()

        res, com  = 0, 0
        for c in flattened_vicinity:
            res += 1 if c.type==1 else 0
            com += 1 if c.type==2 else 0

        commercial_balance = exp(-(com - res)**2/self.K**2) - exp(-res**2/self.K**2)
        sc = self.norm_accessibility*(self.W["Wch"]*res/self.K**2 + self.W["Wcc"]*commercial_balance)
        return sc
    
    def getIndustrialScore(self, vicinity):
        flattened_vicinity = vicinity.flatten()

        ind = 0
        for c in flattened_vicinity:
            ind += 1 if c.type==3 else 0

        si = self.norm_accessibility*(self.W["Wii"]*ind)/self.K**2
        return si


    # Updates data about current cell based on its type and vicinity
    def update(self, map):
        (i, j) = self.idx
        vicinity = map.cells[max(i-self.K,0) : min(i+self.K+1, map.lines),
                            max(j-self.K,0) : min(j+self.K+1, map.columns)]
        
        if self.type == -1: # Undeveloped
            self.type_color_rgb = (1,1,1)
            self.score = 0

        elif self.type == 0: # Road
            self.type_color_rgb = (0,0,0)
            self.score = 0

        elif self.type == 1: # Residential
            self.type_color_rgb = (0,0,1)
            self.score = self.getResidentialScore(vicinity)
 
        elif self.type == 2: # Commercial
            self.type_color_rgb = (1,1,0) 
            self.score = self.getCommercialScore(vicinity)

        elif self.type == 3: # Industrial
            self.type_color_rgb = (1,0,0) 
            self.score = self.getIndustrialScore(vicinity)

        elif self.type == 4: # Recreational
            self.type_color_rgb = (0,1,0) 
            self.score = 0
        
    def setUndeveloped(self):
        if (self.type==0):
            return
        self.type = -1
        self.undeveloped = True
        self.type_color_rgb = (1,1,1)
        self.score = 0

    # Builds in the current cell while changin its own and vicinity scores 
    def setDeveloped(self, dev_type, map):
        self.type = dev_type
        self.undeveloped = False

        (i, j) = self.idx
        vicinity = map.cells[max(i-self.K,0) : min(i+self.K+1, map.lines),
                            max(j-self.K,0) : min(j+self.K+1, map.columns)]

        self.update(map) # Updating current cell score
        for cell in vicinity.flatten(): # Updating vicinity score
            cell.update(map)
